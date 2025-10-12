import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cripser
import gudhi as gd
import time
import SimpleITK as sitk
import multiprocessing as mlp
import math
import cc3d

class TopologicalLoss(nn.Module):
    """
    Simple MONAI-style topological loss.

    What it does (per batch, per class):
      1) Runs persistent homology on the probability map (sublevel on 1 - p).
      2) Compares predicted features to a target:
         - If `match_gt=True`: uses PH of y_true for that class.
         - Else: enforces a fixed target Betti counts you specify (e.g., beta0=1).
      3) Builds two sparse maps:
         - weight_map: where to apply topology nudges (mostly zeros)
         - ref_map:    values to push toward (0, 1, or paired voxel prob)
      4) Loss = MSE on those sparse points only.

    Args:
        classes: list/tuple of class indices to enforce (e.g., organs) in y_pred[:, C, ...]
        match_gt: if True, match PH of y_true; else use `target_betti`.
        target_betti: dict {class_idx: {dim: count}}, e.g. {1: {0:1}} means “1 component” for class 1.
                      Ignored if match_gt=True. If not provided, defaults to {dim 0: 1}.
        dims_to_enforce: which homology dimensions to use. For 2D, typical is (0,1); for 3D, (0,1,2).
        reduction: 'mean' | 'sum' | 'none' for the batch-wise final loss.
        eps: numerical epsilon to avoid division by zero.
    """

    def __init__(
        self,
        classes,
        match_gt: bool = True,
        target_betti: dict | None = None,
        dims_to_enforce=(0,),
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.classes = list(classes)
        self.match_gt = match_gt
        self.target_betti = target_betti or {}  # {cls: {dim: count}}
        self.dims_to_enforce = tuple(dims_to_enforce)
        assert reduction in {"mean", "sum", "none"}
        self.reduction = reduction
        self.eps = eps

    # ------- Helpers: PH parsing & planning which features to keep/remove -------
    @staticmethod
    def _preprocess_ph(info: np.ndarray):
        """
        cripser.computePH returns rows:
        [dim, birth, death, bx, by, bz, dx, dy, dz]  (z fields are 0 for 2D)
        This groups them per dim and clips death<=1 (for numerical stability).
        """
        # clamp death > 1 to 1
        info = info.copy()
        info[info[:, 2] > 1, 2] = 1.0

        pd, bcp, dcp = {}, {}, {}
        dims = np.unique(info[:, 0]).astype(int)
        for d in dims:
            mask = info[:, 0] == d
            rows = info[mask]
            pd[str(d)] = rows[:, 1:3]  # (N,2)
            bcp[str(d)] = rows[:, 3:6]  # (N,3)
            dcp[str(d)] = rows[:, 6:9]  # (N,3)
        return pd, bcp, dcp

    @staticmethod
    def _keep_and_remove_indices(pd_pred: dict, pd_tgt: dict):
        """
        Decide which predicted features to keep (match to target count) and which to remove,
        per homology dim. Keep the most persistent ones.
        """
        idx_fix, idx_remove = {}, {}
        for d in pd_pred.keys():
            pers = np.abs(pd_pred[d][:, 1] - pd_pred[d][:, 0])
            order = np.argsort(pers)[::-1]  # descending persistence
            n_pred = len(pers)
            n_tgt = len(pd_tgt.get(d, [])) if d in pd_tgt else 0

            if n_pred > n_tgt:
                idx_fix[d] = order[:n_tgt]
                idx_remove[d] = order[n_tgt:]
            else:
                idx_fix[d] = order
                idx_remove[d] = np.array([], dtype=int)
        return idx_fix, idx_remove

    @staticmethod
    def _target_pd_from_betti(betti_counts: dict):
        """
        Build a fake target PD with [0,1] intervals, count times.
        Ex: { '0': 1 } -> array([[0,1]])
        """
        pd_tgt = {}
        for d_str, cnt in betti_counts.items():
            if cnt <= 0:
                continue
            pd_tgt[str(int(d_str))] = np.tile(np.array([[0.0, 1.0]]), (int(cnt), 1))
        return pd_tgt

    # ------- Core: make sparse maps for a single (B=1, C=1) probability map -------
    def _sparse_maps_for_one(
        self,
        prob_map: np.ndarray,  # shape: (Z,H,W) or (H,W)
        gt_map: np.ndarray | None,  # same shape, binary; optional
        spatial_dims: int,
    ):
        """
        Returns (weight_map, ref_map) same shape as prob_map.
        """
        # 1) predicted PH
        info_pred = cripser.computePH(1.0 - prob_map, maxdim=spatial_dims)
        pd_pred, bcp_pred, dcp_pred = self._preprocess_ph(info_pred)

        # 2) target PD
        if self.match_gt and gt_map is not None:
            info_gt = cripser.computePH(1.0 - gt_map, maxdim=spatial_dims)
            pd_tgt, _, _ = self._preprocess_ph(info_gt)
        else:
            # default to “one component” if nothing specified
            betti_cfg = {}
            for d in self.dims_to_enforce:
                # use user-specified betti if present; else default beta0=1, others 0
                cnt = 0
                if d == 0:
                    cnt = 1
                # allow override per class via target_betti at caller level
                betti_cfg[str(d)] = cnt
            pd_tgt = self._target_pd_from_betti(betti_cfg)

        # reduce to enforced dims only
        pd_pred = {
            str(d): pd_pred[str(d)]
            for d in pd_pred.keys()
            if int(d) in self.dims_to_enforce
        }
        pd_tgt = {
            str(d): pd_tgt[str(d)]
            for d in pd_tgt.keys()
            if int(d) in self.dims_to_enforce
        }

        # 3) decide keep/remove
        idx_fix, idx_remove = self._keep_and_remove_indices(pd_pred, pd_tgt)

        # 4) build sparse maps (push births toward 0, deaths toward 1; remove spurious by cross-tying)
        w = np.zeros_like(prob_map, dtype=np.float32)
        r = np.zeros_like(prob_map, dtype=np.float32)
        shape = prob_map.shape

        def in_bounds(pt):
            for i, m in enumerate(pt):
                if m < 0 or m >= shape[i]:
                    return False
            return True

        # NOTE: cripser returns (bx,by,bz); if 2D, bz will be 0—safe to index (Z,H,W) or (H,W)
        for d_str in idx_fix.keys():
            # keep: strengthen features -> push birth to 0, death to 1
            for k in idx_fix[d_str]:
                bpt = tuple(int(x) for x in bcp_pred[d_str][k])
                dpt = tuple(int(x) for x in dcp_pred[d_str][k])
                # trim to dimensionality
                bpt = bpt[-prob_map.ndim :]  # (H,W) or (Z,H,W)
                dpt = dpt[-prob_map.ndim :]

                if in_bounds(bpt):
                    w[bpt] = 1.0
                    r[bpt] = 0.0
                if in_bounds(dpt):
                    w[dpt] = 1.0
                    r[dpt] = 1.0

            # remove: tie birth↔death probabilities to annihilate short-lived features
            for k in idx_remove[d_str]:
                bpt = tuple(int(x) for x in bcp_pred[d_str][k])
                dpt = tuple(int(x) for x in dcp_pred[d_str][k])
                bpt = bpt[-prob_map.ndim :]
                dpt = dpt[-prob_map.ndim :]

                b_ok = in_bounds(bpt)
                d_ok = in_bounds(dpt)
                if b_ok and d_ok:
                    # set each to the other's current value
                    w[bpt] = 1.0
                    r[bpt] = prob_map[dpt]
                    w[dpt] = 1.0
                    r[dpt] = prob_map[bpt]
                elif b_ok and not d_ok:
                    w[bpt] = 1.0
                    r[bpt] = 1.0  # push birth up to kill spurious short feature
                elif d_ok and not b_ok:
                    w[dpt] = 1.0
                    r[dpt] = 0.0

        return w, r

    # ------- Public API -------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None):
        """
        y_pred: [B, C, H, W] or [B, C, D, H, W] — probabilities (after sigmoid/softmax).
        y_true: same shape, one-hot (0/1). Required if match_gt=True.
        """
        if y_pred.dim() not in (4, 5):
            raise ValueError("y_pred must be [B,C,H,W] or [B,C,D,H,W]")
        if self.match_gt and y_true is None:
            raise ValueError("match_gt=True requires y_true")

        B, C = y_pred.shape[:2]
        spatial_dims = (
            1 if y_pred.dim() == 4 else 2
        )  # cripser's maxdim: 2D -> 1 (β0,β1); 3D -> 2 (β0,β1,β2)
        # NOTE: cripser uses maxdim = number of homology dims (2D -> up to 1; 3D -> up to 2)
        maxdim = 1 if y_pred.dim() == 4 else 2
        if any(d > maxdim for d in self.dims_to_enforce):
            raise ValueError(
                f"dims_to_enforce {self.dims_to_enforce} incompatible with input dimensionality."
            )

        total = y_pred.new_tensor(0.0)
        per_item = []

        for b in range(B):
            loss_b = y_pred.new_tensor(0.0)
            count_terms = 0
            for cls in self.classes:
                prob = y_pred[b, cls]  # (H,W) or (D,H,W)
                gt = None
                if self.match_gt and y_true is not None:
                    gt = y_true[b, cls]

                # to numpy for PH
                prob_np = prob.detach().float().cpu().numpy()
                gt_np = gt.detach().float().cpu().numpy() if gt is not None else None

                # build sparse maps
                w_np, r_np = self._sparse_maps_for_one(
                    prob_np, gt_np, spatial_dims=maxdim
                )

                # cast back to torch
                w = torch.from_numpy(w_np).to(prob).float()
                r = torch.from_numpy(r_np).to(prob).float()

                # skip if nothing to update
                denom = w.sum()
                if denom <= self.eps:
                    continue

                # sparse MSE on critical voxels
                # normalize by number of updated points to keep scale stable across batches
                mse = F.mse_loss(prob * w, r, reduction="sum") / (denom + self.eps)
                loss_b = loss_b + mse
                count_terms += 1

            if count_terms > 0:
                loss_b = loss_b / count_terms
            per_item.append(loss_b)
            total = total + loss_b

        if self.reduction == "mean":
            return total / max(len(per_item), 1)
        elif self.reduction == "sum":
            return total
        else:
            return torch.stack(per_item)


class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40, last_step_thicken=True):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter
        self.last_step_thicken = last_step_thicken

    def soft_erode(self, img):

        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):

        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        
        if self.last_step_thicken:
            skel = self.soft_dilate(skel)
            
        return skel

    def forward(self, img):

        return self.soft_skel(img)


class SoftCIDice(nn.Module):
    def __init__(self, iter_=3, smooth=1.0, exclude_background=False):
        super(SoftCIDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2.0 * intersection + smooth) / (
        torch.sum(y_true) + torch.sum(y_pred) + smooth
    )
    return 1.0 - coeff


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.0, exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        dice = soft_dice(y_true, y_pred)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice


class topo_loss(nn.Module):
    def __init__(self, package="cripser"):
        super(topo_loss, self).__init__()
        """package = gudhi (8-connectivity in 2d) or cripser (4-connectivity in 2d) """
        self.package = package

    def compute_dgm_force_new(
        self,
        lh_dgm,
        gt_dgm,
        pers_thresh=0,
        pers_thresh_perfect=0.99,
        do_return_perfect=False,
    ):
        idx_fix_holes = {}
        idx_remove_holes = {}

        for dim in list(lh_dgm.keys()):
            idx_fix_holes.update({dim: []})
            idx_remove_holes.update({dim: []})
            dim_int = int(dim)
            lh_pers = abs(lh_dgm[dim][:, 1] - lh_dgm[dim][:, 0])
            lh_pers_idx_ranked = np.argsort(lh_pers)[::-1]
            lh_n = len(lh_pers)

            if dim in gt_dgm.keys():
                gt_pers = abs(gt_dgm[dim][:, 1] - gt_dgm[dim][:, 0])
                gt_n = len(gt_pers)
                assert np.array_equal(gt_pers, np.ones(gt_n))
            else:
                gt_pers = None
                gt_n = 0

            """the number of likelihood complex > gt: some of them fixed and some of them removed """
            if lh_n > gt_n:
                N_holes_2_fix = gt_n
                N_holes_2_remove = lh_n - gt_n

                idx_fix_holes.update({dim: lh_pers_idx_ranked[0:gt_n]})
                idx_remove_holes.update({dim: lh_pers_idx_ranked[gt_n::]})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix
                assert len(idx_remove_holes[dim]) == N_holes_2_remove
            elif lh_n <= gt_n:
                N_holes_2_fix = lh_n
                N_holes_2_remove = 0
                idx_fix_holes.update({dim: lh_pers_idx_ranked})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix

        return idx_fix_holes, idx_remove_holes

    # def recover(self, dim_eff):

    def pre_process(self, info):
        """info in shape [dim, b, d, b_x, b_y. b_z, d_x, d_y, d_z]"""

        revised_row_array = np.where(info[:, 2] > 1)[0]
        for row in revised_row_array:
            info[row, 2] = 1

        dim_eff_all = np.unique(info[:, 0])
        pd_gt_1 = {}
        bcp_gt_1 = {}
        dcp_gt_1 = {}

        for dim_eff in dim_eff_all:
            idx = info[:, 0] == dim_eff
            pd_gt_1.update({str(int(dim_eff)): info[idx][:, 1:3]})
            bcp_gt_1.update({str(int(dim_eff)): info[idx][:, 3:6]})
            dcp_gt_1.update({str(int(dim_eff)): info[idx][:, 6::]})

        return pd_gt_1, bcp_gt_1, dcp_gt_1

    def check_point_exist(self, r_max, p):
        result = True
        for i, point in enumerate(list(p)):
            if point < 0 or point >= r_max[i]:
                result = False
        return result

    def get_prediction_map(self, pd_lh, map, name):
        root = "/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/itn_5/betti_new/"
        threshold_born = 1 - pd_lh[0]
        threshold_death = 1 - pd_lh[1]

        map_b = torch.zeros(map.shape)
        map_d = torch.zeros(map.shape)
        map_b[map > threshold_born] = 1
        map_d[map > threshold_death] = 1
        self.save_nii(
            [
                root + x + name + "_p_" + str(threshold_born) + ".nii.gz"
                for x in [
                    "map_born_",
                ]
            ],
            map_b,
        )
        self.save_nii(
            [
                root + x + name + "_p_" + str(threshold_death) + ".nii.gz"
                for x in ["map_death_"]
            ],
            map_d,
        )

    def save_B_D_points(self, map, pd_lh, bcp_lh, dcp_lh, no=1000):
        result = np.zeros(map.shape)
        result_inv = np.zeros(map.shape)
        for i in range(no):
            coor_b = tuple(bcp_lh["0"][i].astype(np.int))
            coor_d = tuple(dcp_lh["0"][i].astype(np.int))
            # coor_b_inv = tuple(bcp_lh['0'][i].astype(np.int)[::-1])
            # coor_d_inv = tuple(dcp_lh['0'][i].astype(np.int)[::-1])
            result[coor_b] = i
            result[coor_d] = -1 * i

            # result_inv[coor_b_inv] = i
            # result_inv[coor_d_inv] = -1 *i
        self.save_nii(
            [
                "/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/dhcp_1122/betti_gudhi/points.nii.gz"
            ],
            #'/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/dhcp_1122/betti_gudhi/points_inv.nii.gz'],
            result,
        )
        # result_inv)

    """old version"""

    # def reidx_f_gudhi(self, idx, ori_shape):
    #     ''' given the original shape of a map [x_ori, y_ori, z_ori], and an index in the range of np.prod(ori_shape), find the coordinate index in original image'''
    #     ''' try two options:
    #         (1) first occupy ori_shape[0] then goes to ori_shape[1]...
    #         e.g., ori_shape = [2, 3, 4] and idx_max = 2*3*4-1 = 23, coordinate_max= [1, 2, 3]
    #         idx= 4, then return coordinate = [0, 2, 0]
    #         (2) first occupy ori_shape[-1] then goes to ori_shape[-1-1]...
    #         e.g., return coordinate = [0, 1, 0]
    #     '''
    #     re_idx = np.zeros(3, dtype=np.uint8)
    #     reidx_0 = np.array(np.unravel_index(idx, ori_shape, order='C'))
    #     reidx_1 = np.array(np.unravel_index(idx, ori_shape, order='F'))
    #     if len(ori_shape) == 3:
    #         # div_0 = ori_shape[1] * ori_shape[2]
    #         # re_idx[0] = int(idx // div_0)
    #         # mod_0 = idx % div_0
    #         #
    #         # re_idx[1] = int(mod_0 // ori_shape[1])
    #         # re_idx[2] = int(mod_0 % ori_shape[1])
    #         # assert idx == re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[1] + re_idx[2]
    #
    #         div_0 = ori_shape[1] * ori_shape[0]
    #         re_idx[2] = int(idx // div_0)
    #         mod_0 = idx % div_0
    #
    #         re_idx[1] = int(mod_0 // ori_shape[0])
    #         re_idx[0] = int(mod_0 % ori_shape[0])
    #         assert idx == re_idx[2] * ori_shape[1] * ori_shape[0] + re_idx[1] * ori_shape[0] + re_idx[0]
    #         assert (reidx_1 == re_idx).all(), 'the reindex is not following mode F'
    #
    #         if not (re_idx[0]<=ori_shape[0] and re_idx[1]<=ori_shape[1] and re_idx[2]<=ori_shape[2]):
    #             print('hi')
    #
    #     elif len(ori_shape) == 2:
    #         re_idx[0] = int(idx // ori_shape[1])
    #         re_idx[1] = int(idx % ori_shape[1])
    #         assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
    #         assert (reidx_1 == re_idx).all(), 'the reindex is not following mode C, unless the input image size is square'
    #
    #         # re_idx[0] = int(idx // ori_shape[0])
    #         # re_idx[1] = int(idx % ori_shape[0])
    #         # assert idx == re_idx[0] * ori_shape[0] + re_idx[1]
    #
    #     return re_idx
    def reidx_f_gudhi(self, idx, ori_shape):
        re_idx = np.zeros(3, dtype=np.uint16)
        reidx_0 = np.array(np.unravel_index(idx, ori_shape, order="C"))
        reidx_1 = np.array(np.unravel_index(idx, ori_shape, order="F"))
        if len(ori_shape) == 3:
            div_0 = ori_shape[1] * ori_shape[2]
            re_idx[0] = int(idx // div_0)
            mod_0 = idx % div_0

            re_idx[1] = int(
                mod_0 // ori_shape[2]
            )  # updated on 23/09/02 from ori_shape[1] to ori_shape[2]
            re_idx[2] = int(mod_0 % ori_shape[2])
            if (
                idx
                != re_idx[0] * ori_shape[1] * ori_shape[2]
                + re_idx[1] * ori_shape[2]
                + re_idx[2]
            ):  # updated on 23/09/02 from re_idx[1] * ori_shape[1] to re_idx[1] * ori_shape[2]
                print("hold on, wrong reidx")
            assert (
                reidx_0 == re_idx
            ).all(), "Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing."

            # div_0 = ori_shape[1] * ori_shape[0]
            # re_idx[2] = int(idx // div_0)
            # mod_0 = idx % div_0
            #
            # re_idx[1] = int(mod_0 // ori_shape[0])
            # re_idx[0] = int(mod_0 % ori_shape[0])
            # assert idx == re_idx[2] * ori_shape[1] * ori_shape[0] + re_idx[1] * ori_shape[0] + re_idx[0]
            # assert (reidx_1 == re_idx).all(), 'the reindex is not following mode F'
            #
            # if not (re_idx[0] <= ori_shape[0] and re_idx[1] <= ori_shape[1] and re_idx[2] <= ori_shape[2]):
            #     print('hi')

        elif len(ori_shape) == 2:
            re_idx[0] = int(idx // ori_shape[1])
            re_idx[1] = int(idx % ori_shape[1])
            assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
            assert (
                reidx_0 == re_idx
            ).all(), "Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing."

            # re_idx[1] = int(idx // ori_shape[0])
            # re_idx[0] = int(idx % ori_shape[0])
            # assert idx == re_idx[1] * ori_shape[0] + re_idx[0]
        return re_idx  # re_idx

    def get_info_gudhi(self, map):
        cc = gd.CubicalComplex(
            dimensions=map.shape[::-1], top_dimensional_cells=1 - map.flatten()
        )
        ph = cc.persistence()
        # betti_2 = cc.persistent_betti_numbers(from_value=1, to_value=0)
        x = cc.cofaces_of_persistence_pairs()

        """3.1 get birth and death point coordinate from gudhi, and generate info array"""
        info_gudhi = np.zeros((len(ph), 9))
        # x will lack one death point where the filtration is inf
        """3.1.1 manually write the inf death point"""
        reidx_birth_0 = self.reidx_f_gudhi(x[1][0][0], map.shape)
        if len(map.shape) == 2:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1]]
        elif len(map.shape) == 3:
            birth_filtration = (
                1 - map[reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2]]
            )

        info_gudhi[0, :] = [
            0,
            birth_filtration,
            1,
            reidx_birth_0[0],
            reidx_birth_0[1],
            reidx_birth_0[2],
            0,
            0,
            0,
        ]
        idx_row = 1
        for dim in range(len(x[0])):
            for idx in range(x[0][dim].shape[0]):
                idx_brith, idx_death = x[0][dim][idx]
                reidx_birth = self.reidx_f_gudhi(idx_brith, map.shape)
                reidx_death = self.reidx_f_gudhi(idx_death, map.shape)

                if len(map.shape) == 2:
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1]]
                elif len(map.shape) == 3:
                    birth_filtration = (
                        1 - map[reidx_birth[0], reidx_birth[1], reidx_birth[2]]
                    )
                    death_filtration = (
                        1 - map[reidx_death[0], reidx_death[1], reidx_death[2]]
                    )
                else:
                    assert False, "wrong input dimension!"

                info_gudhi[idx_row, :] = [
                    dim,
                    birth_filtration,
                    death_filtration,
                    reidx_birth[0],
                    reidx_birth[1],
                    reidx_birth[2],
                    reidx_death[0],
                    reidx_death[1],
                    reidx_death[2],
                ]
                idx_row += 1

        return info_gudhi

    def get_topo_loss(self, map, batch, cgm_idx):
        # info_gt = cripser.computePH(map, maxdim=3)  # dim, b, d, b_x, b_y. b_z
        # dims_gt, pd_gt, bcp_gt, dcp_gt = self.pre_process(info_gt)

        # ''' test_1 '''
        # t1 = time.time()
        # for i in range(100):
        #     cripser.computePH(1 - map, maxdim=3)
        # t2 = time.time() - t1
        # print('time1: ', t2)
        #
        # map_discrete = (map * 10).astype(np.int)/10
        # ''' test_2 '''
        # t1 = time.time()
        # for i in range(100):
        #     cripser.computePH(1 - map_discrete, maxdim=3)
        # t2 = time.time() - t1
        # print('time2: ', t2)
        #
        # map_discrete = (map * 100).astype(np.int)/100
        # ''' test_2 '''
        # t1 = time.time()
        # for i in range(100):
        #     cripser.computePH(1 - map_discrete, maxdim=3)
        # t2 = time.time() - t1
        # print('time3: ', t2)
        t1 = time.time()
        betti = np.zeros((1, 3))
        if self.package == "cripser":
            info_lh = cripser.computePH(1 - map, maxdim=3)
        elif self.package == "gudhi":
            info_lh = self.get_info_gudhi(map)
        else:
            assert False, print("wrong package to calculate topology")
        t2 = time.time()
        # print('1. calculatePH: ', t2-t1)

        # map_discrete = (map * 10).astype(np.int)/10
        # info_lh_discrete = cripser.computePH(1 - map_discrete, maxdim=3)

        # for i in range(10, -1, -1):
        #     map_threshold = np.zeros(map.shape)
        #     thre = i/10
        #     map_threshold[map >= thre] =1
        #     self.save_nii(['output/itn_8/threshold/' + str(thre) + '.nii.gz'], map_threshold)

        pd_lh, bcp_lh, dcp_lh = self.pre_process(info_lh)
        for i in range(3):
            betti[0, i] = len(pd_lh[str(i)]) if str(i) in list(pd_lh.keys()) else 0

        """check the topology with the top ranked ph length, only in dim 0"""
        deta = pd_lh["0"][:, 1] - pd_lh["0"][:, 0]
        rank = np.argsort(deta)
        # for i in range(5):
        #     self.get_prediction_map(pd_lh['0'][rank[-1 * i]], map, 'rank_' + str(i))

        """show top 10 coordinate points that has the minimal born and death (prob ->1)"""
        # self.save_B_D_points(map, pd_lh, bcp_lh, dcp_lh)
        # self.save_nii( ['/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/dhcp_1122/betti_gudhi/ori.nii.gz'], map)
        # no_gt = int(0.05* len(pd_lh['0'][:, 1] )) if int(0.05* len(pd_lh['0'][:, 1] ))>1 else 1
        no_gt = 1
        a = np.zeros([no_gt, 2])
        a[:, 1] = 1
        pd_gt = {"0": a}  # {'0': a, '2':a}        # np.array([[0, 1]])

        idx_holes_to_fix, idx_holes_to_remove = self.compute_dgm_force_new(
            pd_lh, pd_gt, pers_thresh=0
        )
        """
        topo_cp value map:
        0： background
        1, 2, 3：point to fix (born: force it to 0)
        4, 5, 6: point to fix (death: force it to 1)
        7, 8, 9: point to remove (born: force it to the prob of death)
        10,11,12: point to remove (death: force it to the prob of born)
        """
        topo_size = list(map.shape)
        topo_cp_weight_map = np.zeros(map.shape)
        topo_cp_ref_map = np.zeros(map.shape)

        for dim in ["0", "1", "2"]:
            dim_int = int(dim)
            if dim in list(idx_holes_to_fix.keys()):
                for hole_indx in idx_holes_to_fix[dim]:
                    if self.check_point_exist(topo_size, bcp_lh[dim][hole_indx]):
                        # if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_bb = (coor_b[0], coor_b[1], coor_b[2])
                        topo_cp_weight_map[coor_bb] = (
                            1  # push birth to 0 i.e. min birth prob or likelihood
                        )
                        topo_cp_ref_map[coor_bb] = 0

                        # self.get_prediction_map(pd_lh[dim][hole_indx], map)

                        # topo_cp[coor_bb] = 1 + dim_int

                    # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                    if self.check_point_exist(topo_size, dcp_lh[dim][hole_indx]):
                        # if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_dd = (coor_d[0], coor_d[1], coor_d[2])
                        # topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        # topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                        topo_cp_weight_map[coor_dd] = (
                            1  # push birth to 0 i.e. min birth prob or likelihood
                        )
                        topo_cp_ref_map[coor_dd] = 1

                        # topo_cp[coor_dd] = 4 + dim_int
            no_1 = 0
            no_2 = 0
            np_3 = 0
            if dim in list(idx_holes_to_remove.keys()):
                for hole_indx in idx_holes_to_remove[dim]:
                    coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_bb = (coor_b[0], coor_b[1], coor_b[2])

                    coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_dd = (coor_d[0], coor_d[1], coor_d[2])

                    b_exists = self.check_point_exist(topo_size, bcp_lh[dim][hole_indx])
                    d_exists = self.check_point_exist(topo_size, dcp_lh[dim][hole_indx])
                    if b_exists and d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_bb] = map[coor_dd]
                        topo_cp_ref_map[coor_dd] = map[coor_bb]
                        # topo_cp[coor_bb] = 7 + dim_int
                        # topo_cp[coor_dd] = 10 + dim_int
                        no_1 = no_1 + 1
                    elif b_exists and not d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_ref_map[coor_bb] = 1
                        # topo_cp[coor_bb] = 7 + dim_int
                        no_2 = no_2 + 1

                    elif not b_exists and d_exists:
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_dd] = 0
                        # topo_cp[coor_dd] = 10 + dim_int
                        no_3 = no_3 + 1
        t3 = time.time()
        # print('2. post processing: ', t3-t2)
        return topo_cp_weight_map, topo_cp_ref_map, betti, batch, cgm_idx

    def save_nii(self, name_list, *map_all):
        """map: [batch, x, y, z]"""
        for i, map in enumerate(map_all):
            if len(map.shape) == 4:
                map = map[0, ...]
            elif len(map.shape) == 5:
                map = map[0, 0, ...]

            if type(map) == torch.Tensor:
                map = map.cpu().detach().numpy()
            # elif type(map) == np.ndarray:
            #     map = map

            if map.dtype == np.bool:
                map = map.astype(np.float32)

            map = sitk.GetImageFromArray(map)
            sitk.WriteImage(map, name_list[i])

    def get_arg_map_range(self, argmap, batch, cgm_idx):
        # argmap = argmap.cpu().detach().numpy()
        x = None
        y = None
        z = None
        # for batch in range(argmap.shape[0]):
        index = np.argwhere(argmap == 1)
        if index.shape[0] != 0:
            z = [index[:, 0].min(), index[:, 0].max()]
            y = [index[:, 1].min(), index[:, 1].max()]
            x = [index[:, 2].min(), index[:, 2].max()]
        return z, y, x, batch, cgm_idx

    def onehot(self, map):
        """map: torch.tensor in shape [batch, c, x, y, z]"""
        map_argmax = torch.argmax(map, dim=1)
        map_argmax_np = map_argmax.cpu().detach().numpy()
        map_np = map.cpu().detach().numpy()
        labels_all = np.unique(map_argmax_np)
        labels_all.sort()
        lab_array_one_hot = np.zeros_like(map_np)
        for idx, lab in enumerate(labels_all):
            # dist[idx] = (lab_array == lab).astype(float).sum()
            lab_array_one_hot[:, idx, ...] = map_argmax_np == lab
        return lab_array_one_hot

    def forward(self, map, device, cgm_dims=None, topo_size=160, isBinary=False):
        """for the gt should be betti (0,0,1)"""
        if cgm_dims is not None:
            batch_size, _, xx, yy, zz = map.shape

            cgm_no = len(cgm_dims)
            topo_cp_weight_map = np.zeros((batch_size, cgm_no, xx, yy, zz))
            topo_cp_ref_map = np.zeros((batch_size, cgm_no, xx, yy, zz))
            betti = np.zeros((batch_size, cgm_no, 3))

            argmap = torch.argmax(map, dim=1)
            t1 = time.time()
            """step_1: calculate the sub-patch for calculating the betti"""
            """z in the size of [batch, cgm_no, 2]"""
            z = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            y = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            x = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            res_all_1 = []
            pool_1 = mlp.Pool(2 * batch_size)
            for batch in range(batch_size):
                for cgm_idx, cgm_dim in enumerate(cgm_dims):
                    # idx = np.s_[batch, cgm_dim, :]
                    mmap = (
                        (argmap[batch, ...] == cgm_dim).float().cpu().detach().numpy()
                    )
                    if mmap.max() != 1:
                        print("hi")
                    res = pool_1.apply_async(
                        func=self.get_arg_map_range, args=(mmap, batch, cgm_idx)
                    )
                    res_all_1.append(res)
                    #
            pool_1.close()
            pool_1.join()
            t2 = time.time()
            print_time = False
            if print_time:
                print("step1.1: ", t2 - t1)
            """get the result"""
            assert len(res_all_1) == batch_size * cgm_no
            for i, res in enumerate(res_all_1):
                z_t, y_t, x_t, batch, cgm_idx = res.get()
                assert batch * cgm_no + cgm_idx == i
                idx = np.s_[batch, cgm_idx, :]
                z[idx], y[idx], x[idx] = z_t, y_t, x_t

            t3 = time.time()
            if print_time:
                print("step1.2: ", t3 - t2)
            """step_2: calculate betti based on the sub-patch"""
            res_all_2 = []

            pool_2 = mlp.Pool(batch_size * cgm_no)  #
            for batch in range(batch_size):
                for cgm_idx, cgm_dim in enumerate(cgm_dims):
                    idx = np.s_[
                        batch,
                        cgm_dim,
                        z[batch, cgm_idx, 0] : z[batch, cgm_idx, 1],
                        y[batch, cgm_idx, 0] : y[batch, cgm_idx, 1],
                        x[batch, cgm_idx, 0] : x[batch, cgm_idx, 1],
                    ]
                    if isBinary:
                        map_onehot = self.onehot(map)
                        input = map_onehot[idx]
                    else:
                        input = map[idx].cpu().detach().numpy()
                    # input[input>0.5] = 1
                    # input[input<0.5] =0
                    res = pool_2.apply_async(
                        func=self.get_topo_loss, args=(input, batch, cgm_idx)
                    )
                    res_all_2.append(res)

            pool_2.close()
            pool_2.join()
            t4 = time.time()
            if print_time:
                print("step2.1: ", t4 - t3)
            """get the result"""
            assert len(res_all_2) == batch_size * cgm_no
            for i, res in enumerate(res_all_2):
                weight_map, ref_map, bbet, batch, cgm_idx = res.get()
                assert batch * cgm_no + cgm_idx == i
                idx = np.s_[
                    batch,
                    cgm_idx,
                    z[batch, cgm_idx, 0] : z[batch, cgm_idx, 1],
                    y[batch, cgm_idx, 0] : y[batch, cgm_idx, 1],
                    x[batch, cgm_idx, 0] : x[batch, cgm_idx, 1],
                ]
                (
                    topo_cp_weight_map[idx],
                    topo_cp_ref_map[idx],
                    betti[batch, cgm_idx, :],
                ) = (
                    weight_map,
                    ref_map,
                    bbet,
                )
            t5 = time.time()
            if print_time:
                print("step2.2: ", t5 - t4)
        else:
            assert (
                not isBinary
            ), "when cgm_dim is not given, the input is 4d tensor, not supprot argmax calculation"
            topo_cp_weight_map = np.zeros(map.shape)
            topo_cp_ref_map = np.zeros(map.shape)
            betti = np.zeros((map.shape[0], 3))
            for batch in range(map.shape[0]):
                for z in range(0, map.shape[1], topo_size):
                    for y in range(0, map.shape[2], topo_size):
                        for x in range(0, map.shape[3], topo_size):
                            if not isBinary:
                                map_patch = map[
                                    batch,
                                    z : min(z + topo_size, map.shape[1]),
                                    y : min(y + topo_size, map.shape[2]),
                                    x : min(x + topo_size, map.shape[3]),
                                ]
                            else:
                                map_argamax = map.unsqueeze(dim=1)
                                map_patch = map_argamax[
                                    batch,
                                    z : min(z + topo_size, map.shape[1]),
                                    y : min(y + topo_size, map.shape[2]),
                                    x : min(x + topo_size, map.shape[3]),
                                ]

                            # self.save_labelmap_thres(map_batch, 0.05)
                            (
                                topo_cp_weight_map[
                                    batch,
                                    z : min(z + topo_size, map.shape[1]),
                                    y : min(y + topo_size, map.shape[2]),
                                    x : min(x + topo_size, map.shape[3]),
                                ],
                                topo_cp_ref_map[
                                    batch,
                                    z : min(z + topo_size, map.shape[1]),
                                    y : min(y + topo_size, map.shape[2]),
                                    x : min(x + topo_size, map.shape[3]),
                                ],
                                bb,
                            ) = self.get_topo_loss(map_patch.cpu().detach().numpy())
                            betti[batch, :] = bb

        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float32).to(
            device
        )
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float32).to(device)

        loss_topo = torch.zeros(cgm_no, dtype=torch.float32).to(device)
        num_points_updating = 0
        for cgm_idx, cgm_dim in enumerate(cgm_dims):
            idx = np.s_[:, cgm_idx, ...]
            idx_for_map = np.s_[:, cgm_dim, ...]
            loss_topo[cgm_idx] = (
                1
                / (map.shape[0] * map.shape[2] * map.shape[3] * map.shape[4])
                * (
                    (
                        (map[idx_for_map] * topo_cp_weight_map[idx])
                        - topo_cp_ref_map[idx]
                    )
                    ** 2
                ).sum()
            )
            num_points_updating += topo_cp_weight_map[idx].sum()
        betti_return = betti.mean(axis=0)

        return (
            loss_topo,
            betti_return,
            num_points_updating
            / (
                map.shape[0]
                * map.shape[2]
                * map.shape[3]
                * map.shape[4]
                * len(cgm_dims)
            ),
        )


import torch
import torch.nn.functional as F

# -------- utilities --------


def soft_step(x, t, tau=0.1):
    # differentiable ~threshold
    return torch.sigmoid((x - t) / tau)


def reachability_soft(prob, seeds, t=0.45, steps=24):
    """
    Differentiable flood-fill:
    prob: [N,1,D,H,W] in [0,1]
    seeds: [N,K,1,D,H,W] one seed map per component (soft/binary)
    returns: [N,K,1,D,H,W] soft connected regions grown within prob>=t corridors
    """
    corridor = soft_step(prob, t, tau=0.1)  # in (0,1); soft corridor
    # Expand corridor to [N,1,D,H,W] -> [N,K,1,D,H,W] for broadcast
    corridor = corridor.unsqueeze(
        1
    )  # [N,1,1,D,H,W] if you prefer, but keep dims matching seeds
    R = seeds  # [N,K,1,D,H,W]
    R = R.squeeze(2)
    for _ in range(steps):
        # 6/26-neighborhood via 3x3x3 max-pool; stride=1 keeps shape
        R = F.max_pool3d(R, 3, stride=1, padding=1)
        R = R * corridor  # constrain growth to high-prob corridor
    R = R.unsqueeze(2)
    return R.clamp(0, 1)


def soft_dice(a, b, eps=1e-6):
    # a,b in [0,1], same shape
    num = 2.0 * (a * b).sum(dim=(2, 3, 4, 5))
    den = (a * a).sum(dim=(2, 3, 4, 5)) + (b * b).sum(dim=(2, 3, 4, 5)) + eps
    return (num / den).mean()


# -------- soft-LCC construction --------


def make_seeds(prob, K=4, mode="argmax_detached", gt_mask=None):
    """
    Returns seeds: [N,K,1,D,H,W] with small Gaussians or 1-vox seeds.
    - mode="gt_centers": sample K seeds from GT foreground (best in supervised)
    - mode="argmax_detached": top-K of prob with stop-grad
    """
    N, C, D, H, W = prob.shape
    device = prob.device
    seeds = torch.zeros((N, K, 1, D, H, W), device=device, dtype=prob.dtype)

    if mode == "gt_centers" and gt_mask is not None:
        # simple: random GT voxels as seeds
        for n in range(N):
            fg = (gt_mask[n, 0] > 0).nonzero(as_tuple=False)
            if fg.numel() == 0:
                continue
            idx = torch.randint(0, fg.shape[0], (K,), device=device)
            for k, ijk in enumerate(fg[idx]):
                z, y, x = ijk.tolist()
                seeds[n, k, 0, z, y, x] = 1.0
    else:
        # top-K of prob (detached indices)
        flat = prob.detach().view(N, -1)
        vals, idxs = torch.topk(flat, k=K, dim=1)  # [N,K]
        for n in range(N):
            for k in range(K):
                idx = idxs[n, k].item()
                z = idx // (H * W)
                y = (idx % (H * W)) // W
                x = idx % W
                seeds[n, k, 0, z, y, x] = 1.0

    return seeds


def soft_lcc_mask(prob, K=4, t=0.45, steps=24, temp=5.0, seeds=None, **seed_kwargs):
    """
    Build soft-LCC by growing K regions and selecting the largest with a softmax over sizes.
    Returns M in [0,1]: [N,1,D,H,W]
    """
    if seeds is None:
        seeds = make_seeds(prob, K=K, **seed_kwargs)  # [N,K,1,D,H,W]
    comps = reachability_soft(prob, seeds, t=t, steps=steps)  # [N,K,1,D,H,W]

    # Soft sizes per component
    sizes = comps.sum(dim=(2, 3, 4, 5))  # [N,K]

    # Soft selection weights over K comps (approx. "pick the largest")
    w = torch.softmax(temp * sizes, dim=1)  # higher temp -> sharper "argmax"

    # Weighted mixture => soft-LCC
    M = (w.view(w.shape[0], w.shape[1], 1, 1, 1, 1) * comps).sum(dim=1)  # [N,1,D,H,W]
    return M.clamp(0, 1)


# -------- final loss: Dice(pred, pred ⊙ soft-LCC) --------


def largest_component_dice_loss(
    prob,
    K=4,
    t=0.45,
    steps=24,
    temp=5.0,
    seeds=None,
    seed_mode="argmax_detached",
    gt_mask=None,
):
    """
    prob: [N,1,D,H,W] in [0,1]
    returns 1 - SoftDice(prob, prob * softLCC)
    """
    M = soft_lcc_mask(
        prob,
        K=K,
        t=t,
        steps=steps,
        temp=temp,
        seeds=seeds,
        mode=seed_mode if seeds is None else None,
        gt_mask=gt_mask,
    )
    dice = soft_dice(prob, prob * M)
    return 1.0 - dice

# ---- helpers you already have (unchanged) ----
def _as_batched_spacing(spacing, B, device, dtype):
    sp = torch.as_tensor(spacing, device=device, dtype=dtype)
    if sp.ndim == 1:
        sp = sp.unsqueeze(0).expand(B, -1)
    assert sp.shape == (B, 3)
    sx = sp[:, 0].view(B, 1, 1, 1, 1)  # spacing along D
    sy = sp[:, 1].view(B, 1, 1, 1, 1)  # spacing along H
    sz = sp[:, 2].view(B, 1, 1, 1, 1)  # spacing along W
    return sx, sy, sz

def gradients_3d_batched(x, sx, sy, sz):
    xpad = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    dx = (xpad[:, :, 2:, 1:-1, 1:-1] - xpad[:, :, 0:-2, 1:-1, 1:-1]) / (2.0 * sx)
    dy = (xpad[:, :, 1:-1, 2:, 1:-1] - xpad[:, :, 1:-1, 0:-2, 1:-1]) / (2.0 * sy)
    dz = (xpad[:, :, 1:-1, 1:-1, 2:] - xpad[:, :, 1:-1, 1:-1, 0:-2]) / (2.0 * sz)
    return dx, dy, dz

# ---- NEW: surface area when phi is NORMALIZED ([-1,1]) ----
def surface_area_from_sdf_normalized(phi_norm, spacing, eps_mm=None, tau_mm=None):
    """
    phi_norm: (B,1,D,H,W), normalized SDF in arbitrary units (e.g., [-1,1])
    spacing:  (3,) or (B,3) in mm/voxel, ordered (sD, sH, sW)
    eps_mm:   smoothing width ε in mm for delta; default ~ one voxel diagonal (per sample)
    tau_mm:   band half-width in mm used to estimate the scale; default ~ one voxel diagonal

    Returns: (B,) surface areas in mm^2
    """
    assert phi_norm.ndim == 5 and phi_norm.size(1) == 1
    B = phi_norm.size(0)
    device, dtype = phi_norm.device, phi_norm.dtype
    sD, sH, sW = _as_batched_spacing(spacing, B, device, dtype)
    voxel_vol = sD * sH * sW

    # default widths in mm
    vox_diag_mm = torch.sqrt(sD**2 + sH**2 + sW**2)  # (B,1,1,1,1)
    if eps_mm is None:
        eps_mm = vox_diag_mm  # ~1 voxel diagonal
    if tau_mm is None:
        tau_mm = vox_diag_mm  # band to estimate scale

    # Gradients of normalized SDF w.r.t. WORLD coords (mm)
    dx, dy, dz = gradients_3d_batched(phi_norm, sD, sH, sW)
    grad_mag_norm = torch.sqrt(
        dx * dx + dy * dy + dz * dz + 1e-12
    )  # ≈ 1/s near interface

    # Estimate per-sample scale s_hat so that |∇phi_mm| ≈ 1 in band:
    # Need the band in NORMALIZED units: tau_norm = tau_mm / s_hat (implicit).
    # We solve by fixed-point: start with s_hat=1, refine a couple times.
    s_hat = torch.ones((B, 1, 1, 1, 1), device=device, dtype=dtype)
    for _ in range(2):  # 2 iterations are enough in practice
        tau_norm = tau_mm / s_hat
        band = (phi_norm.abs() <= tau_norm).float()
        denom = band.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        g_mean = (grad_mag_norm * band).sum(
            dim=(2, 3, 4), keepdim=True
        ) / denom  # ≈ 1/s
        s_hat = (1.0 / g_mean).clamp(min=1e-6)

    # Either de-normalize phi, or keep normalized and scale epsilon; both are equivalent.
    # We'll KEEP phi normalized and scale epsilon: eps_norm = eps_mm / s_hat
    eps_norm = eps_mm / s_hat

    # Delta_ε*(phi_norm) * |∇phi_norm|
    delta = (eps_norm / (eps_norm**2 + phi_norm**2)) / math.pi
    sa = (delta * grad_mag_norm).sum(dim=(2, 3, 4)) * voxel_vol.squeeze(-1).squeeze(
        -1
    ).squeeze(-1)
    return sa.squeeze(1)  # (B,)

# volume function unchanged (expects mask/prob and spacing in mm)
def volume_from_mask_batched(mask_or_prob, spacing):
    assert mask_or_prob.ndim == 5 and mask_or_prob.size(1) == 1
    B = mask_or_prob.size(0)
    sD, sH, sW = _as_batched_spacing(
        spacing, B, mask_or_prob.device, mask_or_prob.dtype
    )
    voxel_vol = sD * sH * sW
    vol = mask_or_prob.sum(dim=(2, 3, 4)) * voxel_vol.squeeze(-1).squeeze(
        -1
    ).squeeze(-1)
    return vol.squeeze(1)

def loss_isoperimetric(
    area_mm2,
    volume_mm3,
    alpha=1e-6,
    beta=1e-6,
    reduction="mean",
    gated=False,
    pred_prob=None,
):
    # add small unit-consistent stabilizers
    a = area_mm2 + alpha
    v = volume_mm3 + beta
    # minimize A^3 / V^2  ==  minimize 3*log(A) - 2*log(V)
    per_sample = 3.0 * torch.log(a) - 2.0 * torch.log(v)

    if gated and (pred_prob is not None):
        # Count number of connected components in pred_prob
        B = pred_prob.shape[0]
        binv = (pred_prob > 0.5).float()

        gates = []
        for b in range(B):
            with torch.no_grad():
                binb = binv[b, 0].detach().cpu().numpy()
                lab = cc3d.connected_components(binb, connectivity=6)
                counts = np.bincount(lab.ravel())
                fg_counts = (
                    counts[1:]
                    if counts.size > 1
                    else np.array([], dtype=counts.dtype)
                )  # Ignore background count
                n_comp = int((fg_counts >= 10).sum())
                gates.append(1.0 if n_comp >= 2 else 0.0)

        gates = torch.tensor(
            gates,
            device=per_sample.device,
            dtype=per_sample.dtype,
            requires_grad=False,
        )
        per_sample = per_sample * gates  # Zero out if single component

    active = gates.sum()
    if reduction == "mean":
        return (per_sample * gates).sum() / active.clamp_min(1.0)
    elif reduction == "sum":
        return per_sample.sum()
    else:
        return per_sample  # no reduction

def eikonal_band_match_gt_vec(
    phi_pred,
    phi_gt,
    spacing,
    band_width,
    lambda_dir=0.25,
    reduction="mean",
    eps=1e-12,
):
    B = phi_pred.size(0)
    device, dtype = phi_pred.device, phi_pred.dtype
    sx, sy, sz = _as_batched_spacing(spacing, B, device, dtype)

    dxp, dyp, dzp = gradients_3d_batched(phi_pred, sx, sy, sz)
    dxg, dyg, dzg = gradients_3d_batched(phi_gt, sx, sy, sz)

    band = torch.logical_or(
        phi_pred.abs() <= band_width,
        phi_gt.abs() <= band_width,
    ).float()

    denom = band.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    w = band / denom

    # Inner products and norms (weighted)
    # <gp, gg> and ||gg||^2 for optimal scalar fit
    ip = (w * (dxp * dxg + dyp * dyg + dzp * dzg)).sum(
        dim=(2, 3, 4), keepdim=True
    )  # (B,1,1,1,1)
    gg2 = (
        (w * (dxg * dxg + dyg * dyg + dzg * dzg))
        .sum(dim=(2, 3, 4), keepdim=True)
        .clamp_min(eps)
    )
    s_star = (ip / gg2).detach()

    # L1 residual on vectors
    rx = (dxp - s_star * dxg).abs()
    ry = (dyp - s_star * dyg).abs()
    rz = (dzp - s_star * dzg).abs()
    vec_resid = (w * (rx + ry + rz)).sum(dim=(2, 3, 4)).squeeze(1)  # (B,)

    # Optional direction alignment (cosine) encourages normal consistency
    gp_norm = torch.sqrt(dxp * dxp + dyp * dyp + dzp * dzp + eps)
    gg_norm = torch.sqrt(dxg * dxg + dyg * dyg + dzg * dzg + eps)
    cos = (dxp * dxg + dyp * dyg + dzp * dzg) / (gp_norm * gg_norm + eps)
    dir_loss = (w * (1.0 - cos)).sum(dim=(2, 3, 4)).squeeze(1)  # (B,)

    per_sample = vec_resid + lambda_dir * dir_loss

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    else:
        return per_sample
