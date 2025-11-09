import torch
from monai.utils.enums import CommonKeys as Keys
import monai
from scipy.ndimage import distance_transform_edt
import json
import numpy as np
from typing import Optional, Sequence
from monai.transforms.utils import distance_transform_edt as monai_edt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def add_spacing(sample: dict) -> dict:
    # grab the affine out of the meta-dict
    aff = sample[Keys.IMAGE].meta["affine"]

    # squeeze the affine to 2D
    aff = torch.as_tensor(aff, dtype=torch.float32).squeeze()
    R = aff[:3, :3]

    spacing = torch.norm(R, p=2, dim=1)

    sample["spacing_tensor"] = spacing
    return sample


def binary_mask_labels(x: torch.Tensor, labels: list) -> torch.Tensor:
    """Create a binary mask for the specified labels in the label tensor."""
    # get max label + 1
    tmp_lbl = x.max() + 1
    for label in labels:
        x[x == label] = tmp_lbl
    x[x != tmp_lbl] = 0
    x[x == tmp_lbl] = 1
    return x


def remove_labels(x: torch.Tensor, labels: list, relabel: bool=False) -> torch.Tensor:
    """Remove the specified labels from the label tensor."""
    for label in labels:
        x[x == label] = 0
    
    if relabel:
        # get unique values in tensor x
        unique_values = x.unique()
        # Sort the unique values
        sorted_uv = sorted(unique_values)
        # Remap the labels
        for new_label, old_label in enumerate(sorted_uv):
            x[x == old_label] = new_label

    return x

def transform_labels(x: torch.Tensor, label_map: dict) -> torch.Tensor:
    """Transform labels in the tensor according to the provided label_map."""
    label_map_items = label_map.items()
    # sort it with old_labels
    sorted_items = sorted(label_map_items, key=lambda x: x[0])
    for old_label, new_label in sorted_items:
        mask = x == old_label
        if mask.any():
            x[mask] = new_label
    return x


def list_from_jsonl(jsonl_path, image_key="image", label_key="label", include_body_filled=False, body_filled_key="body_filled_mask"):
    """Pure function: read a .jsonl and return a list of dicts for MONAI."""
    files = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line)
            entry = {Keys.IMAGE: d[image_key], Keys.LABEL: d[label_key]}
            if include_body_filled:
                entry[body_filled_key] = d[body_filled_key]
            files.append(entry)
    return files


def split_colon_into_3_sections(binary_mask, method="spatial", device="cpu"):
    """
    Convert binary colon segmentation mask into 3-channel tensor with separate sections.

    Args:
        binary_mask: torch.Tensor or np.ndarray of shape [D, H, W] or [H, W, D]
                     Binary mask with 0=background, 1=colon
        method: str, either 'spatial' (faster) or 'skeleton' (more accurate)
        device: str, torch device for output tensor

    Returns:
        torch.Tensor of shape [3, D, H, W] where:
            channel 0: proximal section
            channel 1: middle section
            channel 2: distal section
    """

    # Convert to numpy if torch tensor
    if torch.is_tensor(binary_mask):
        mask_np = binary_mask.cpu().numpy()
    else:
        mask_np = binary_mask

    # Ensure binary
    mask_np = (mask_np > 0).astype(np.uint8)

    if method == "spatial":
        three_channel_mask = _split_spatial(mask_np)
    elif method == "skeleton":
        three_channel_mask = _split_skeleton(mask_np)
    else:
        raise ValueError(f"Method must be 'spatial' or 'skeleton', got {method}")

    # Convert to torch tensor [3, D, H, W]
    output_tensor = torch.from_numpy(three_channel_mask).float().to(device)

    return output_tensor


def _split_spatial(mask_np):
    """
    Split mask into 3 sections based on dominant spatial axis (faster method).
    """
    # Find colon voxels
    colon_coords = np.argwhere(mask_np > 0)

    if len(colon_coords) == 0:
        # Empty mask, return empty 3-channel
        return np.zeros((3, *mask_np.shape), dtype=np.float32)

    # Find principal axis using PCA
    centered = colon_coords - colon_coords.mean(axis=0)
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Project coordinates onto principal axis
    projections = colon_coords @ principal_axis

    # Divide into thirds based on projection values
    percentile_33 = np.percentile(projections, 33.33)
    percentile_66 = np.percentile(projections, 66.67)

    # Create 3-channel output
    three_channel = np.zeros((3, *mask_np.shape), dtype=np.float32)

    for i, coord in enumerate(colon_coords):
        proj_val = projections[i]
        if proj_val < percentile_33:
            channel = 0  # Proximal
        elif proj_val < percentile_66:
            channel = 1  # Middle
        else:
            channel = 2  # Distal

        three_channel[channel, coord[0], coord[1], coord[2]] = 1.0

    return three_channel


def _split_skeleton(mask_np):
    """
    Split mask into 3 sections based on centerline skeleton (more accurate).
    """
    # Find largest connected component
    from scipy.ndimage import label

    labeled, num_features = label(mask_np)

    if num_features == 0:
        return np.zeros((3, *mask_np.shape), dtype=np.float32)

    # Get largest component
    if num_features > 1:
        sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
        largest_label = np.argmax(sizes) + 1
        mask_np = (labeled == largest_label).astype(np.uint8)

    # Extract skeleton
    skeleton = skeletonize(mask_np)
    skeleton_coords = np.argwhere(skeleton)

    if len(skeleton_coords) < 3:
        # Fallback to spatial method if skeleton too small
        return _split_spatial(mask_np)

    # Order skeleton points along path
    ordered_skeleton = _order_skeleton_points(skeleton_coords)

    # Divide into thirds
    n = len(ordered_skeleton)
    section_1 = ordered_skeleton[: n // 3]
    section_2 = ordered_skeleton[n // 3 : 2 * n // 3]
    section_3 = ordered_skeleton[2 * n // 3 :]

    # Build KD-trees for each section
    tree_1 = cKDTree(section_1)
    tree_2 = cKDTree(section_2)
    tree_3 = cKDTree(section_3)

    # Get all colon voxels
    colon_coords = np.argwhere(mask_np > 0)

    # Create 3-channel output
    three_channel = np.zeros((3, *mask_np.shape), dtype=np.float32)

    # Assign each voxel to nearest section
    for coord in colon_coords:
        dist_1 = tree_1.query(coord)[0]
        dist_2 = tree_2.query(coord)[0]
        dist_3 = tree_3.query(coord)[0]

        distances = [dist_1, dist_2, dist_3]
        nearest_section = np.argmin(distances)

        three_channel[nearest_section, coord[0], coord[1], coord[2]] = 1.0

    return three_channel


def _order_skeleton_points(skeleton_coords):
    """
    Order skeleton points along the path from one end to the other.
    """
    n = len(skeleton_coords)

    if n < 2:
        return skeleton_coords

    # Build distance matrix (use sparse for efficiency)
    tree = cKDTree(skeleton_coords)

    # For each point, find k nearest neighbors (k=5 for colon topology)
    k = min(5, n)
    distances, indices = tree.query(skeleton_coords, k=k)

    # Build adjacency matrix
    rows = np.repeat(np.arange(n), k)
    cols = indices.flatten()
    data = distances.flatten()

    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Find endpoints (points with only 1-2 neighbors within small distance)
    threshold = np.percentile(
        distances[:, 1], 25
    )  # 25th percentile of nearest neighbor distances
    degree = (distances < threshold * 1.5).sum(axis=1)

    endpoints = np.where(degree <= 2)[0]

    if len(endpoints) < 2:
        # No clear endpoints, just use furthest points
        endpoints = [0, n - 1]

    # Compute shortest path from first to last endpoint
    start_idx = endpoints[0]

    # Find furthest endpoint from start
    distances_from_start = shortest_path(adj_matrix, indices=start_idx)
    valid_endpoints = [ep for ep in endpoints if np.isfinite(distances_from_start[ep])]

    if len(valid_endpoints) < 2:
        # Fallback: use simple ordering
        return skeleton_coords

    end_idx = valid_endpoints[
        np.argmax([distances_from_start[ep] for ep in valid_endpoints])
    ]

    # Get shortest path
    _, predecessors = shortest_path(
        adj_matrix, indices=start_idx, return_predecessors=True
    )

    # Reconstruct path
    path = []
    current = end_idx
    while current != -9999 and current != start_idx:
        path.append(current)
        current = predecessors[current]
        if current == -9999:
            break

    path.append(start_idx)
    path = path[::-1]

    if len(path) < n // 2:
        # Path reconstruction failed, fallback to spatial ordering
        projections = skeleton_coords @ skeleton_coords.mean(axis=0)
        sorted_indices = np.argsort(projections)
        return skeleton_coords[sorted_indices]

    return skeleton_coords[path]


def dataset_depended_transform_labels(x, kidneys_same_index=False, split_colon=False, split_colon_method="skeleton") -> torch.Tensor:
    """
    Apply the transform_labels function to the dependent dataset.
    Resulting label map:
      "1": "colon",
      "2": "rectum",
      "3": "small_bowel",
      "4": "stomach",
      "5": "liver",
      "6": "spleen",
      "7": "kidney_left",
      "8": "kidney_right",
      "9": "pancreas",
      "10": "urinary_bladder",
      "11": "duodenum",
      "12": "gallbladder",
    """
    pathname = str(x.meta["filename_or_obj"])

    if "colon_refined_by_mobina" in pathname:
        label_map = {
            0: 30,
            1: 30,
            2: 35,
            3: 30,
            4: 30,
            5: 37,
            6: 38,
            7: 36,
            8: 39,
            9: 42,
            10: 40,
            11: 34,
            12: 30,
            13: 33,
            14: 41,
            15: 31,
            16: 32,
            17: 30,
        }
        x = transform_labels(x, label_map)
        x.sub_(30)

    elif "female_cases_refined_by_md" in pathname:

        label_map = {
            0: 30,
            1: 30,
            2: 42,
            3: 37,
            4: 38,
            5: 35,
            6: 39,
            7: 30,
            8: 36,
            9: 34,
            10: 30,
            11: 30,
            12: 40,
            13: 31,
            14: 32,
            15: 41,
            16: 33,
            17: 30,
            18: 30,
            19: 30,
            20: 30,
            21: 30,  # uterus
            22: 30,  # portal vein and splenic vein
            23: 30,  # portal vein and splenic vein
            24: 30,  # portal vein and splenic vein
        }

        x = transform_labels(x, label_map)
        x.sub_(30)

    elif "male_cases_refined_by_md" in pathname:
        label_map = {
            0: 30,  # background
            1: 30,
            2: 42,
            3: 37,
            4: 38,
            5: 35,
            6: 39,
            7: 30,
            8: 36,
            9: 34,
            10: 30,
            11: 30,
            12: 40,
            13: 31,
            14: 32,
            15: 41,
            16: 33,
            17: 30,
            18: 30,
            19: 30,
            20: 30,
            21: 30,
            22: 30,
            23: 30,
        }
        x = transform_labels(x, label_map)
        x.sub_(30)

    elif ("a_grade_colons_not_in_refined_by_md" in pathname) or (
        "c_grade_colons/masks/" in pathname
    ):
        label_map = {
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
        }
        x = transform_labels(x, label_map)
    else:
        raise ValueError(f"Unknown dataset for {pathname}")

    if kidneys_same_index:
        # Map kidney_right (8) to kidney_left (7)
        kidney_merge_map = {8: 7}
        x = transform_labels(x, kidney_merge_map)

    if split_colon:
        # Split colon (1) into 3 sections
        colon_mask = (x == 1).float()

        # Store original shape and squeeze both colon_mask and x
        original_shape = x.shape
        original_ndim = colon_mask.ndim

        if colon_mask.ndim != 3:
            colon_mask = colon_mask.squeeze()
            x_squeezed = x.squeeze()  # Squeeze x as well
            if colon_mask.ndim != 3:
                raise ValueError("Colon mask must be 3D after squeezing.")
        else:
            x_squeezed = x

        three_channel_colon = split_colon_into_3_sections(
            colon_mask, method=split_colon_method, device=x.device
        )

        # Remove original colon label
        x_squeezed[x_squeezed == 1] = 0

        # Label 3 parts as 101, 102, 103
        for i in range(3):
            new_label = 101 + i
            x_squeezed[three_channel_colon[i] > 0] = new_label

        # Restore original shape if it was squeezed
        if original_ndim != 3:
            x = x_squeezed.reshape(original_shape)
        else:
            x = x_squeezed

    return x


def mask_to_sdf(
    mask: torch.Tensor,
    spacing: Optional[torch.Tensor | Sequence[float]] = None,
    *,
    inside_positive: bool = True,
    normalize: bool = True,
    float64_distances: bool = False,
) -> torch.Tensor:
    """
    Convert a (B,C,Z,Y,X) binary (or one-hot) mask to a per-channel SDF using MONAI's EDT.
    - spacing: (3,) or (B,3). If None, uses unit spacing.
    - inside_positive=True => inside > 0, outside < 0
    - normalize=True scales each channel by its own max |sdf| (safe for diffusion targets).
    """
    assert mask.ndim in (3, 4, 5), "mask must be (Z,Y,X), (C,Z,Y,X), or (B,C,Z,Y,X)"
    device = mask.device
    dtype = torch.float64 if float64_distances else torch.float32
    
    original_dim = mask.ndim

    # normalize shape to (B,C,Z,Y,X)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask.unsqueeze(0)
    B, C, *spatial = mask.shape

    # spacing -> per-batch 3-tuple
    if spacing is None:
        spacings = [(1.0, 1.0, 1.0)] * B
    else:
        if isinstance(spacing, torch.Tensor):
            s = spacing.detach().to("cpu")
            if s.ndim == 1:
                s = s.unsqueeze(0).repeat(B, 1)
            spacings = [tuple(map(float, s[b, -3:].tolist())) for b in range(B)]
        else:
            spacings = [tuple(float(x) for x in spacing[-3:])] * B

    out = torch.zeros((B, C, *spatial), device=device, dtype=dtype)

    # Per-sample/channel so spacing can vary across batch
    for b in range(B):
        for c in range(C):
            inside = mask[b, c] > 0
            # handle degenerate channels
            if not torch.any(inside) or torch.all(inside):
                # empty or full => 0 SDF (no surface)
                continue

            # MONAI EDT expects (num_channels, H, W[,D]); pass 1-channel tensors
            di = monai_edt(
                inside.unsqueeze(0).float(),
                sampling=spacings[b],
                float64_distances=float64_distances,
            ).squeeze(0)
            do = monai_edt(
                (~inside).unsqueeze(0).float(),
                sampling=spacings[b],
                float64_distances=float64_distances,
            ).squeeze(0)

            sdf = do - di  # outside positive, inside negative
            if inside_positive:
                sdf = -sdf  # flip so inside positive

            if normalize:
                m = torch.amax(torch.abs(sdf))
                if m > 0:
                    sdf = sdf / m

            out[b, c] = sdf.to(dtype)
    
    if original_dim == 3:
        out = out.squeeze(0).squeeze(0)
    elif original_dim == 4:
        out = out.squeeze(0)

    return out


def sdf_to_mask(
    sdf: torch.Tensor,
    *,
    inside_positive: bool = True,
    zero_level: float = 0.0,
    multi_label: bool = False,
    add_background_channel: bool = False,
    constraint: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert SDF back to a binary (or multi-hot) mask.
    - If multi_label=False (one-hot): picks the class with largest signed margin if > zero_level.
      If no class is positive, returns all-zeros (or adds background channel if requested).
    - If multi_label=True: thresholds each channel independently (multi-hot).
    - constraint: (B,1,...) mask to AND with the output.
    Returns tensor with shape:
      - multi_label=False, add_background_channel=False: (B,C,...) one-hot (or all zeros)
      - multi_label=False, add_background_channel=True:  (B,C+1,...) with background at index 0
      - multi_label=True:                                (B,C,...) multi-hot
    """
    assert sdf.ndim in (3, 4, 5), "sdf must be (Z,Y,X), (C,Z,Y,X), or (B,C,Z,Y,X)"

    original_dim = sdf.ndim
    # normalize shape to (B,C,spatial)
    if sdf.ndim == 3:
        sdf = sdf.unsqueeze(0).unsqueeze(0)
    elif sdf.ndim == 4:
        sdf = sdf.unsqueeze(0)
    B, C, *spatial = sdf.shape

    sign = sdf if inside_positive else -sdf

    if multi_label:
        out = (sign > float(zero_level)).to(sdf.dtype)
    else:
        # one-hot by largest positive margin
        max_vals, max_idx = torch.max(sign, dim=1)  # (B, …)
        pos = max_vals > float(zero_level)
        out = torch.zeros_like(sdf).to(sdf.dtype)  # (B,C,…)
        # scatter 1 where positive margin
        out.scatter_(1, max_idx.unsqueeze(1), pos.unsqueeze(1).to(sdf.dtype))

        if add_background_channel:
            # background at channel 0 (1 if no positive class)
            bg = (~pos).to(sdf.dtype).unsqueeze(1)  # (B,1,…)
            out = torch.cat([bg, out], dim=1)  # (B,C+1,…)

    if constraint is not None:
        # constraint assumed (B,1,…) or broadcastable; keep dtype of out
        out = out * (constraint > 0).to(out.dtype)

    if original_dim == 3:
        out = out.squeeze(0).squeeze(0)
    elif original_dim == 4:
        out = out.squeeze(0)
        
    return out


class MaskToSDFd(monai.transforms.MapTransform):
    """
    MapTransform version of the mask_to_sdf function.
    """

    def __init__(
        self,
        keys: list,
        spacing_key: Optional[str] = None,
        inside_positive: bool = True,
        normalize: bool = True,
        float64_distances: bool = False,
        allow_missing_keys: bool = False,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spacing_key: key for the spacing tensor. If None, unit spacing is used.
            inside_positive: passed to mask_to_sdf
            normalize: passed to mask_to_sdf
            float64_distances: passed to mask_to_sdf
            allow_missing_keys: see MapTransform
        """
        super().__init__(keys, allow_missing_keys)
        self.spacing_key = spacing_key
        self.inside_positive = inside_positive
        self.normalize = normalize
        self.float64_distances = float64_distances
        self.device = device

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            spacing = d.get(self.spacing_key, None)
            original_device = d[key].device
            arr = d[key]
            sdf_array = mask_to_sdf(
                arr,
                spacing=spacing.squeeze(),
                inside_positive=self.inside_positive,
                normalize=self.normalize,
                float64_distances=self.float64_distances,
            )
            d[key].set_array(sdf_array)
        return d
