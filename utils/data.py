import torch
from monai.utils.enums import CommonKeys as Keys
import monai
from scipy.ndimage import distance_transform_edt
import json
import numpy as np
from typing import Optional, Sequence
from monai.transforms.utils import distance_transform_edt as monai_edt


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
    #sort it with old_labels
    sorted_items = sorted(label_map_items, key=lambda x: x[0])
    for old_label, new_label in sorted_items:
        x[x == old_label] = new_label
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


def dataset_depended_transform_labels(x):
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
        x = x - 30

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
        x = x - 30

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
        x = x - 30
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
        device: Optional[torch.device] = "cuda",
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
            sdf_array = mask_to_sdf(
                d[key].as_tensor().to(self.device),
                spacing=spacing.squeeze(),
                inside_positive=self.inside_positive,
                normalize=self.normalize,
                float64_distances=self.float64_distances,
            )
            d[key] = d[key].set_array(sdf_array.to(original_device))
            d[key] = d[key].to("cpu")
        return d
    
