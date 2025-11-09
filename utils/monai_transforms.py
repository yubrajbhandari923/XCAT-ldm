from monai.transforms import (
    MapTransform,
    Transform,
    LoadImage,
    Pad,
    SpatialCrop,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    SaveImage,
)
from typing import Callable, List, Optional, Tuple, Dict, Any
import pandas as pd
from monai.data import ITKReader, ImageReader, ImageWriter, MetaTensor
from typing import Sequence, Union
from os import PathLike
import os
import glob
from pathlib import Path
from monai.data.meta_tensor import MetaTensor

from skimage.morphology import erosion, dilation, ball
from monai.handlers import TensorBoardImageHandler

import SimpleITK as sitk
import numpy as np
import torch
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from .visualize import Visualizer, render3d
from typing import Union, Optional, Tuple, List
from .loss import SoftSkeletonize 
from skimage.morphology import skeletonize
# from skimage.morphology import skeletonize_3d
from .data import dataset_depended_transform_labels

import random
from collections import deque

import SimpleITK as sitk
import numpy as np
import logging

import gc

logging.getLogger(__name__)


class LogImgShaped(MapTransform):
    def __init__(
        self,
        keys,
        msg="Transform",
        log_unique=False,
        log_volumes=False,
        *args,
        **kwargs,
    ):
        self.msg = msg
        self.log_unique = log_unique
        self.log_volumes = log_volumes
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if not self.log_unique:
                logging.info(f"{self.msg} {key} Shape: {data[key].shape}")
            else:
                logging.info(
                    f"{self.msg} {key} Shape: {data[key].shape} Unique: {np.unique(data[key])}"
                )
            if self.log_volumes:
                for idx in np.unique(data[key]):
                    if idx == 0:
                        continue
                    logging.info(
                        f"{self.msg} {key} Volume {idx}: {np.sum(data[key] == idx)}"
                    )

        return data


class DropStructured(MapTransform):
    def __init__(
        self,
        keys,
        label_index=-1,
        background_index=0,
        random_drop_num=-1,
        keep_index=None,
    ):
        """
        If label_index is -1, then all labels except keep_index will be dropped.
        If keep_index is None, then all labels except label_index will be dropped.
        If random_drop_num is -1, then all labels will be dropped.
        If random_drop_num is 0, then no labels will be dropped.
        If random_drop_num is greater than 0, then random_drop_num labels will be dropped, excluding keep index.
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

        if label_index is None and keep_index is None:
            raise ValueError("Either label_index or keep_index must be provided")

        if label_index == -1:
            label_index = [-1]

        self.label_index = label_index
        self.keep_index = keep_index

        self.background_index = background_index
        self.random_drop_num = random_drop_num

    def __call__(self, data):
        for key in self.keys:

            # CHeck if label_index for out of range
            if not len(self.label_index) > 0:
                self.label_index = [-1]

            if self.label_index[0] == -1:
                self.label_index = np.unique(data[key])
                self.label_index = self.label_index[
                    self.label_index != self.background_index
                ]
                if self.keep_index is not None:
                    # remove the keep index from the label index
                    if not isinstance(self.keep_index, (list, tuple)):
                        self.keep_index = [self.keep_index]

                    self.label_index = [
                        i for i in self.label_index if i not in self.keep_index
                    ]

            if self.random_drop_num == 0:
                drop_structures = []
            elif self.random_drop_num > 0:
                drop_structures = np.random.choice(
                    self.label_index, self.random_drop_num, replace=False
                )
            else:
                drop_structures = self.label_index

            # logging.info(f"DropStructured: {key} Dropping {data[key]}")
            data[key] = data[key].to(torch.int32)

            for label in drop_structures:
                if label == 0:
                    continue
                data[key][data[key] == label] = self.background_index

        return data


class ResidualStructuresLabeld(MapTransform):
    def __init__(self, input_key, label_key, background_index=0):
        """Take all the structures in the input_key and remove them from the label_key. To get residual structures as label to predict."""
        self.input_key = input_key
        self.label_key = label_key
        self.background_index = background_index

    def __call__(self, data):
        input_img = data[self.input_key]
        label_img = data[self.label_key]

        for idx in np.unique(input_img):
            if idx == 0:
                continue
            label_img[label_img == idx] = self.background_index

        data[self.label_key] = label_img
        return data


class CombineStructuresd(MapTransform):
    def __init__(self, keys, label_to_structure, structure_to_label):
        """
        Args:
            keys: keys to combine the structures
            label_to_structure: function mapping label to structure name
            structure_to_label: function mapping structure name to label
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

        self.label_to_structure = label_to_structure
        self.structure_to_label = structure_to_label

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img = img.to(torch.int32)

            for idx in np.unique(img):
                if idx == 0:
                    continue
                structure = self.label_to_structure(idx)
                label = self.structure_to_label(structure)
                img[img == idx] = label

            data[key] = img

        return data


class Uniond(MapTransform):
    def __init__(self, key, label_key):
        self.key = key
        self.label_key = label_key

    def __call__(self, data):
        # for key in self.keys:
        data[self.key][data[self.label_key] != 0] = data[self.label_key][
            data[self.label_key] != 0
        ]
        # data[self.label_key][data[key] != 0] = data[key][data[key] != 0]
        data[self.label_key] = data[self.key]  # Just to make sure it is updated
        return data


class IndividualMasksToCombinedMask(Transform):
    def __init__(
        self,
        value_mapping,
        random_drop=-1,
        structures=None,
        match_shape_if_needed=True,
        allow_missing_structures=False,
    ):
        if isinstance(value_mapping, dict):
            value_mapping = value_mapping.get

        self.value_mapping = value_mapping
        self.random_drop = random_drop
        self.structures = structures
        self.match_shape_if_needed = match_shape_if_needed
        self.allow_missing_structures = allow_missing_structures

    def __call__(self, folder: str):
        if type(folder) is str:
            folder = Path(folder)

        structures = os.listdir(folder) if self.structures is None else self.structures

        if self.random_drop > 0:
            structures = random.sample(structures, len(structures) - self.random_drop)

        meta = None

        # structures = structures[:20]
        # combined_mask = np.array()
        logging.info(f"==============================================================")
        for idx, structure in enumerate(structures):
            structure_path = folder / structure

            if not os.path.isfile(structure_path):
                if os.path.exists(structure_path.with_suffix(".nii.gz")):
                    structure_path = structure_path.with_suffix(".nii.gz")
                else:
                    if self.allow_missing_structures:
                        logging.info(
                            f"\n\n Structure {structure} not found in {folder}. Skipping. \n\n"
                        )
                        continue

                    raise ValueError(
                        f"Structure {structure} not found in {structure_path}"
                    )

            try:
                structure_img = LoadImage()(str(structure_path))
            except Exception as e:
                raise ValueError(f"Error loading {structure_path}: {e}")

            structure_data = structure_img.get_array()
            structure_data = np.squeeze(structure_data).astype(bool)

            if idx == 0:
                meta = structure_img.meta
                affine = structure_img.affine
                combined_mask = np.zeros_like(structure_data, dtype=np.int16)

            # Check if affine are the same for structure_img and meta
            # assert bool(
            #     (structure_img.affine == affine).all()
            # ), f"Affine mismatch {affine} vs {structure_img.affine}"

            # if not bool((structure_img.affine == affine).all()):
            #     logging.info(f"Affine mismatch {affine} vs {structure_img.affine}")

            # logging.info(
            #     f"Structure: {structure} Value: {self.value_mapping(structure)}"
            # )
            # logging.info(
            #     f"Shape: {structure_data.shape} Unique: {np.unique(structure_data)}"
            # )

            if combined_mask.shape != structure_data.shape:
                logging.info(
                    f"Shape mismatch: {combined_mask.shape} != {structure_data.shape}"
                )
                if self.match_shape_if_needed:
                    combined_img = MetaTensor(combined_mask, meta=meta)
                    tmp_data = {
                        "combined": combined_img,
                        "structure": structure_img,
                    }
                    tmp_data = MatchShapeByPadd(keys=["combined", "structure"])(
                        tmp_data
                    )
                    combined_mask = tmp_data["combined"].squeeze().get_array()
                    structure_data = (
                        tmp_data["structure"].squeeze().get_array().astype(bool)
                    )
                    logging.info(
                        f"Shape after match: {combined_mask.shape} == {structure_data.shape}"
                    )
                else:
                    raise ValueError(
                        f"Shape mismatch: {combined_mask.shape} != {structure_data.shape}"
                    )

            combined_mask[structure_data] = self.value_mapping(
                structure.replace(".nii.gz", "")
            )
            # logging.info(f"Structure: {structure} Value: {self.value_mapping(structure)}, combined_mask: {np.unique(combined_mask)}")

        # combined_mask = combined_mask.astype(np.int16)
        # logging.info(f"Combine Unique: {np.unique(combined_mask)}")

        combined_mask = MetaTensor(combined_mask, meta=meta)
        combined_mask.meta["filename_or_obj"] = folder.name
        return combined_mask


class IndividualMasksToCombinedMaskd(MapTransform):

    def __init__(
        self,
        keys,
        value_mapping,
        random_drop_num=-1,
        structures=None,
        match_shape_if_needed=True,
        allow_missing_structures=False,
    ):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.value_mapping = value_mapping
        self.random_drop_num = random_drop_num
        self.structures = structures
        self.match_shape_if_needed = match_shape_if_needed
        self.allow_missing_structures = allow_missing_structures

    def __call__(self, data):
        for key in self.keys:
            folder = data[key]
            combined_mask = IndividualMasksToCombinedMask(
                self.value_mapping,
                self.random_drop_num,
                self.structures,
                self.match_shape_if_needed,
                self.allow_missing_structures,
            )(folder)
            data[key] = combined_mask
        return data


class ConvertMapping(Transform):
    def __init__(self, original_map, new_map):
        self.original_map = original_map
        self.new_map = new_map

    def __call__(self, img):

        # make the dtype int16
        img = img.to(torch.int32)

        for label in np.unique(img):
            if label == 0:
                continue

            try:
                img[img == label] = self.new_map(self.original_map(label))
            except:
                raise ValueError(f"Error converting label {label}")
        return img


class ConvertMappingd(MapTransform):
    def __init__(self, keys, original_map, new_map):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.original_map = original_map
        self.new_map = new_map

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img = ConvertMapping(self.original_map, self.new_map)(img)
            data[key] = img
        return data


class CopyKeyd(MapTransform):
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        # data[self.target_key] = data[self.source_key]
        if isinstance(data[self.source_key], MetaTensor):
            data[self.target_key] = data[self.source_key].clone()
        elif isinstance(data[self.source_key], str):
            data[self.target_key] = data[self.source_key]
        else:
            data[self.target_key] = data[self.source_key].clone()
        return data


class Visualize(Transform):
    def __init__(self, output_path, filename=None, **kwargs):
        self.output_path = output_path
        self.kwargs = kwargs
        self.filename = filename

    def __call__(self, img):
        if self.filename is not None:
            file_name = self.filename
        else:
            file_name = (
                img.meta["filename_or_obj"].replace(".nii.gz", "").split("/")[-1]
            )

        render3d(img, self.output_path, file_name=file_name, **self.kwargs)

        return img


class Visualized(MapTransform):

    def __init__(self, keys, output_path, filename=None, **kwargs):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        if filename is not None:
            if not isinstance(filename, (list, tuple)):
                filename = [filename]

        self.keys = keys
        self.output_path = output_path
        self.kwargs = kwargs
        self.filename = filename

    def __call__(self, data):
        for idx, key in enumerate(self.keys):
            img = data[key]

            if self.filename is not None:
                file_name = self.filename[idx]
                if file_name == "":
                    file_name = (
                        img.meta["filename_or_obj"]
                        .replace(".nii.gz", "")
                        .split("/")[-1]
                    )
            else:
                file_name = (
                    img.meta["filename_or_obj"].replace(".nii.gz", "").split("/")[-1]
                )

            render3d(img, self.output_path, file_name=file_name, **self.kwargs)
        return data


class MatchShapeByPadd(MapTransform):
    def __init__(self, keys, add=None):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.add = add

    def __call__(self, data):
        maximum = np.array([0, 0, 0])
        for k in self.keys:
            data[k] = data[k].squeeze()
            img = data[k]

            if len(img.shape) != 3:
                raise ValueError(f"Image shape {img.shape} not supported")

            maximum = np.maximum(maximum, img.shape)

        if self.add is not None:
            maximum += self.add

        data = EnsureChannelFirstd(keys=self.keys)(data)
        for k in self.keys:
            logging.info(f"Key: {k}, Shape: {data[k].shape}")
        logging.info(f"Maximum shape: {maximum}, Keys: {self.keys}")
        data = ResizeWithPadOrCropd(keys=self.keys, spatial_size=maximum)(data)

        return data


class AddTensord(MapTransform):
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        source_img = data[self.source_key].squeeze().numpy()
        target_img = data[self.target_key].squeeze().numpy()
        data[self.target_key].set_array(target_img + source_img)
        return data


class DropKeysd(MapTransform):
    def __init__(self, keys):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            del data[key]

        torch.cuda.empty_cache()
        gc.collect()

        return data


class LogDeviced(MapTransform):
    def __init__(self, keys, msg=""):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.msg = msg

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            logging.info(
                f"Key: {key} Device: {img.device}, Shape: {img.shape}, {self.msg}"
            )
        return data


class LogDiced(MapTransform):
    def __init__(
        self,
        target_key,
        template_key,
        target_label=None,
        template_label=None,
        msg="Dice Score",
    ):
        self.target_key = target_key
        self.template_key = template_key
        self.target_label = target_label
        self.template_label = template_label
        self.msg = msg

    def dice(self, img1, img2):
        if len(img1.unique()) > 2 or len(img2.unique()) > 2:
            logging.warning(
                f"Unique values in img1: {img1.unique()}, img2: {img2.unique()}"
            )

            for i in img1.unique():
                if i not in img2.unique():
                    img1[img1 == i] = 0
            for i in img2.unique():
                if i not in img1.unique():
                    img2[img2 == i] = 0

            total_dice = 0
            for i in img1.unique():
                if i == 0:
                    continue
                dice_score = self.dice(img1 == i, img2 == i)
                logging.info(f"Dice score for {i}: {dice_score}")
                total_dice += dice_score

            return total_dice / len(img1.unique())

        return 2 * (img1 * img2).sum() / (img1.sum() + img2.sum())

    def __call__(self, data):
        target = data[self.target_key]
        template = data[self.template_key]

        if self.target_label is not None:
            target = target == self.target_label
        if self.template_label is not None:
            template = template == self.template_label

        logging.info(f"{self.msg}: {self.dice(target.squeeze(), template.squeeze())}")

        return data


class SaveLabelsSeperate(MapTransform):
    def __init__(self, keys, output_path_key, class_map, labels=None):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.labels = labels
        self.output_path_key = output_path_key
        self.class_map = class_map

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            save_path = data[self.output_path_key]

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if self.labels is None:
                self.labels = img.unique()

            for label in self.labels:
                if label == 0:
                    continue

                img_copy = img.clone()
                img_copy[img_copy != label] = 0
                img_copy[img_copy == label] = 1
                img_copy.meta["filename_or_obj"] = os.path.join(
                    save_path, f"{self.class_map[label.item()]}.nii.gz"
                )

                SaveImage(
                    output_dir=save_path, output_postfix="", separate_folder=False
                )(img_copy)

        return data

def _default_select_fn(x):
    return x > 0

class CropForegroundAxisd(MapTransform):
    """
    Crop the tensors in `keys` along a single spatial axis based on the foreground
    of `source_key`. Other axes are left untouched.
    """

    def __init__(self, keys, source_key, axis=0, select_fn=_default_select_fn, margin=5):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        super().__init__(keys)
        if axis not in (0, 1, 2):
            raise ValueError(f"`axis` must be 0, 1, or 2; got {axis}")
        if margin < 0:
            raise ValueError("`margin` must be >= 0")
        self.keys = list(keys)
        self.source_key = source_key
        self.axis = axis
        self.select_fn = select_fn
        self.margin = margin

    def _to_tensor(self, x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    def _get_spatial_axis_index(self, arr_ndim: int) -> int:
        if arr_ndim < 3:
            raise ValueError(
                f"Input must have at least 3 dims (D,H,W). Got ndim={arr_ndim}"
            )
        # spatial dims are the last 3 dims
        return arr_ndim - 3 + self.axis

    def _compute_crop_indices(self, src):
        t = self._to_tensor(src)

        # Reduce all non-spatial dims to a 3D spatial volume (D,H,W)
        if t.ndim == 3:
            spatial = t
        else:
            n_spatial = 3
            reduce_dims = tuple(range(t.ndim - n_spatial))  # e.g., (0,) for C,D,H,W
            spatial = t.any(dim=reduce_dims).to(t.dtype)

        mask = self.select_fn(spatial)
        mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        mask = mask.bool()

        if mask.ndim != 3:
            raise ValueError(f"Foreground mask must be 3D; got {tuple(mask.shape)}")

        axis = self.axis
        other = tuple(d for d in (0, 1, 2) if d != axis)
        # ↓↓↓ fix: reduce both non-axis dims at once to get a 1D presence vector
        presence_1d = mask.any(dim=other)

        if not presence_1d.any():
            return None

        idxs = presence_1d.nonzero(as_tuple=False).squeeze(-1)
        start = int(idxs.min().item())
        end_inclusive = int(idxs.max().item())
        size_axis = mask.shape[axis]

        start = max(0, start - self.margin)
        end = min(size_axis, end_inclusive + 1 + self.margin)  # [start, end)

        # Safety: never empty
        if end <= start:
            center = int((idxs.float().mean().round().item()))
            start = max(0, min(center, size_axis - 1))
            end = start + 1

        return start, end

    def __call__(self, data):
        d = dict(data)

        if self.source_key not in d:
            return d

        crop_range = self._compute_crop_indices(d[self.source_key])
        if crop_range is None:
            return d  # nothing to crop

        start, end = crop_range

        def _safe_crop(arr):
            arr_ndim = arr.ndim if hasattr(arr, "ndim") else np.asarray(arr).ndim
            gaxis = self._get_spatial_axis_index(arr_ndim)
            slicers = [slice(None)] * arr_ndim
            slicers[gaxis] = slice(start, end)
            out = arr[tuple(slicers)]
            # --------- NEW: safety net, avoid 0-size dim ----------
            if out.shape[gaxis] == 0:
                return arr  # fallback to no crop for this key
            # ------------------------------------------------------
            return out

        for key in self.keys:
            if key not in d:
                continue
            arr = d[key]
            d[key] = _safe_crop(arr)

            # meta_key = f"{key}_meta_dict"
            # if meta_key in d and isinstance(d[meta_key], dict):
            #     d[meta_key]["spatial_shape"] = np.asarray(
            #         d[key].shape[-3:], dtype=np.int64
            #     )

        # Also crop source_key itself if it's not already included
        if self.source_key not in self.keys and self.source_key in d:
            arr = d[self.source_key]
            d[self.source_key] = _safe_crop(arr)
            # meta_key = f"{self.source_key}_meta_dict"
            # if meta_key in d and isinstance(d[meta_key], dict):
            #     d[meta_key]["spatial_shape"] = np.asarray(
            #         d[self.source_key].shape[-3:], dtype=np.int64
            #     )

        return d


import numpy as np
from typing import Union
from monai.transforms import Transform
from monai.config import KeysCollection
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    distance_transform_edt,
    label,
)
import torch


class SmoothColonMask(Transform):
    """
    Enhanced colon mask smoothing with endpoint protection and anatomical awareness.

    Processing pipeline:
    1. (Optional) Identify and protect colon endpoints using skeleton + anatomical orientation
    2. Break thin necks in middle regions only
    3. Remove small components
    4. Apply dilation-erosion smoothing (excluding protected endpoints)
    5. Merge protected endpoints with smoothed middle

    Args:
        iterations: Number of dilation/erosion iterations for smoothing (default: 3)
        connectivity: Structuring element connectivity (1, 2, or 3) (default: 2)
        min_neck_thickness: Minimum thickness for connections in voxels (default: 3)
        min_component_ratio: Minimum component size ratio (default: 0.1)
        protect_endpoints: Whether to protect colon endpoints from smoothing (default: True)
        endpoint_protection_radius: Radius of protection zone around endpoints (default: 8)
        data_orientation: Anatomical orientation to identify correct endpoints (default: "RAS")
                         Format: three letters indicating positive direction for [x, y, z]
                         - R/L: Right/Left
                         - A/P: Anterior/Posterior
                         - S/I: Superior/Inferior
                         Common: "RAS" (Right-Anterior-Superior)

                         For colon: ascending starts lower-right, descending ends lower-left
    """

    def __init__(
        self,
        iterations: int = 3,
        connectivity: int = 2,
        min_neck_thickness: int = 3,
        min_component_ratio: float = 0.1,
        protect_endpoints: bool = True,
        endpoint_protection_radius: int = 8,
        data_orientation: str = "RAS",
    ):
        self.iterations = iterations
        self.connectivity = connectivity
        self.min_neck_thickness = min_neck_thickness
        self.min_component_ratio = min_component_ratio
        self.protect_endpoints = protect_endpoints
        self.endpoint_protection_radius = endpoint_protection_radius
        self.data_orientation = data_orientation.upper()

        # Create structuring element for dilation/erosion
        self.struct_element = generate_binary_structure(3, connectivity)

        # Validate orientation string
        if len(self.data_orientation) != 3:
            raise ValueError(
                f"data_orientation must be 3 characters, got: {self.data_orientation}"
            )

    def _parse_orientation(self) -> Tuple[int, int, int]:
        """
        Parse orientation string to determine anatomical directions.

        Returns:
            (lr_axis, ap_axis, si_axis): Axes for Left-Right, Anterior-Posterior, Superior-Inferior
                                         Values are 0, 1, or 2 (for x, y, z)
        """
        orientation_map = {
            "R": (0, 1),
            "L": (0, -1),  # Right/Left on axis 0
            "A": (1, 1),
            "P": (1, -1),  # Anterior/Posterior on axis 1
            "S": (2, 1),
            "I": (2, -1),  # Superior/Inferior on axis 2
        }

        axes = []
        directions = []

        for char in self.data_orientation:
            if char not in orientation_map:
                raise ValueError(f"Invalid orientation character: {char}")
            axis, direction = orientation_map[char]
            axes.append(axis)
            directions.append(direction)

        return tuple(axes), tuple(directions)

    def _get_skeleton_endpoints(self, mask: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Find endpoint voxels using skeletonization.

        Args:
            mask: Binary mask

        Returns:
            List of (x, y, z) coordinates of endpoint voxels
        """
        # Create skeleton (1-voxel thick centerline)
        skeleton = skeletonize(mask, method="lee")

        # Find skeleton voxels
        skeleton_coords = np.argwhere(skeleton)

        if len(skeleton_coords) == 0:
            return []

        # For each skeleton voxel, count 26-connected neighbors that are also skeleton
        endpoints = []

        for coord in skeleton_coords:
            x, y, z = coord

            # Check 26-neighborhood
            neighbor_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        nx, ny, nz = x + dx, y + dy, z + dz

                        # Check bounds
                        if (
                            0 <= nx < skeleton.shape[0]
                            and 0 <= ny < skeleton.shape[1]
                            and 0 <= nz < skeleton.shape[2]
                        ):
                            if skeleton[nx, ny, nz]:
                                neighbor_count += 1

            # Endpoint has only 1 neighbor on skeleton
            if neighbor_count == 1:
                endpoints.append((x, y, z))

        return endpoints

    def _identify_colon_endpoints(
        self, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Identify ascending (start) and descending (end) colon endpoints using anatomical orientation.

        Anatomical knowledge:
        - Ascending colon: starts at cecum (lower right abdomen)
        - Descending colon: ends at sigmoid/rectum (lower left abdomen)

        Returns:
            (start_coord, end_coord): Coordinates of start and end, or (None, None) if not found
        """
        # Get all candidate endpoints from skeleton
        candidate_endpoints = self._get_skeleton_endpoints(mask)

        if len(candidate_endpoints) < 2:
            # Not enough endpoints found, return None
            return None, None

        # Parse orientation to understand spatial layout
        axes, directions = self._parse_orientation()

        # Find which axis corresponds to Left-Right and Superior-Inferior
        lr_axis = None
        si_axis = None
        lr_direction = None
        si_direction = None

        for i, char in enumerate(self.data_orientation):
            if char in ["R", "L"]:
                lr_axis = axes[i]
                lr_direction = directions[i]
            elif char in ["S", "I"]:
                si_axis = axes[i]
                si_direction = directions[i]

        if lr_axis is None or si_axis is None:
            # Can't determine anatomical directions
            return None, None

        # Convert candidates to array for easier manipulation
        candidates_array = np.array(candidate_endpoints)

        # Ascending colon (cecum): inferior + right
        # Descending colon end: inferior + left

        # Find most inferior points (lowest in superior-inferior axis)
        if si_direction > 0:  # S orientation, inferior is low values
            inferior_threshold = np.percentile(candidates_array[:, si_axis], 30)
            inferior_candidates = candidates_array[
                candidates_array[:, si_axis] <= inferior_threshold
            ]
        else:  # I orientation, inferior is high values
            inferior_threshold = np.percentile(candidates_array[:, si_axis], 70)
            inferior_candidates = candidates_array[
                candidates_array[:, si_axis] >= inferior_threshold
            ]

        if len(inferior_candidates) < 2:
            # Fall back to all candidates
            inferior_candidates = candidates_array

        # Among inferior candidates, find rightmost and leftmost
        if lr_direction > 0:  # R orientation
            # Rightmost (highest value) = ascending start
            # Leftmost (lowest value) = descending end
            right_idx = np.argmax(inferior_candidates[:, lr_axis])
            left_idx = np.argmin(inferior_candidates[:, lr_axis])
        else:  # L orientation
            # Leftmost (highest value) = descending end
            # Rightmost (lowest value) = ascending start
            right_idx = np.argmin(inferior_candidates[:, lr_axis])
            left_idx = np.argmax(inferior_candidates[:, lr_axis])

        start_coord = inferior_candidates[right_idx]  # Ascending (right)
        end_coord = inferior_candidates[left_idx]  # Descending (left)

        return start_coord, end_coord

    def _create_endpoint_protection_mask(
        self,
        mask_shape: Tuple,
        start_coord: Optional[np.ndarray],
        end_coord: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Create a binary mask marking protected regions around endpoints.

        Args:
            mask_shape: Shape of the mask
            start_coord: Coordinates of start endpoint
            end_coord: Coordinates of end endpoint

        Returns:
            Binary mask with True in protected regions
        """
        protection_mask = np.zeros(mask_shape, dtype=bool)

        if start_coord is None and end_coord is None:
            return protection_mask

        # Create spherical protection zones around each endpoint
        coords = np.indices(mask_shape)

        for endpoint in [start_coord, end_coord]:
            if endpoint is not None:
                # Calculate distance from endpoint
                dist_sq = sum((coords[i] - endpoint[i]) ** 2 for i in range(3))

                # Mark voxels within protection radius
                protection_mask |= dist_sq <= self.endpoint_protection_radius**2

        return protection_mask

    def _break_thin_necks(
        self, mask: np.ndarray, protection_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Break thin connections (necks) between regions, excluding protected areas.

        Args:
            mask: Binary mask
            protection_mask: Optional mask of regions to protect

        Returns:
            Mask with thin necks removed
        """
        if self.min_neck_thickness <= 0:
            return mask

        # Compute distance transform
        dist_transform = distance_transform_edt(mask)

        # Identify thin regions
        thin_threshold = self.min_neck_thickness / 2.0
        thick_mask = dist_transform >= thin_threshold

        # Apply thin neck removal
        result = mask & thick_mask

        # Restore protected regions
        if protection_mask is not None:
            result = result | (mask & protection_mask)

        return result

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove connected components below size threshold.

        Args:
            mask: Binary mask

        Returns:
            Mask with small components removed
        """
        if self.min_component_ratio <= 0:
            return mask

        labeled_mask, num_components = label(mask)

        if num_components == 0:
            return mask

        # Calculate size of each component
        component_sizes = {}
        for i in range(1, num_components + 1):
            component_sizes[i] = np.sum(labeled_mask == i)

        # Find largest component
        max_size = max(component_sizes.values())
        threshold_size = max_size * self.min_component_ratio

        # Create output mask with only large components
        result = np.zeros_like(mask, dtype=bool)
        for component_label, size in component_sizes.items():
            if size >= threshold_size:
                result |= labeled_mask == component_label

        return result

    def _smooth_morphological(
        self, mask: np.ndarray, protection_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Smooth mask using dilation followed by erosion, excluding protected regions.

        Args:
            mask: Binary mask
            protection_mask: Optional mask of regions to protect

        Returns:
            Smoothed mask
        """
        # Extract regions to smooth
        if protection_mask is not None:
            protected_region = mask & protection_mask
            smoothing_region = mask & ~protection_mask
        else:
            protected_region = None
            smoothing_region = mask

        # Apply dilation-erosion to smoothing region only
        dilated = smoothing_region.copy()
        for _ in range(self.iterations):
            dilated = binary_dilation(dilated, structure=self.struct_element)

        smoothed = dilated.copy()
        for _ in range(self.iterations):
            smoothed = binary_erosion(smoothed, structure=self.struct_element)

        # Merge back with protected region
        if protected_region is not None:
            result = smoothed | protected_region
        else:
            result = smoothed

        return result

    def __call__(
        self, mask: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply full processing pipeline to mask with smart early exits and endpoint protection.

        Args:
            mask: Binary segmentation mask

        Returns:
            Processed mask in same format as input
        """
        # Handle torch tensors
        is_torch = isinstance(mask, torch.Tensor)
        if is_torch:
            device = mask.device
            dtype = mask.dtype
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask.copy()

        # Ensure binary
        mask_np = mask_np.astype(bool)

        # Handle channel dimension
        squeeze_dim = False
        if mask_np.ndim == 4 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
            squeeze_dim = True

        # EARLY EXIT CHECK: If already single component, skip all processing
        _, num_components_initial = label(mask_np)

        if num_components_initial <= 1 and not self.protect_endpoints:
            # Already clean and no endpoint protection needed
            result = mask_np
        else:
            # Step 0: Identify and protect endpoints if enabled
            protection_mask = None
            if self.protect_endpoints:
                start_coord, end_coord = self._identify_colon_endpoints(mask_np)
                if start_coord is not None or end_coord is not None:
                    protection_mask = self._create_endpoint_protection_mask(
                        mask_np.shape, start_coord, end_coord
                    )

            # Proceed with processing if multiple components
            if num_components_initial > 1:
                # Step 1: Break thin necks (avoiding protected regions)
                if self.min_neck_thickness > 0:
                    result = self._break_thin_necks(mask_np, protection_mask)
                else:
                    result = mask_np

                # Step 2: Remove small components
                if self.min_component_ratio > 0:
                    result = self._remove_small_components(result)

                # Check components after preprocessing
                _, num_components_after_prep = label(result)

                # Step 3: Apply smoothing if multiple components remain
                if num_components_after_prep > 1:
                    result = self._smooth_morphological(result, protection_mask)
            else:
                # Single component, but we may still want to smooth (avoiding endpoints)
                if self.iterations > 0:
                    result = self._smooth_morphological(mask_np, protection_mask)
                else:
                    result = mask_np

        # Restore dimensions
        if squeeze_dim:
            result = result[np.newaxis, ...]

        # Convert back to torch if needed
        if is_torch:
            result = torch.from_numpy(result.astype(np.float32)).to(
                device=device, dtype=dtype
            )

        return result


class SmoothColonMaskd(Transform):
    """
    Dictionary-based version for MONAI pipelines with endpoint protection.

    Args:
        keys: Keys to apply transform to
        iterations: Number of dilation/erosion iterations (default: 3)
        connectivity: Structuring element connectivity (1, 2, or 3)
        min_neck_thickness: Minimum thickness for connections in voxels (default: 3)
        min_component_ratio: Minimum component size ratio (default: 0.1)
        protect_endpoints: Whether to protect colon endpoints (default: True)
        endpoint_protection_radius: Radius of protection zone (default: 8)
        data_orientation: Anatomical orientation string (default: "RAS")
        allow_missing_keys: Don't raise error for missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        iterations: int = 3,
        connectivity: int = 2,
        min_neck_thickness: int = 3,
        min_component_ratio: float = 0.1,
        protect_endpoints: bool = True,
        endpoint_protection_radius: int = 8,
        data_orientation: str = "RAS",
        allow_missing_keys: bool = False,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.transform = SmoothColonMask(
            iterations=iterations,
            connectivity=connectivity,
            min_neck_thickness=min_neck_thickness,
            min_component_ratio=min_component_ratio,
            protect_endpoints=protect_endpoints,
            endpoint_protection_radius=endpoint_protection_radius,
            data_orientation=data_orientation,
        )
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: dict) -> dict:
        """Apply transform to dictionary"""
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key].set_array(self.transform(d[key]))
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in data")
        return d


class HarmonizeLabelsd(MapTransform):
    """
    Wraps your dataset_depended_transform_labels(...) into a MapTransform.
    Applies in-place harmonization on multi-organ masks.
    """

    def __init__(self, keys: Sequence[str], kidneys_same_index: bool = True, split_colon: bool = False, split_colon_method: str = "skeleton"):
        super().__init__(keys)
        self.kidneys_same_index = kidneys_same_index
        self.split_colon = split_colon
        self.split_colon_method = split_colon_method

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            # Expecting Tensor/MetaTensor-like with .set_array; preserve dtype
            out = dataset_depended_transform_labels(
                arr, kidneys_same_index=self.kidneys_same_index, split_colon=self.split_colon, split_colon_method=self.split_colon_method
            )
            d[key] = out
        return d


class EnsureAllTorchd(MapTransform):
    """
    Minimal: only handle top-level values that are MetaTensor.
    For each such value, convert any numpy items in .meta to torch tensors.
    Prints (data_key, meta_key) pairs that were converted.
    """

    def __init__(self, print_changes: bool = True):
        super().__init__(keys=None)
        self.print_changes = print_changes

    def __call__(self, data):
        out = dict(data)
        changes = []

        for data_key, val in list(out.items()):
            if isinstance(val, MetaTensor):
                # Iterate through the meta dict and convert numpy -> torch
                for meta_key, meta_val in list(val.meta.items()):
                    if isinstance(meta_val, (np.ndarray, np.generic)):
                        val.meta[meta_key] = torch.as_tensor(meta_val)
                        changes.append((data_key, meta_key))

        if self.print_changes and changes:
            print(
                f"[EnsureAllTorchd] Converted {len(changes)} numpy item(s) in MetaTensor.meta -> torch.Tensor:"
            )
            for dk, mk in changes:
                print(f"  - data_key='{dk}', meta_key='{mk}'")

        return out


class AddSpacingTensord(MapTransform):
    """
    Extracts spacing from meta (after Spacingd/Orientationd) and stores a torch tensor
    under key 'spacing_tensor'. Works once per sample (no 'keys' needed at call site).
    """

    def __init__(self, ref_key: str):
        super().__init__(keys=[ref_key])
        self.ref_key = ref_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        aff = d[self.ref_key].meta["affine"]

        # squeeze the affine to 2D
        aff = torch.as_tensor(aff, dtype=torch.float32).squeeze()
        R = aff[:3, :3]

        spacing = torch.norm(R, p=2, dim=1)

        d["spacing_tensor"] = spacing
        return d


class FilterAndRelabeld(MapTransform):
    """
    - IMAGE: union of conditioning organs -> binary {0,1}
    - LABEL: target organ only           -> binary {0,1}
    Uses memory-lean ops (torch.isin) and preserves dtype/container.
    """

    def __init__(
        self,
        image_key: str,
        label_key: str,
        conditioning_organs: Sequence[int],
        target_organ: int,
    ):
        super().__init__(keys=[image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.target_organ = int(target_organ)
        self.conditioning = torch.as_tensor(
            list(conditioning_organs), dtype=torch.int64
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        # IMAGE -> union mask of conditioning organs
        img = d[self.image_key]
        if self.conditioning.numel() > 0:
            cond = torch.isin(img, self.conditioning)
            d[self.image_key] = cond.to(img.dtype)  # preserve dtype
        else:
            d[self.image_key].set_array(torch.zeros_like(img))
  

        # LABEL -> target organ only
        lbl = d[self.label_key]
        tgt = lbl == self.target_organ
        d[self.label_key] = tgt.to(lbl.dtype)  # preserve dtype

        return d


class DivideFilterAndRelabeld(MapTransform):
    """
    - IMAGE: union of conditioning organs -> binary {0,1}
    - LABEL: target organ only           -> binary {0,1}
    Uses memory-lean ops (torch.isin) and preserves dtype/container.
    """

    def __init__(
        self,
        image_key: str,
        label_key: str,
        # conditioning_organs: Sequence[int],
        generation_sequence: list,
        target_organs: list,
        label_to_organ_name: dict,
    ):
        super().__init__(keys=[image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.target_organs = target_organs
        self.generation_sequence = generation_sequence
        self.label_to_organ_name = label_to_organ_name
        
    def get_conditioning_organs(
        self, target_organ_index: int, generation_order: list = None
    ) -> List[int]:
        if generation_order is None:
            generation_order = self.generation_sequence
            
        if target_organ_index not in generation_order:
            raise ValueError(
                f"Target organ {target_organ_index} not in generation order: {generation_order}"
            )
            
        pos = generation_order.index(target_organ_index)
        return generation_order[:pos]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for target_organ in self.target_organs:
            tmp_dict = dict()
            conditioning_organs = self.get_conditioning_organs(target_organ, self.generation_sequence)
            
            # Create a copy of image and label for each target organ
            tmp_dict["Image"] = data[self.image_key].clone()
            tmp_dict["Label"] = data[self.label_key].clone()
            
            transform = FilterAndRelabeld(
                image_key="Image",
                label_key="Label",
                conditioning_organs=conditioning_organs,
                target_organ=target_organ,
            )

            tmp_dict = transform(tmp_dict)
            organ_name = self.label_to_organ_name[target_organ]
            data[f"Image_{organ_name}"] = tmp_dict["Image"]
            data[f"Label_{organ_name}"] = tmp_dict["Label"]

        return data


class Probe(MapTransform):
    def __init__(self, keys=[], allow_missing_keys = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"Probe transform called on data keys: {list(data.keys())}")
        self.keys = self.keys if self.keys else list(data.keys())
        # Log Data Types for each key
        for key in self.keys:
            if key in data:
                logging.info(f"Key: {key}, Type: {type(data[key])}, UniqueValues Len: {data[key].unique().numel()}, Non-zero Volume: {data[key].sum()} Shape: {getattr(data[key], 'shape', 'N/A')}")
            else:
                logging.warning(f"Key: {key} not found in data.")
        return data


class CombineKeysd(MapTransform):
    def __init__(self, keys, result_key, as_binary=True):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.result_key = result_key
        self.as_binary = as_binary
        
    def __call__(self, data):
        # Check if all keys are present and have same shape
        shape = data[self.keys[0]].shape
        for k in self.keys:
            if data[k].shape != shape:
                raise ValueError(f"All keys must have the same shape. Key {k} has shape {data[k].shape}, expected {shape}.")
        
        combined = torch.zeros_like(data[self.keys[0]])
        for idx, k in enumerate(self.keys):
            if self.as_binary:
                combined += (data[k] > 0).to(combined.dtype)
            else:
                combined[data[k] > 0] = idx + 1  # Labels start from 1
        
        data[self.result_key].set_array(combined)
        return data