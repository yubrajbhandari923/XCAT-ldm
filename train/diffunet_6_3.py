# region imports
# Implements FLim DiffUNet

# Iterative Generation of Abdominal Organs
import functools
import json
import logging
import os
import sys
import time
import itertools
import math
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import aim

import torch
import torch.multiprocessing as tmp_mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel
from ignite.utils import setup_logger
from torch.nn.parallel import DistributedDataParallel, DataParallel

import numpy as np
import cc3d

from ignite.engine import Events
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim
from ignite.metrics import MeanAbsoluteError

from aim.pytorch_ignite import AimLogger


# from ignite.utils import setup_logger

import monai
from monai import transforms
from monai.data import list_data_collate
from monai.data.utils import collate_meta_tensor
from monai.handlers import MeanDice, StatsHandler, from_engine
from monai.inferers import LatentDiffusionInferer, DiffusionInferer
from monai.engines.utils import IterationEvents
from monai.networks.nets import DiffusionModelUNet, BasicUNet
from monai.networks.nets import PatchDiscriminator
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.engines import SupervisedTrainer, SupervisedEvaluator, Evaluator, Trainer
from monai.utils import set_determinism, AdversarialIterationEvents, AdversarialKeys
from monai.utils.enums import CommonKeys as Keys
from monai.losses import PatchAdversarialLoss, HausdorffDTLoss
from monai.engines.utils import DiffusionPrepareBatch
from monai.data import decollate_batch

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

from model.diffUNet.BTCV import DiffUNet
from model.DiffUNetFLiM import DiffusionModelUNetFiLM, FiLMAdapter

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils import log_config, _prepare_batch_factory
from utils.data import (
    add_spacing,
    binary_mask_labels,
    remove_labels,
    transform_labels,
    list_from_jsonl,
    dataset_depended_transform_labels,
    mask_to_sdf,
    sdf_to_mask,
    MaskToSDFd,

)
from utils.monai_transforms import CropForegroundAxisd, SmoothColonMaskd, HarmonizeLabelsd, AddSpacingTensord, FilterAndRelabeld, Probe, EnsureAllTorchd, CombineKeysd, DivideFilterAndRelabeld
from utils.resume import resume_from_checkpoint
from utils.handlers import attach_handlers, attach_inference_saver
from utils.loss import SoftCIDice, topo_loss, largest_component_dice_loss, loss_isoperimetric, eikonal_band_match_gt_vec, surface_area_from_sdf_normalized, volume_from_mask_batched

tmp_mp.set_sharing_strategy("file_system")
torch.serialization.add_safe_globals([monai.utils.enums.CommonKeys])
# stash the original loader
_torch_load = torch.load

def __torch_load(f, **kwargs):
    if 'weights_only' in kwargs:
        kwargs.pop('weights_only')
    
    return _torch_load(f, weights_only=False, **kwargs)
# override so all loads are unguarded, remove weights_only from kwargs if present
torch.load = __torch_load

torch.autograd.set_detect_anomaly(True)

# endregion


def get_aim_logger(config):
    logging.info(
        f"[Rank 0] Initializing Aim Logger with repo: {config.logging.aim_repo}"
    )

    if not config.experiment.name or len(config.experiment.name) == 0:
        raise ValueError("Experiment name is required")

    aim_logger = AimLogger(
        repo=config.logging.aim_repo,
        experiment=f"{config.experiment.name}_{config.experiment.version}",
    )

    config.experiment.hash = aim_logger.experiment.hash[:8]

    logging.info(f"[Rank 0] Aim Logger initialized")

    if config.training.inference_mode:
        aim_logger.experiment.add_tag("Inference")
    else:
        aim_logger.experiment.add_tag("Train")

    for tag in config.experiment.tags:
        aim_logger.experiment.add_tag(tag)

    # aim_logger.experiment.add_tag(config.name)

    aim_logger.experiment.description = config.experiment.description
    aim_logger.log_params(OmegaConf.to_container(config, resolve=True))
    aim_logger.experiment.log_info(
        OmegaConf.to_yaml(config),
    )

    # Log this script's content
    script_path = os.path.abspath(__file__)
    with open(script_path, "r") as script_file:
        script_content = script_file.read()
    aim_logger.experiment.log_info(script_content)

    # Save the updated config to a file for easy recreation
    if config.training.save_config_yaml and (not config.training.inference_mode):
        # create dir if not exists
        if not os.path.exists(config.training.save_dir):
            os.makedirs(config.training.save_dir, exist_ok=True)

        with open(
            os.path.join(config.training.save_dir, "config.yaml"),
            "w",
        ) as f:
            OmegaConf.save(config=config, f=f)

        with open(
            os.path.join(config.training.save_dir, "train_script.py"),
            "w",
        ) as f:
            f.write(script_content)
    return aim_logger


# region Data Loading and Preprocessing


ORGAN_NAMES = {
    1: "colon",
    2: "rectum",
    3: "small_bowel",
    4: "stomach",
    5: "liver",
    6: "spleen",
    7: "kidneys",
    9: "pancreas",
    10: "urinary_bladder",
    11: "duodenum",
    12: "gallbladder",
}
NAME_TO_INDEX = {v: k for k, v in ORGAN_NAMES.items()}
NAME_TO_INDEX["kidneys"] = 7  # project-specific convention


# ------------------------------- helpers ------------------------------------ #

def get_organ_index_from_name(organ_name: str) -> Optional[int]:
    return NAME_TO_INDEX.get(organ_name)


def get_conditioning_organs(
    generation_order: List[int], target_organ_index: int
) -> List[int]:
    if target_organ_index not in generation_order:
        raise ValueError(
            f"Target organ {target_organ_index} not in generation order: {generation_order}"
        )
    pos = generation_order.index(target_organ_index)
    return generation_order[:pos]

def add_conditioning_prediction_path(config, data_list):
    organ_name = config.task
    target_organ_index = get_organ_index_from_name(organ_name)
    if target_organ_index is None:
        raise ValueError(f"Unknown organ task name: {organ_name}")

    generation_order = list(config.data.generation_order)
    conditioning_organs = get_conditioning_organs(generation_order, target_organ_index)
    
    for organ_idx in conditioning_organs:
        organ_name = ORGAN_NAMES.get(organ_idx, str(organ_idx))
        organ_predictions_base = getattr(config, organ_name).predictions_dir
        
        all_pred_paths = os.listdir(organ_predictions_base)
        
        for data_item in data_list:
            image_path = data_item[Keys.IMAGE]
            filename = os.path.basename(image_path)
            # pred_path = os.path.join(pred_path_base, organ_name, filename)

            # Go through all the files in pred_path and find the one that contains the name
            # pred_path = None
            # for candidate in all_pred_paths:
            #     if filename in candidate:
            #         pred_path = os.path.join(organ_predictions_base, candidate)
            #         break
            pred_path = os.path.join(organ_predictions_base, f"{filename.replace('.nii.gz', '')}_pred.nii.gz")
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f"Prediction file for {filename} not found in {organ_predictions_base}")
            
            key_name = f"pred_{organ_name}"
            data_item[key_name] = pred_path
    return data_list
# ---------------------------- transforms ------------------------------------ #


def build_full_transform_pipeline(config, train: bool, rank: int) :
    """
    Single function that builds the entire pipeline:
      load → spacing → orientation → CC(labels) → harmonize → optional axis-crop
      → foreground crop → resize → add spacing tensor → filter+relabel
      → SDF (cpu) → EnsureTyped → optional SaveImaged
    Notes:
      * Use nearest for all resampling (masks).
      * KeepLargestConnectedComponentd on LABEL only.
      * No functools.partial or Lambda — all custom steps are MapTransforms.
    """
    has_body = bool(getattr(config.data, "body_filled_channel", False))
    data_keys = [Keys.IMAGE, Keys.LABEL] + (["body_filled_mask"] if has_body else [])

    all_organ_keys = [(f"Image_{v}", f"Label_{v}") for k, v in ORGAN_NAMES.items()]
    all_organ_keys = [item for sublist in all_organ_keys for item in sublist]

    all_organ_indices = list(ORGAN_NAMES.keys())

    organ_name = config.task
    if organ_name != "all_organs":
        target_organ_index = get_organ_index_from_name(organ_name)
        if target_organ_index is None:
            raise ValueError(f"Unknown organ task name: {organ_name}")

        generation_order = list(config.data.generation_order)
        conditioning_organs = get_conditioning_organs(generation_order, target_organ_index)

        if rank == 0:
            target_name = ORGAN_NAMES.get(target_organ_index, str(target_organ_index))
            logging.info("=" * 60)
            logging.info(
                f"Configuring organ filtering for: {target_name} (idx {target_organ_index})"
            )
            logging.info(
                f"Position: {generation_order.index(target_organ_index) + 1}/{len(generation_order)}"
            )
            logging.info(
                "Conditioning on: %s",
                [ORGAN_NAMES.get(i, str(i)) for i in conditioning_organs],
            )
            logging.info("=" * 60)
    else:
        target_organ_index = None
        conditioning_organs = []

        if rank == 0:
            logging.info("=" * 60)
            logging.info(
                f"Configuring for all organs generation."
            )
            logging.info("=" * 60)

    # Choose crop source
    crop_source = "body_filled_mask" if has_body else Keys.IMAGE

    t = [
        # --- pre-SDF (cache-friendly) ---
        monai.transforms.LoadImaged(keys=data_keys),
        monai.transforms.EnsureChannelFirstd(keys=data_keys),
        monai.transforms.Spacingd(
            keys=data_keys, pixdim=config.data.pixdim, mode="nearest"
        ),
        monai.transforms.Orientationd(keys=data_keys, axcodes=config.data.orientation),
        monai.transforms.KeepLargestConnectedComponentd(keys=[Keys.IMAGE, Keys.LABEL]),
        # dataset-dependent harmonization (no partial)
        HarmonizeLabelsd(keys=[Keys.IMAGE, Keys.LABEL], kidneys_same_index=True),
        # Probe(keys=[Keys.IMAGE, Keys.LABEL] + (["body_filled_mask"] if has_body else [])),
        # optional axis crop
        (
            CropForegroundAxisd(
                keys=data_keys,
                source_key=Keys.IMAGE,
                axis=config.data.slice_axis,
                margin=5,
            )
            if has_body
            else monai.transforms.Identityd(keys=data_keys)
        ),
        monai.transforms.CropForegroundd(keys=data_keys, source_key=crop_source, margin=5),
        monai.transforms.Resized(keys=data_keys, spatial_size=config.data.roi_size, mode="nearest"),
        AddSpacingTensord(ref_key=Keys.IMAGE),
        (
            FilterAndRelabeld(
                image_key=Keys.IMAGE,
                label_key=Keys.LABEL,
                conditioning_organs=conditioning_organs,
                target_organ=target_organ_index,
            )
            if target_organ_index is not None
            else DivideFilterAndRelabeld(
                image_key=Keys.IMAGE,
                label_key=Keys.LABEL,
                generation_sequence=config.data.generation_order,
                target_organs=all_organ_indices,
                label_to_organ_name=ORGAN_NAMES,
            )
        ),
        # --- post-filtering ---
        MaskToSDFd(keys=data_keys + all_organ_keys if organ_name == "all_organs" else data_keys, spacing_key="spacing_tensor", device=torch.device("cpu")),
        monai.transforms.FromMetaTensord(keys=data_keys + all_organ_keys if organ_name == "all_organs" else data_keys, data_type="tensor"),
        monai.transforms.ToMetaTensord(keys=data_keys + all_organ_keys if organ_name == "all_organs" else data_keys),
        EnsureAllTorchd(print_changes=False),
        monai.transforms.EnsureTyped(keys=data_keys + ["spacing_tensor"] + all_organ_keys if organ_name == "all_organs" else data_keys + ["spacing_tensor"], track_meta=True),
   
    ]

    if config.training.inference_mode:
        pred_keys = [f"pred_{ORGAN_NAMES.get(idx, str(idx))}" for idx in conditioning_organs]
        if len(pred_keys) != 0:
            t.extend(
                [
                    monai.transforms.LoadImaged(keys=pred_keys),
                    monai.transforms.EnsureChannelFirstd(keys=pred_keys),
                    # monai.transforms.Spacingd(keys=pred_keys, pixdim=config.data.pixdim, mode="nearest"),
                    monai.transforms.Orientationd(keys=pred_keys, axcodes=config.data.orientation),
                    CombineKeysd(keys=pred_keys, result_key=Keys.IMAGE, as_binary=True),
                    # monai.transforms.Resized(keys=[Keys.IMAGE], spatial_size=config.data.roi_size, mode="nearest"),
                    EnsureAllTorchd(),
                    monai.transforms.EnsureTyped(keys=[Keys.IMAGE], track_meta=True),
                ]
            )

    # Optional saving for debug/inspection
    if getattr(config.data, "save_data", False):
        phase = "training_samples" if train else "validation_samples"

        def _save(key: str, postfix: str, subdir: str):
            return monai.transforms.SaveImaged(
                keys=[key],
                meta_keys=[f"{key}_meta_dict"],
                output_dir=os.path.join(
                    config.training.save_dir, config.task, phase, subdir
                ),
                output_postfix=postfix,
                separate_folder=False,
            )

        t.extend(
            [
                _save(Keys.IMAGE, "image", "images"),
                _save(Keys.LABEL, "label", "labels"),
            ]
        )
        if has_body:
            t.append(_save("body_filled_mask", "body_filled_mask", "body_filled_masks"))

        if target_organ_index is None:
            for organ_idx in all_organ_indices:
                organ_name = ORGAN_NAMES.get(organ_idx, str(organ_idx))
                image_key = f"Image_{organ_name}"
                label_key = f"Label_{organ_name}"
                t.extend(
                    [
                        _save(image_key, f"image_{organ_name}", f"images_{organ_name}"),
                        _save(label_key, f"label_{organ_name}", f"labels_{organ_name}"),
                    ]
                )

    return monai.transforms.Compose(t)


# ------------------------------ dataloaders --------------------------------- #


def get_dataloaders(config, aim_logger, rank: int):
    """
    Single-level persistent cache per organ task.

    Cache path example:
      {config.data.cache_dir}/{config.task}_idx{target_organ_index}/{train|val}
    """
    # ---------------- files ----------------
    has_body = bool(getattr(config.data, "body_filled_channel", False))
    train_files = list_from_jsonl(
        config.data.train_jsonl,
        image_key="mask",
        label_key="mask",
        include_body_filled=has_body,
        body_filled_key="body_filled_mask",
    )
    val_files = list_from_jsonl(
        config.data.val_jsonl,
        image_key="mask",
        label_key="mask",
        include_body_filled=has_body,
        body_filled_key="body_filled_mask",
    )

    if config.training.inference_mode:
        if rank == 0:
            logging.info("Inference mode: Adding conditioning predictions to validation files")
        val_files = add_conditioning_prediction_path(config, val_files)

    if aim_logger is not None and rank == 0:
        aim_logger.experiment.track(
            aim.Text(json.dumps(train_files, indent=2)), name="Training Files", step=1
        )
        aim_logger.experiment.track(
            aim.Text(json.dumps(val_files, indent=2)), name="Validation Files", step=1
        )
        logging.info(
            f"Training files: {len(train_files)}, Validation files: {len(val_files)}"
        )

    if getattr(config.experiment, "debug", False):
        train_files = train_files[:4]
        val_files = val_files[:4]
        if rank == 0:
            logging.info("DEBUG mode: using small subset.")
            logging.info(
                f"Training files: {len(train_files)}, Validation files: {len(val_files)}"
            )

    # Optional cap on validation size
    vmax = getattr(config.evaluation, "validation_max_num_samples", None)
    if vmax is not None and len(val_files) > vmax:
        if rank == 0:
            logging.info(f"Validation files before sampling: {len(val_files)}")
        rng = np.random.default_rng(
            config.seed if getattr(config, "seed", None) else 42
        )
        val_files = list(rng.choice(val_files, size=vmax, replace=False))
        if rank == 0:
            logging.info(f"Validation files after sampling: {len(val_files)}")

    # ---------------- organ / cache ----------------
    organ_name = config.task
    if organ_name != "all_organs":      
        target_organ_index = get_organ_index_from_name(organ_name)
        if target_organ_index is None:
            raise ValueError(f"Unknown organ task name: {organ_name}")

        base_cache_root = os.path.join(
            config.data.cache_dir, f"{organ_name}_idx{target_organ_index}"
        )
        if rank == 0:
            pos = config.data.generation_order.index(target_organ_index) + 1
            logging.info(
                f"Cache root: {base_cache_root} | Organ {organ_name} (pos {pos}/{len(config.data.generation_order)})"
            )
    else:
        base_cache_root = os.path.join(
            config.data.cache_dir, f"{organ_name}"
        )
        if rank == 0:
            logging.info(
                f"Cache root: {base_cache_root} | All organs"
            )
    os.makedirs(base_cache_root, exist_ok=True)

    # ---------------- transforms ----------------
    train_transform = build_full_transform_pipeline(
        config, train=True, rank=rank
    )
    val_transform = build_full_transform_pipeline(
        config, train=False, rank=rank
    )

    # ---------------- datasets (single-level persistent cache) ----------------
    train_ds = monai.data.CacheNTransDataset(
        data=train_files, transform=train_transform, cache_dir=base_cache_root, cache_n_trans=13
    )
    val_ds = monai.data.CacheNTransDataset(
        data=val_files, transform=val_transform, cache_dir=base_cache_root, cache_n_trans=13
    )

    # ---------------- dataloaders ----------------
    train_loader = auto_dataloader(
        train_ds,
        batch_size=config.data.batch_size_per_gpu * config.training.num_gpus,
        num_workers=config.data.num_workers_per_gpu * config.training.num_gpus,
        collate_fn=collate_meta_tensor,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=config.data.shuffle_train_data,
    )

    val_loader = auto_dataloader(
        val_ds,
        batch_size=config.data.val_batch_size,
        num_workers=config.data.val_num_workers,
        collate_fn=collate_meta_tensor,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=False,
    )

    if rank == 0:
        logging.info(
            f"Datasets ready. Train samples: {len(train_ds)}, Val samples: {len(val_ds)}"
        )

    return train_loader, val_loader


# endregion


class VolumeSpacingEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        is_discrete=False,
        volume_bins=[0, 1327, 1589, 1888, 6000],  # List of bin edges, e.g., [0, 100, 300, 500, 1000]
        use_soft_binning=False,  # Smooth vs hard bin selection
        temperature=1.0,  # For soft binning - controls smoothness
        convert_to_cm3=False,  # If True, convert volume from mm³ to cm³
    ):
        """
        Args:
            embed_dim: Dimension of output embedding
            volume_bins: List/array of bin edges in ml. If None, uses continuous MLP.
                        E.g., [0, 100, 300, 500] creates 3 bins:
                        - Bin 0: 0-100ml (small)
                        - Bin 1: 100-300ml (medium)
                        - Bin 2: 300-500ml (large)
            use_soft_binning: If True, uses soft assignment with smooth transitions
            temperature: Controls sharpness of soft binning (lower = sharper)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_soft_binning = use_soft_binning
        self.temperature = temperature

        # Determine if using discrete bins or continuous
        self.is_discrete = is_discrete
        self.convert_to_cm3 = convert_to_cm3

        if self.is_discrete:
            # Register bin edges as buffer
            self.register_buffer(
                "bin_edges", torch.tensor(volume_bins, dtype=torch.float32)
            )
            self.num_bins = len(volume_bins) - 1

            # Learnable embeddings for each volume bin
            self.volume_embedding = nn.Embedding(self.num_bins, embed_dim // 2)

            # MLP for spacing
            self.spacing_mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.SiLU(),
                nn.Linear(64, embed_dim // 2),
            )

            # Fusion layer
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        else:
            # Continuous mode - original MLP
            self.mlp = nn.Sequential(
                nn.Linear(4, 64),
                nn.SiLU(),
                nn.Linear(64, 128),
                nn.SiLU(),
                nn.Linear(128, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def _get_volume_embedding_hard(self, volume):
        """Hard binning: each volume assigned to exactly one bin"""
        # Find which bin each volume belongs to
        bin_indices = torch.searchsorted(self.bin_edges[:-1], volume, right=False)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # Get embedding for assigned bin
        return self.volume_embedding(bin_indices)  # (B, embed_dim//2)

    def _get_volume_embedding_soft(self, volume):
        """
        Soft binning: weighted combination of nearby bin embeddings
        Uses distance-based weighting with learnable temperature
        """
        # Compute distances to all bin centers
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # (num_bins,)

        # Distance from each volume to each bin center
        # volume: (B,), bin_centers: (num_bins,)
        distances = torch.abs(
            volume.unsqueeze(-1) - bin_centers.unsqueeze(0)
        )  # (B, num_bins)

        # Convert distances to weights using softmax (closer = higher weight)
        weights = torch.softmax(-distances / self.temperature, dim=-1)  # (B, num_bins)

        # Get all embeddings and compute weighted sum
        all_embeddings = self.volume_embedding.weight  # (num_bins, embed_dim//2)
        volume_embed = torch.matmul(weights, all_embeddings)  # (B, embed_dim//2)

        return volume_embed

    def forward(self, volume, spacing, return_bin_info=False):
        """
        Args:
            volume: (B,) or (B, 1) - organ volume in ml
            spacing: (B, 3) - [spacing_x, spacing_y, spacing_z] in mm
            return_bin_info: If True, also return bin indices/weights for analysis
        Returns:
            embedding: (B, embed_dim)
            bin_info: (optional) dict with 'bin_indices' or 'bin_weights'
        """
        if volume.dim() == 2:
            volume = volume.squeeze(-1)  # (B,)
            
        if self.convert_to_cm3:
            volume = volume / 1000.0  # convert mm³ to cm³

        if self.is_discrete:
            # Get volume embedding (hard or soft)
            if self.use_soft_binning:
                volume_embed = self._get_volume_embedding_soft(volume)
                if return_bin_info:
                    bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                    distances = torch.abs(
                        volume.unsqueeze(-1) - bin_centers.unsqueeze(0)
                    )
                    bin_weights = torch.softmax(-distances / self.temperature, dim=-1)
            else:
                volume_embed = self._get_volume_embedding_hard(volume)
                if return_bin_info:
                    bin_indices = torch.searchsorted(
                        self.bin_edges[:-1], volume, right=False
                    )
                    bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

            # Process spacing
            spacing_embed = self.spacing_mlp(spacing)  # (B, embed_dim//2)

            # Combine
            combined = torch.cat([volume_embed, spacing_embed], dim=-1)
            embedding = self.fusion_mlp(combined)

            if return_bin_info:
                bin_info = {
                    "bin_weights" if self.use_soft_binning else "bin_indices": (
                        bin_weights if self.use_soft_binning else bin_indices
                    )
                }
                return embedding, bin_info
        else:
            # Continuous mode
            volume_spacing = torch.cat([volume.unsqueeze(-1), spacing], dim=-1)
            embedding = self.mlp(volume_spacing)

            if return_bin_info:
                return embedding, {}

        return embedding

    def get_bin_labels(self):
        """Get human-readable labels for bins"""
        if not self.is_discrete:
            return None

        labels = []
        for i in range(self.num_bins):
            low = self.bin_edges[i].item()
            high = self.bin_edges[i + 1].item()
            labels.append(f"Bin {i}: {low:.0f}-{high:.0f}ml")
        return labels


def prepare_batch(batch, device=None, non_blocking=True, add_body_filled=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[Keys.LABEL].to(device, non_blocking=non_blocking)
    
    # labels = torch.cat([labels, images[:, 0:1, ...]], dim=1)  # add image SDF as 2nd channel

    if "body_filled_mask" in batch and add_body_filled:
        body_filled_mask = batch["body_filled_mask"].to(device, non_blocking=non_blocking)
        images = torch.cat([images, body_filled_mask], dim=1)
        
    spacing = batch["spacing_tensor"]
    
    spacing = spacing.to(device, non_blocking=non_blocking)
    
    # if type(images) is torch.Tensor turn into MetaTensor
    return images, labels, spacing


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def build_model(config, rank, train_loader):
    net = DiffusionModelUNetFiLM(
        spatial_dims=config.model.params.spatial_dims,
        in_channels=config.model.params.in_channels + config.model.params.out_channels,  # add image SDF as extra channel
        out_channels=config.model.params.out_channels,
        channels=config.model.params.features,
        attention_levels=config.model.params.attention_levels,
        num_res_blocks=1,
        transformer_num_layers=0,
        num_head_channels=config.model.params.num_head_channels,
        with_conditioning=False,
        cross_attention_dim=None,
    )

    train_scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.diffusion_steps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule=config.diffusion.beta_schedule,
        clip_sample=False,
        prediction_type=config.diffusion.model_mean_type,
    )
    # FiLM adapter instead of VolumeSpacingEmbedding
    film_adapter = FiLMAdapter(
        unet_channels=config.model.params.features,  # e.g., (32, 64, 64, 64)
        embed_dim=256,
        volume_mean=1625.45,
        volume_std=535.51,
        use_log_volume=True,  # Try True if volumes vary widely
    )
    optimizer = optim.AdamW(
        itertools.chain(net.parameters(), film_adapter.parameters()),
        config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    lr_scheduler = None
    if config.lr_scheduler.name == "LinearWarmupCosineAnnealingLR":
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=config.lr_scheduler.warmup_epochs,
            max_epochs=config.lr_scheduler.max_epochs,
        )

    scaler = GradScaler(enabled=config.amp.enabled)

    if rank == 0:
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in film_adapter.parameters() if p.requires_grad)

        logging.info(f"[Rank {rank}] Model parameters: {num_params / 1e6:.2f}M")

    net = auto_model(net)
    film_adapter = auto_model(film_adapter)
    optimizer = auto_optim(optimizer)

    def train_step(engine, batchdata):
        accum = config.training.accumulate_grad_steps
        images, labels, spacing_tensor = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking, config.model.params.in_channels > 1
        )
        images = images.float()
        labels = labels.float()
        voxel_volume = spacing_tensor.prod(dim=1)

        y_logits = sdf_to_mask(labels)

        gt_volume = y_logits.sum(dim=[1,2,3,4]) * voxel_volume  # (B,)   
        gt_volume = gt_volume / 1000.0  # convert mm³ to ml

        film_params = engine.film_adapter(gt_volume, spacing_tensor)

        engine.network.train()
        if engine.state.iteration == 1:
            engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        if torch.rand(1).item() < config.diffusion.condition_drop_prob:
            images = torch.zeros_like(images)

        timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (labels.shape[0],), device=images.device).long()
        noise = torch.randn_like(labels)

        noisy_labels = train_scheduler.add_noise(
            original_samples=labels,
            noise=noise,
            timesteps=timesteps,
        )
        noisy_labels = torch.cat([noisy_labels, images], dim=1)  # concat image SDF as extra channel

        pred_xstart = engine.network(
            x = noisy_labels,
            timesteps = timesteps,
            context= None,
            film_params=film_params,
        )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)

        # loss = nn.MSELoss()(pred_xstart, noise)  # L2 loss on noise prediction
        loss_l1 = nn.L1Loss()(pred_xstart, labels)
        pred_logits = pred_xstart * 10.0
        pos_weight = (y_logits == 0).sum() / ((y_logits == 1).sum() + 1e-5)

        loss_bce = engine.bce(pred_logits, y_logits, pos_weight=pos_weight)

        # loss_ci = engine.ci_dice(y_logits, torch.sigmoid(pred_logits))
        loss_ci = 0

        pred_prob = torch.sigmoid(pred_logits)
        image_logits = sdf_to_mask(images[:, 0:1, ...])  # assuming first channel is the mask SDF
        loss_reverse_dice = 1 - engine.reverse_dice_loss(pred_prob, image_logits)

        pred_volume = pred_prob.sum(dim=[1,2,3,4]) * voxel_volume  # (B,)
        pred_volume = pred_volume / 1000.0  # convert mm³ to ml
        
        vol_difference = gt_volume - pred_volume
        gt_volume_norm = (gt_volume - 1625) / 535
        pred_volume_norm = (pred_volume - 1625) / 535
        loss_volume = F.mse_loss(pred_volume_norm, gt_volume_norm)

        loss = (loss_l1 + loss_bce) + loss_ci + loss_reverse_dice + loss_volume

        # loss_l1 = loss_bce = loss_ci = loss_reverse_dice = loss_volume = torch.tensor(0.0, device=images.device)
        # vol_difference = torch.zeros_like(gt_volume)
        loss = loss / accum

        engine.fire_event(IterationEvents.LOSS_COMPLETED)

        # backward
        scaler.scale(loss).backward() if engine.amp else loss.backward()

        # optimizer step on the last micro-step
        if engine.state.iteration % accum == 0:
            if engine.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        out = {
            "loss": loss * accum,
            "loss_bce": loss_bce,
            "loss_l1": loss_l1,
            "loss_ci": loss_ci,
            "loss_reverse_dice": loss_reverse_dice,
            "loss_volume": loss_volume,
            "gt_vol-pred_vol": vol_difference.mean(),
        }
        return out

    trainer = Trainer(
        device=idist.device(),
        max_epochs=config.training.epochs,
        data_loader=train_loader,
        prepare_batch=prepare_batch,
        iteration_update=train_step,
        additional_metrics=None,
        amp=config.amp.enabled,
    )
    trainer.network = net
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
    trainer.scaler_ = scaler
    trainer.config = config
    trainer.optim_set_to_none = config.optimizer.set_to_none

    trainer.bce = nn.functional.binary_cross_entropy_with_logits
    trainer.dice_loss = monai.losses.DiceLoss(
        sigmoid=True, include_background=False, reduction="mean"
    )
    trainer.reverse_dice_loss = monai.losses.DiceLoss(
        sigmoid=False, include_background=False, reduction="mean"
    )
    trainer.ci_dice = SoftCIDice()
    trainer.film_adapter = film_adapter

    if bool(config.model.adverserial_train.enabled):
        # Lightweight PatchGAN discriminator that looks at SDF logits (same channel count as your labels/preds)
        d_net = PatchDiscriminator(
            spatial_dims=config.model.params.spatial_dims,
            num_layers_d=3,
            channels=32,  # SDF/pred channels
            in_channels=1,
            out_channels=1,
            norm="INSTANCE",
            activation=("LeakyReLU", {"negative_slope": 0.2, "inplace": False}),
        )
        d_net = auto_model(d_net)

        d_optimizer = optim.AdamW(
            d_net.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        d_optimizer = auto_optim(d_optimizer)

        adv_loss = PatchAdversarialLoss()  # stable for medical patches

        # expose on trainer so we can use inside train_step
        # (no change to return signature)
        trainer.d_network = d_net
        trainer.d_optimizer = d_optimizer
        trainer.adv_loss = adv_loss
        trainer.adv_lambda = float(config.model.adverserial_train.adverserial_loss_weight)
        trainer.adv_start_epoch = int(config.model.adverserial_train.start_epoch)
    else:
        trainer.d_network = None
        trainer.d_optimizer = None
        trainer.adv_loss = None
        trainer.adv_lambda = 0.0
        trainer.adv_start_epoch = 0

    return net, optimizer, lr_scheduler, trainer, film_adapter


def get_evaluator(cfg, model, val_loader, film_adapter):
    target_ch = cfg.model.params.out_channels

    # Build the evaluator
    post = transforms.Compose(
        [
            # transforms.Lambdad(keys=Keys.PRED, func=torch.softmax),
            # transforms.AsDiscreted(keys=Keys.PRED, threshold=0.5),
            # transforms.AsDiscreted(keys=Keys.LABEL, threshold=0.5),
            # transforms.AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=target_ch),
            # transforms.AsDiscreted(keys=Keys.LABEL, argmax=True, to_onehot=target_ch),
            transforms.Identityd(keys=[Keys.IMAGE, Keys.LABEL, Keys.PRED]),
            SmoothColonMaskd(
                keys=[Keys.PRED],
                iterations=10,
                connectivity=2,
                min_neck_thickness=0,
                data_orientation="RAS",
            ) if cfg.task == "colon" else transforms.Identityd(keys=[Keys.PRED]),
            # transforms.KeepLargestConnectedComponentd(
            #     keys=[Keys.PRED],
            #     is_onehot=False,
            #     num_components=1 if not cfg.task == 'kidneys' else 2,
            # )if cfg.task != "colon" else transforms.Identityd(keys=[Keys.PRED]),
        ]
    )

    metrics = {
        "Mean Dice": MeanDice(
            include_background=True,
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=target_ch,
        ),
        # "Mean Volume Error": MeanAbsoluteError(
        #     output_transform=lambda x: (
        #         x["Predicted Volume Difference"].abs().mean(),
        #         torch.zeros(1, device=x["Predicted Volume Difference"].device),
        #     )
        # ),
    }

    inference_scheduler = DDIMScheduler(
        num_train_timesteps=cfg.diffusion.diffusion_steps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule=cfg.diffusion.beta_schedule,
        clip_sample=False,
        prediction_type=cfg.diffusion.model_mean_type,
    )
    inference_scheduler.set_timesteps(num_inference_steps=cfg.diffusion.ddim_steps)

    # --- One evaluation step: sample conditioned predictions ---
    @torch.no_grad()
    def _eval_step(engine, batchdata):
        # 1) prepare
        config = engine.config
        image, masks, spacing_tensor = engine.prepare_batch(
            batchdata,
            engine.state.device,
            engine.non_blocking,
            config.model.params.in_channels > 1,
        )
        image = image.float()
        masks = masks.float()
        spacing_tensor = spacing_tensor.float()

        voxel_volume = spacing_tensor.prod(dim=1)
        y_logits = sdf_to_mask(masks)
        gt_volume = y_logits.sum(dim=[1, 2, 3, 4]) * voxel_volume  # (B,)
        gt_volume = gt_volume / 1000.0  # convert mm³ to ml
        
        film_params = engine.film_adapter(gt_volume, spacing_tensor)

        cfg = config.diffusion.guidance_scale

        engine.network.eval()

        all_next_timesteps = torch.cat(
            (
                engine.scheduler.timesteps[1:],
                torch.tensor([0], dtype=engine.scheduler.timesteps.dtype),
            )
        )
        pred = torch.randn_like(masks)
        for t, next_t in iter(zip(engine.scheduler.timesteps, all_next_timesteps)):
            model_input = torch.cat([pred, image], dim=1)  # concat image SDF as extra channel # modify to include classifier guidance later

            model_output = engine.network(x=model_input, timesteps=torch.Tensor((t,)).to(image.device), context=None, film_params=film_params)
            if cfg != 1.0:
                # perform guidance
                image_fill = torch.zeros_like(image)
                image_fill.fill_(0.0)
                model_input_uncond = torch.cat(
                    [pred, image_fill], dim=1
                )  # zeroed SDF for uncond
                uncond_output = engine.network(x=model_input_uncond, timesteps=torch.Tensor((t,)).to(image.device))
                model_output = uncond_output + cfg * (model_output - uncond_output)

            pred, _ = engine.scheduler.step(
                model_output, t, pred
            )

        engine.state.output = {Keys.IMAGE: sdf_to_mask(image), Keys.LABEL: sdf_to_mask(masks)}
        engine.state.output["SDF"] = pred.clone().detach()
        pred = sdf_to_mask(pred)
        pred_volume = pred.sum(dim=[1,2,3,4]) * voxel_volume # (B,)
        pred_volume = pred_volume / 1000.0  # convert mm³ to ml

        engine.state.output[Keys.PRED] = pred
        engine.state.output["Predicted Volume Difference"] = (gt_volume - pred_volume).unsqueeze(1)
        # engine.state.output[Keys.PRED] = engine.inferer(image, engine.network, pred_type="ddim_sample")

        if config.experiment.debug:
            raw_logits = engine.state.output[Keys.PRED]
            logging.info(
                f"Logits stats: min={raw_logits.min():.4f}, max={raw_logits.max():.4f}, mean={raw_logits.mean():.4f}"
            )
            # Log Shapes and Unique Values
            logging.info(f"Output shape: {engine.state.output[Keys.PRED].shape}")
            logging.info(f"Image shape: {image.shape}")
            logging.info(f"Masks shape: {masks.shape}")

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output

    evaluator = Evaluator(
        device=idist.device(),
        val_data_loader=val_loader,
        iteration_update=_eval_step,
        postprocessing=post,
        key_val_metric=metrics,
        amp=bool(cfg.amp.enabled),
        prepare_batch=prepare_batch,
    )

    evaluator.network = model
    evaluator.config = cfg
    evaluator.scheduler = inference_scheduler
    evaluator.film_adapter = film_adapter
    # evaluator.inferer = monai.inferers.SlidingWindowInferer(
    #     roi_size=cfg.data.roi_size, sw_batch_size=1, overlap=0
    # )
    evaluator.scheduler = inference_scheduler
    return evaluator


# endregion


# ---------------------------
# Distributed run
# ---------------------------
def _distributed_run(rank: int, cfg):
    device = idist.device()
    world_size = idist.get_world_size()

    setup_logger(
        name="training_logger",
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s: %(message)s",
        reset=True,
    )
    logging.info(f"[Rank {rank}] Running on device: {device}, world size: {world_size}")

    if cfg.seed is not None:
        set_determinism(seed=int(cfg.seed))

    # Optional Aim/profiler
    aim_logger = (
        get_aim_logger(cfg)
        if (rank == 0 and get_aim_logger is not None and bool(cfg.logging.use_aim))
        else None
    )

    # Data
    train_loader, val_loader = get_dataloaders(cfg, aim_logger, rank)
    logging.info(f"[Rank {rank}] Train/Val loaders ready") 

    model, optimizer, lr_scheduler, trainer, film_adapter = build_model(cfg, rank, train_loader)

    ema_model = None
    if bool(cfg.ema.enable):
        ema_model = AveragedModel(
            model,
            avg_fn=lambda avg_p, new_p, _: avg_p.mul_(float(cfg.ema.rate)).add_(
                new_p, alpha=1 - float(cfg.ema.rate)
            ),
        )

    evaluator = get_evaluator(
        cfg, ema_model if bool(cfg.ema.enable) else model, val_loader, film_adapter
    )

    savables = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "trainer": trainer,
        "ema_model": ema_model,
        "discriminator": trainer.d_network,
        "d_optimizer": trainer.d_optimizer,
        "vol_embed_net": film_adapter,
    }

    resumed = resume_from_checkpoint(
        stage_name=cfg.task, config=cfg, to_load=savables, rank=rank
    )

    attach_handlers(
        trainer=trainer,
        val_evaluator=evaluator,
        objects_to_save=savables,
        cfg=cfg,
        aim_logger=aim_logger,
        rank=rank,
        stage_name=cfg.task,
        aim_log_items=[
            (trainer, Events.ITERATION_COMPLETED, "Iter Loss", Keys.LOSS),
            *[
                (trainer, Events.EPOCH_COMPLETED, f"Epoch {key}", key)
                for key in [
                    "loss",
                    "loss_bce",
                    "loss_l1",
                    "loss_ci",
                    "loss_reverse_dice",
                    "loss_volume",
                    "gt_vol-pred_vol",
                ]
            ],
        ],
        metric_name="Mean Dice",
        ema_model=ema_model,
        step_lr=True,
    )

    if bool(cfg.training.inference_mode):
        if rank == 0:
            logging.info("Running inference only")

        evaluator.run()
        return

    if rank == 0:
        logging.info(
            f"[Rank {rank}] >>> Stage 1 training for {cfg.training.epochs} epochs"
        )

    idist.utils.barrier()
    trainer.run()


# ---------------------------
# Hydra entry-point
# ---------------------------


if __name__ == "__main__":

    def derive_experiment_metadata(cfg: DictConfig) -> None:
        parts = [cfg.experiment.name, cfg.constraint, "iterative"]

        cfg.experiment.name = "-".join(parts)
        # drop the version itself, leave tags for Aim
        cfg.experiment.tags.extend(parts[2:])
        cfg.experiment.tags.append(cfg.task)

        cfg.training.save_dir = os.path.join(
            cfg.training.save_dir,
            cfg.experiment.name.lower(),
            f"{cfg.experiment.version}",
        )
        os.makedirs(cfg.training.save_dir, exist_ok=True)

        if cfg.experiment.debug:
            cfg.experiment.name = f"debug_{cfg.experiment.name}"
            cfg.training.save_dir = os.path.join(cfg.training.save_dir, "debug")
            cfg.evaluation.validation_interval = 1
            cfg.experiment.tags.append("debug")
            # cfg.training.resume = None  # Don't resume from any previous run
            # set the cfg.[cfg.task].resume to None
            setattr(cfg[cfg.task], "resume", None)

            logging.info(
                f""" {'-'* 50}
                        DEBUG MODE:
                        - Using only 10 training and validation samples.
                        - Validation interval set to 1 epoch.
                        {'-'* 50}
                        """
            )

    @hydra.main(
        version_base=None,
        config_path="/home/yb107/cvpr2025/DukeDiffSeg/configs/diffunet_v6",
        config_name="config",
    )
    def main(cfg: DictConfig):
        # pretty-print resolved config on rank 0
        derive_experiment_metadata(cfg)

        if idist.get_rank() == 0:
            print("\n===== Resolved Config =====\n")
            print(OmegaConf.to_yaml(cfg, resolve=True))
            print("==========================\n")

        nproc = int(cfg.training.num_gpus)
        backend = (
            "nccl"
            if (
                str(cfg.training.device).startswith("cuda")
                and torch.cuda.is_available()
            )
            else "gloo"
        )

        with idist.Parallel(backend=backend, nproc_per_node=nproc, master_port=2225) as parallel:
            parallel.run(_distributed_run, cfg)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    main()
