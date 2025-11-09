# region imports

# Iterative Generation of Abdominal Organs

import functools
import json
import logging
import os
import sys
import time
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
from utils.monai_transforms import CropForegroundAxisd, SmoothColonMaskd, HarmonizeLabelsd, AddSpacingTensord, FilterAndRelabeld, Probe, EnsureAllTorchd, CombineKeysd
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

    organ_name = config.task
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

    # Choose crop source
    crop_source = "body_filled_mask" if has_body else Keys.IMAGE
    
    t = [
        # --- pre-SDF (cache-friendly) ---
        monai.transforms.LoadImaged(keys=data_keys),
        monai.transforms.EnsureChannelFirstd(keys=data_keys),
        monai.transforms.Spacingd(keys=data_keys, pixdim=config.data.pixdim, mode="nearest"),
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
            else monai.transforms.Transform.identity()
        ),
        # Probe(keys=[Keys.IMAGE, Keys.LABEL] + (["body_filled_mask"] if has_body else [])),
        # foreground crop + resize
        monai.transforms.CropForegroundd(keys=data_keys, source_key=crop_source, margin=5),
        monai.transforms.Resized(keys=data_keys, spatial_size=config.data.roi_size, mode="nearest"),
        # attach spacing metadata (no Lambda)
        AddSpacingTensord(ref_key=Keys.IMAGE),
        # organ filtering (no partial)
        FilterAndRelabeld(
            image_key=Keys.IMAGE,
            label_key=Keys.LABEL,
            conditioning_organs=conditioning_organs,
            target_organ=target_organ_index,
        ),
        # --- post-filtering ---
        MaskToSDFd(keys=data_keys, spacing_key="spacing_tensor", device=torch.device("cpu")),
        monai.transforms.FromMetaTensord(keys=data_keys, data_type="tensor"),
        monai.transforms.ToMetaTensord(keys=data_keys),
        EnsureAllTorchd(print_changes=False),
        monai.transforms.EnsureTyped(keys=data_keys + ["spacing_tensor"], track_meta=True),
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
    target_organ_index = get_organ_index_from_name(organ_name)
    if target_organ_index is None:
        raise ValueError(f"Unknown organ task name: {organ_name}")

    base_cache_root = os.path.join(
        config.data.cache_dir, f"{organ_name}_idx{target_organ_index}"
    )
    os.makedirs(base_cache_root, exist_ok=True)

    if rank == 0:
        pos = config.data.generation_order.index(target_organ_index) + 1
        logging.info(
            f"Cache root: {base_cache_root} | Organ {organ_name} (pos {pos}/{len(config.data.generation_order)})"
        )

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


def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[Keys.LABEL].to(device, non_blocking=non_blocking)
    if "body_filled_mask" in batch:
        body_filled_mask = batch["body_filled_mask"].to(device, non_blocking=non_blocking)
        images = torch.cat([images, body_filled_mask], dim=1)
    spacing = batch["spacing_tensor"].squeeze()
    spacing = spacing.to(device, non_blocking=non_blocking)
    
    # if type(images) is torch.Tensor turn into MetaTensor
    return images, labels, spacing


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def build_model(config, rank, train_loader):
    net = DiffUNet(
        spatial_dims=config.model.params.spatial_dims,
        in_channels=config.model.params.in_channels,
        out_channels=config.model.params.out_channels,
        features=config.model.params.features,
        # activation=tuple(config.model.params.activation),
        # dropout_rate=config.model.params.dropout_rate,
        # use_checkpointing=config.model.params.use_checkpointing,
        diffusion_steps=config.diffusion.diffusion_steps,
        beta_schedule=config.diffusion.beta_schedule,
        ddim_steps=config.diffusion.ddim_steps,
        image_size=config.data.roi_size,
        use_spacing_info=config.model.params.use_spacing_info,
        model_mean_type=config.diffusion.model_mean_type,
    )

    optimizer = optim.AdamW(
        net.parameters(),
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
        logging.info(f"[Rank {rank}] Model parameters: {num_params / 1e6:.2f}M")

    net = auto_model(net)
    optimizer = auto_optim(optimizer)

    def train_step(engine, batchdata):
        accum = config.training.accumulate_grad_steps

        images, labels, spacing_tensor = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking
        )
        images = images.float()
        labels_1hot = labels.float()
        # structure_mask = (images.sum(dim=1, keepdim=True) > 0).float()

        engine.network.train()
        if engine.state.iteration == 1:
            engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        # x_start = (labels_1hot) * 2 - 1
        x_start = labels_1hot

        x_t, t, noise = engine.network(x=x_start, pred_type="q_sample")

        if torch.rand(1).item() < config.diffusion.condition_drop_prob:
            images = torch.zeros_like(images)

        pred_xstart = engine.network(
            x=x_t,
            step=t,
            image=images,
            pred_type="denoise",
            spacing_tensor=spacing_tensor,
            guidance_scale=1.0,
        )
        # pred_noise = engine.network(
        #     x=x_t, step=t, image=images, pred_type="denoise", spacing_tensor=spacing_tensor, guidance_scale=1.0
        # )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)


        loss_l1 = nn.L1Loss()(pred_xstart, x_start)
        pred_logits = pred_xstart * 10.0
        y_logits = sdf_to_mask(labels_1hot)
        pos_weight = (y_logits == 0).sum() / ((y_logits == 1).sum() + 1e-5)

        # loss_bce = 0
        loss_bce = engine.bce(pred_logits, y_logits, pos_weight=pos_weight)

        # loss_ci = engine.ci_dice(y_logits, torch.sigmoid(pred_logits))
        loss_ci = 0
        
        pred_prob = torch.sigmoid(pred_logits)
        image_logits = sdf_to_mask(images[:, 0, ...].unsqueeze(1))  # assuming first channel is the mask SDF
        loss_reverse_dice = 1 - engine.reverse_dice_loss(pred_prob, image_logits)

        # volume = volume_from_mask_batched(pred_prob, spacing_tensor)
        # volume = 0
        # loss_geometry = loss_isoperimetric(
        #     surface_area, volume, alpha=1e-6, beta=1e-6, reduction="mean", gated=True, pred_prob=pred_prob
        # )
        # loss_geometry = 0


        loss = (loss_l1 + loss_bce) + loss_ci
        loss = loss / accum

        # --- Optional adversarial training ---
        adv_active = (engine.d_network is not None) and (
            engine.state.epoch >= engine.adv_start_epoch
        )
        if adv_active:
            # 1) Discriminator update
            #    Real: ground-truth SDF (x_start), Fake: predicted SDF (pred_xstart.detach())
            d_loss = 0.0
            engine.d_network.train()
            if engine.amp:
                with torch.autocast("cuda", **engine.amp_kwargs):
                    d_real = engine.d_network(x_start)
                    d_fake = engine.d_network(pred_xstart.detach())
                    d_loss = (
                        engine.adv_loss(d_real, target_is_real=True)
                        + engine.adv_loss(d_fake, target_is_real=False)
                    ) * 0.5
            else:
                d_real = engine.d_network(x_start)
                d_fake = engine.d_network(pred_xstart.detach())
                d_loss = (
                    engine.adv_loss(d_real, target_is_real=True, for_discriminator=True)
                    + engine.adv_loss(d_fake, target_is_real=False, for_discriminator=True)
                ) * 0.5

            # Accum for D as well
            d_loss = d_loss / accum

            # Backward on D branch
            engine.d_optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
            if engine.amp:
                scaler.scale(d_loss).backward()
            else:
                d_loss.backward()

            # Step D only on accumulation boundary
            if engine.state.iteration % accum == 0:
                if engine.amp:
                    scaler.step(engine.d_optimizer)
                else:
                    engine.d_optimizer.step()

            # 2) Generator adversarial loss (encourage D to think pred is real)
            engine.d_network.eval()
            _set_requires_grad(engine.d_network, False)
            try: 
                if engine.amp:
                    with torch.autocast("cuda", **engine.amp_kwargs):
                        g_adv = engine.adv_loss(
                            engine.d_network(pred_xstart),
                            target_is_real=True,
                            for_discriminator=False,
                        )
                else:
                    g_adv = engine.adv_loss(
                        engine.d_network(pred_xstart), target_is_real=True, for_discriminator=False
                    )
            finally:
                _set_requires_grad(engine.d_network, True)
                engine.d_network.train()

            # Add weighted adversarial term to the generator loss
            loss = loss + engine.adv_lambda * (g_adv / accum)

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
        }
        if adv_active:
            out["d_loss"] = d_loss * accum
            out["g_adv_loss"] = g_adv * engine.adv_lambda
        else:
            out["d_loss"] = torch.tensor(0.0)
            out["g_adv_loss"] = torch.tensor(0.0)

        return out

    trainer = Trainer(
        device=idist.device(),
        max_epochs=config.training.epochs,
        data_loader=train_loader,
        prepare_batch=prepare_batch,
        iteration_update=train_step,
        # key_metric={'loss': None},           # or your dict of Metrics
        additional_metrics=None,
        amp=config.amp.enabled,
    )

    trainer.network = net
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
    trainer.scaler_ = scaler
    trainer.config = config

    trainer.ce = nn.CrossEntropyLoss()
    trainer.mse = nn.MSELoss()
    trainer.bce = nn.functional.binary_cross_entropy_with_logits
    trainer.dice_loss = monai.losses.DiceLoss(
        sigmoid=True, include_background=False, reduction="mean"
    )
    trainer.reverse_dice_loss = monai.losses.DiceLoss(
        sigmoid=False, include_background=False, reduction="mean"
    )
    trainer.ci_dice = SoftCIDice()
    trainer.focal_loss = monai.losses.FocalLoss(
        gamma=2.0, alpha=0.25, reduction="mean"
    )
    trainer.tversky_loss = monai.losses.TverskyLoss(
        alpha=0.3, beta=0.7, reduction="mean", sigmoid=True
    )
    trainer.topo_loss = topo_loss().to(idist.device())

    trainer.optim_set_to_none = config.optimizer.set_to_none
    trainer.klc = monai.transforms.KeepLargestConnectedComponent(applied_labels=[0], is_onehot=True)

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


    return net, optimizer, lr_scheduler, trainer


def get_evaluator(cfg, model, val_loader):
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
            transforms.KeepLargestConnectedComponentd(
                keys=[Keys.PRED],
                is_onehot=False,
                num_components=1 if not cfg.task == 'kidneys' else 2,
            )if cfg.task != "colon" else transforms.Identityd(keys=[Keys.PRED]),
        ]
    )

    metrics = {
        "Mean Dice": MeanDice(
            include_background=True,
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=target_ch,
        )
    }

    # --- One evaluation step: sample conditioned predictions ---
    @torch.no_grad()
    def _eval_step(engine, batchdata):
        # 1) prepare
        image, masks, spacing_tensor = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking
        )
        image = image.float()
        masks = masks.float()
        spacing_tensor = spacing_tensor.float()

        # Save original image and mask
        # torch.save(image, f"/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/4.16/debug_image.pt")
        # torch.save(masks, f"/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/4.16/debug_masks.pt")

        config = engine.config

        engine.network.eval()

        with engine.mode(engine.network):
            model_fn = lambda x: engine.network(
                image=x,
                pred_type="ddim_sample",
                spacing_tensor=spacing_tensor,
                guidance_scale=config.diffusion.guidance_scale,
            )
            if engine.amp:
                with torch.autocast("cuda", **engine.amp_kwargs):
                    engine.state.output[Keys.PRED] = engine.inferer(
                        image,
                        engine.network,
                        pred_type="ddim_sample",
                        spacing_tensor=spacing_tensor,
                        guidance_scale=config.diffusion.guidance_scale,
                    )
            else:
                # engine.state.output[Keys.PRED] = engine.inferer(image, engine.network, pred_type="ddim_sample")
                engine.state.output = {Keys.IMAGE: sdf_to_mask(image), Keys.LABEL: sdf_to_mask(masks)}
                pred = engine.inferer(image, model_fn)
                engine.state.output["SDF"] = pred.clone().detach()
                pred = sdf_to_mask(pred * 10.0)

                engine.state.output[Keys.PRED] = pred


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
    evaluator.inferer = monai.inferers.SlidingWindowInferer(
        roi_size=cfg.data.roi_size, sw_batch_size=1, overlap=0
    )
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
    
    model, optimizer, lr_scheduler, trainer = build_model(cfg, rank, train_loader)

    ema_model = None
    if bool(cfg.ema.enable):
        ema_model = AveragedModel(
            model,
            avg_fn=lambda avg_p, new_p, _: avg_p.mul_(float(cfg.ema.rate)).add_(
                new_p, alpha=1 - float(cfg.ema.rate)
            ),
        )

    evaluator = get_evaluator(
        cfg, ema_model if bool(cfg.ema.enable) else model, val_loader
    )

    savables = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "trainer": trainer,
        "ema_model": ema_model,
        "discriminator": trainer.d_network,
        "d_optimizer": trainer.d_optimizer,
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

    idist.utils.barrier()
    if rank == 0:
        logging.info(
            f"[Rank {rank}] >>> Stage 1 training for {cfg.training.epochs} epochs"
        )

    trainer.run()
    idist.utils.barrier()

    if rank == 0:
        os.makedirs(os.path.join(cfg.training.save_dir, "checkpoints"), exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(cfg.training.save_dir, "checkpoints", "final_unet.pth"),
        )


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

        with idist.Parallel(backend=backend, nproc_per_node=nproc, master_port=2232) as parallel:
            parallel.run(_distributed_run, cfg)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    main()
