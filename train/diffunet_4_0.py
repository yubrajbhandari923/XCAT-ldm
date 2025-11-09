# region imports
import functools
import json
import logging
import os
import sys
import time
import math


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
    MaskToSDFd
)
from utils.monai_transforms import CropForegroundAxisd, SmoothColonMaskd
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


def get_transforms(config, train=True):
    """Get the MONAI transforms for training or validation."""

    def custom_name_formatter(meta_dict, saver):
        full_path = meta_dict["filename_or_obj"]
        base = os.path.basename(full_path)
        # If the filename itself contains "colon", pull the parent folder as the ID

        if "labels" in full_path.lower():
            postfix = "_label"
        else:
            postfix = "_image"

        return {"filename": f"{base.replace('.nii.gz', '')}"}

    data_keys = [Keys.IMAGE, Keys.LABEL]

    if config.data.body_filled_channel:
        data_keys.append("body_filled_mask")

    custom_transforms = [
        transforms.LoadImaged(keys=data_keys),
        transforms.EnsureChannelFirstd(keys=data_keys),
        transforms.Spacingd(
            keys=data_keys,
            pixdim=config.data.pixdim,
            mode="nearest",
        ),
        transforms.Orientationd(
            keys=data_keys,
            axcodes=config.data.orientation,
        ),
        transforms.KeepLargestConnectedComponentd(
            keys=[Keys.LABEL, Keys.IMAGE],
        ),
        transforms.Lambdad(
            keys=[Keys.IMAGE, Keys.LABEL],
            func=functools.partial(
                dataset_depended_transform_labels,
            ),
        ),
    ]

    if config.data.body_filled_channel:
        custom_transforms.append(
          CropForegroundAxisd(
                keys=data_keys, source_key=Keys.IMAGE, axis=config.data.slice_axis, margin=5
          )
        )

    # if train:
    #     custom_transforms.append(
    #         transforms.RandCropByPosNegLabeld(
    #             keys=[Keys.IMAGE, Keys.LABEL],
    #             label_key=Keys.LABEL,
    #             spatial_size=(96, 96, 96),
    #             pos=5,
    #             neg=1,
    #             num_samples=4 if not config.experiment.debug else 1,
    #             image_key=Keys.IMAGE,
    #             image_threshold=0,
    #         )
    #     )

    if config.task == "colon_bowel":
        # Remove rectum (2) and transform small bowel (3) to rectum (2)
        custom_transforms.append(transforms.Lambdad(
            keys=[Keys.LABEL],
            func=functools.partial(
                remove_labels,
                labels=range(4, 14),  # Assuming labels 4-13 are the organs,
            ),
        ))
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=functools.partial(
                    remove_labels,
                    labels=[2],
                ),
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=functools.partial(transform_labels, label_map={3: 2}),
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels, label_map={1: 0, 3: 0}
                ),  # Remove colon and small bowel from Image
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels,
                    label_map={
                        2: 1,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 5,
                        8: 6,
                        9: 7,
                        10: 8,
                        11: 9,
                        12: 10,
                    },
                ),
            ),
        )
    elif config.task == "colon":

        custom_transforms.append( transforms.Lambdad(
            keys=[Keys.LABEL],
            func=functools.partial(
                remove_labels,
                labels=range(4, 14),  # Assuming labels 4-13 are the organs,
            ),
        ))

        # Remove rectum (2) and small bowel (3)
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=functools.partial(remove_labels, labels=[2, 3]),
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels, label_map={1: 0, 3: 0}
                ),  # Remove colon and small bowel from Image
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels,
                    label_map={
                        2: 1,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 5,
                        8: 6,
                        9: 7,
                        10: 8,
                        11: 9,
                        12: 10,
                    },
                ),
            ),
        )
    elif config.task == "liver":
        # Remove rectum (2) and small bowel (3)
        # custom_transforms.append(        # transforms.Lambdad(
        #     keys=[Keys.LABEL],
        #     func=functools.partial(
        #         remove_labels,
        #         labels=range(6, 14),  # Assuming labels 4-13 are the organs,
        #     ),
        # )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=functools.partial(remove_labels, labels=[1,2,3,4,6,7,8,9,10,11,12,13]),
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels, label_map={5: 0, 3: 0}
                ),  # Remove colon and small bowel from Image
            ),
        )
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels,
                    label_map={
                        6: 5,
                        7: 6,
                        8: 7,
                        9: 8,
                        10: 9,
                        11: 10,
                        12: 11,
                    },
                ),
            ),
        )

    if config.constraint == "binary":
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    binary_mask_labels,
                    labels=range(1, 14),  # Assuming labels 1-13 are the organs,
                ),
            ),
        )

    custom_transforms.extend(
        [
            # transforms.AsDiscreted(
            #     keys=[Keys.IMAGE, Keys.LABEL],
            #     to_onehot=[
            #         config.model.params.in_channels,
            #         config.model.params.out_channels,
            #     ],
            # ),
            transforms.CropForegroundd(
                keys=data_keys, source_key="body_filled_mask" if config.data.body_filled_channel else Keys.IMAGE, margin=5
            ),
            transforms.Resized(
                keys=data_keys,
                spatial_size=config.data.roi_size,
                mode="nearest",
            ),
            transforms.Lambda(add_spacing),  # Add spacing tensor to the sample
            MaskToSDFd(
                keys=data_keys,
                spacing_key="spacing_tensor",
                device=torch.device("cpu"),
            ),
            transforms.ToTensord(
                keys=data_keys + ["spacing_tensor"],
                device=torch.device("cpu"),
            ),
        ]
    )

    if config.data.save_data:
        custom_transforms.append(
            transforms.SaveImaged(
                keys=[Keys.IMAGE],
                meta_keys=[f"{Keys.IMAGE}_meta_dict"],
                output_dir=os.path.join(
                    config.training.save_dir,
                    "training_samples" if train else "validation_samples",
                    "images",
                ),
                output_postfix="image",
                separate_folder=False,
                # output_name_formatter=custom_name_formatter,
            )
        )
        custom_transforms.append(
            transforms.SaveImaged(
                keys=[Keys.LABEL],
                meta_keys=[f"{Keys.LABEL}_meta_dict"],
                output_dir=os.path.join(
                    config.training.save_dir,
                    "training_samples" if train else "validation_samples",
                    "labels",
                ),
                output_postfix="label",
                separate_folder=False,
                # output_name_formatter=custom_name_formatter,
            )
        )
        if config.data.body_filled_channel:
            custom_transforms.append(
                transforms.SaveImaged(
                    keys=["body_filled_mask"],
                    meta_keys=["body_filled_mask_meta_dict"],
                    output_dir=os.path.join(
                        config.training.save_dir,
                        "training_samples" if train else "validation_samples",
                        "body_filled_masks",
                    ),
                    output_postfix="body_filled_mask",
                    separate_folder=False,
                    # output_name_formatter=custom_name_formatter,
                )
            )

    return transforms.Compose(custom_transforms)


# @profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger):
    train_files = list_from_jsonl(
        config.data.train_jsonl, image_key="mask", label_key="mask", include_body_filled=config.data.body_filled_channel, body_filled_key="body_filled_mask"
    )
    val_files = list_from_jsonl(
        config.data.val_jsonl, image_key="mask", label_key="mask", include_body_filled=config.data.body_filled_channel, body_filled_key="body_filled_mask"
    )

    if aim_logger is not None:
        aim_logger.experiment.track(
            aim.Text(f"{json.dumps(train_files, indent=2)}"),
            name="Training Files",
            step=1,
        )
        logging.info(f"Training files length: {len(train_files)}")
        aim_logger.experiment.track(
            aim.Text(f"{json.dumps(val_files, indent=2)}"),
            name="Validation Files",
            step=1,
        )
        logging.info(f"Validation files length: {len(val_files)}")

    if config.experiment.debug:
        train_files = train_files[:12]
        val_files = val_files[:4]
        logging.info("Debug mode is on, using a small subset of the data.")
        logging.info(f"Training files length: {len(train_files)}")
        logging.info(f"Validation files length: {len(val_files)}")

    if len(val_files) > config.evaluation.validation_max_num_samples:
        # Randomly sample validation slices to limit the number of samples
        logging.info(f"Validation files length before sampling: {len(val_files)}")
        np.random.seed(config.seed if config.seed is not None else 42)
        np.random.shuffle(val_files)
        val_files = val_files[: config.evaluation.validation_max_num_samples]
        logging.info(f"Validation files length after sampling: {len(val_files)}")

    # create a training data loader
    if config.data.cache_dir is not None:
        # train_ds = monai.data.PersistentDataset(
        trans = get_transforms(config, train=True)
        train_ds = monai.data.CacheNTransDataset(
            data=train_files,
            transform=trans,
            cache_dir=config.data.cache_dir,
            # cache_dir=os.path.join(config.data.cache_dir, f"{config.constraint.target}_{config.task.target}"),
            cache_n_trans=len(trans) if config.data.cache_n_transforms == 'all' else int(config.data.cache_n_transforms),
        )
        train_loader = auto_dataloader(
            train_ds,
            batch_size=config.data.batch_size_per_gpu * config.training.num_gpus,
            num_workers=config.data.num_workers_per_gpu * config.training.num_gpus,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            shuffle=config.data.shuffle_train_data,
        )
        # create a validation data loader
        trans = get_transforms(config, train=False)
        val_ds = monai.data.CacheNTransDataset(
            data=val_files,
            transform=trans,
            cache_dir=config.data.cache_dir,
            # cache_dir=os.path.join(config.data.cache_dir, f"{config.constraint.target}_{config.task.target}"),
            cache_n_trans=len(trans) if config.data.cache_n_transforms == 'all' else int(config.data.cache_n_transforms),
        )
        val_loader = auto_dataloader(
            val_ds,
            batch_size=config.data.val_batch_size,
            num_workers=config.data.val_num_workers,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
    else:
        train_ds = monai.data.Dataset(
            data=train_files, transform=get_transforms(config, train=True)
        )
        train_loader = monai.data.DataLoader(
            train_ds,
            batch_size=config.data.batch_size_per_gpu * config.training.num_gpus,
            num_workers=config.data.num_workers_per_gpu * config.training.num_gpus,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            shuffle=config.data.shuffle_train_data,
        )
        # create a validation data loader
        val_ds = monai.data.Dataset(
            data=val_files, transform=get_transforms(config, train=False)
        )
        val_loader = monai.data.DataLoader(
            val_ds,
            batch_size=config.data.val_batch_size,
            num_workers=config.data.val_num_workers,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
    return train_loader, val_loader


# endregion


# region Models and Eval
# p: (B,1,H,W,D) probabilities. tau: threshold for CCs.
def single_component_penalty(p, tau=0.5):
    losses = []
    B = p.shape[0]
    for b in range(B):
        with torch.no_grad():
            binb = (p[b, 0] > tau).float().detach().cpu().numpy()
            lab = cc3d.connected_components(binb, connectivity=6)
            counts = np.bincount(lab.ravel())
            keep_id = counts[1:].argmax() + 1 if counts.size > 1 else 0
            keep = (lab == keep_id).astype(np.float32)
            keep = (
                torch.from_numpy(keep).to(p.device).unsqueeze(0).unsqueeze(0)
            )  # (1,1,H,W,D)

        # penalize prob mass outside the largest component
        losses.append((p[b : b + 1] * (1 - keep)).mean())

    return torch.stack(losses).mean()


def prepare_batch(batch, device=None, non_blocking=True):
    images = batch[Keys.IMAGE].to(device, non_blocking=non_blocking)
    labels = batch[Keys.LABEL].to(device, non_blocking=non_blocking)
    if "body_filled_mask" in batch:
        body_filled_mask = batch["body_filled_mask"].to(device, non_blocking=non_blocking)
        images = torch.cat([images, body_filled_mask], dim=1)
    spacing = batch["spacing_tensor"].squeeze()
    

    # Convert the small bowel label to convex hull mask
    # small_bowel_label = 2  # Assuming small bowel is labeled as 2
    # small_bowel_label_mask = labels[:, small_bowel_label, ...]
    # small_bowel_convex_hull_mask = convex_hull_mask_3d(small_bowel_label_mask, spacing=spacing.clone().detach().cpu().numpy())
    # Replace the small bowel label with the convex hull mask
    # labels[:, small_bowel_label, ...] = small_bowel_convex_hull_mask

    spacing = spacing.to(device, non_blocking=non_blocking)

    # images = mask_to_sdf(images, spacing=spacing)
    # labels = mask_to_sdf(labels, spacing=spacing)
    # logging.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
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


        # with torch.no_grad():
        #     pred_prob = torch.sigmoid(pred_xstart * 10.0)
        #     pred_mask = (pred_prob >= 0.5).float()
            
        #     # Count the number of connected components in the predicted mask
        #     B = pred_mask.shape[0]
        #     binv = (pred_mask > 0.5).float()
        #     n_components = []
        #     for b in range(B):
        #         binb = binv[b, 0].detach().cpu().numpy()
        #         lab = cc3d.connected_components(binb, connectivity=6)
        #         counts = np.bincount(lab.ravel())
        #         fg_counts = counts[1:] if counts.size > 1 else np.array([], dtype=counts.dtype)  # Ignore background count
        #         n_comp = int((fg_counts >= 10).sum())
                
        #         # if there is only one component, use that as the grouth truth
        #         if n_comp == 1:
        #             # Replace corresponding x_start with pred_xstart
        #             x_start[b] = pred_xstart[b]
            

        loss_l1 = nn.L1Loss()(pred_xstart, x_start)
        # loss_l2 = nn.MSELoss()(pred_xstart, x_start)

        surface_x_start = (x_start > -0.15)
        # loss_surface_l1 = nn.L1Loss()(
        #     pred_xstart * surface_x_start, x_start * surface_x_start
        # )
        loss_surface_l1 = 0
        # surface_area = surface_area_from_sdf_normalized(pred_xstart, spacing_tensor)
        surface_area = 0

        # pred_logits = torch.sigmoid(pred_logits)
        pred_logits = pred_xstart * 10.0
        y_logits = sdf_to_mask(labels_1hot)
        pos_weight = (y_logits == 0).sum() / ((y_logits == 1).sum() + 1e-5)

        # loss_bce = 0
        loss_bce = engine.bce(pred_logits, y_logits, pos_weight=pos_weight)
        # loss_ft = engine.focal_loss(pred_logits, y_logits) + engine.tversky_loss(pred_logits, y_logits)

        loss_ci = engine.ci_dice(y_logits, torch.sigmoid(pred_logits))
        # loss_ci = 0

        pred_prob = torch.sigmoid(pred_logits)
        # volume = volume_from_mask_batched(pred_prob, spacing_tensor)
        volume = 0
        # loss_geometry = loss_isoperimetric(
        #     surface_area, volume, alpha=1e-6, beta=1e-6, reduction="mean", gated=True, pred_prob=pred_prob
        # )
        loss_geometry = 0

        # loss_eki = eikonal_band_match_gt_vec(
        #     pred_xstart, x_start, spacing_tensor, band_width=0.15,
        # )
        loss_eki = 0
        
        # with torch.no_grad():
        #     hard = (pred_prob > 0.5).float()
        #     M_lcc = torch.zeros_like(hard)
        #     # vol_all = hard.sum(dim=(2,3,4), keepdim=True) + 1e-6
        #     for b in range(pred_prob.size(0)):
        #         lab = cc3d.connected_components(hard[b,0].cpu().numpy(), connectivity=6)
        #         counts = np.bincount(lab.ravel())
        #         counts[0] = 0
        #         if counts.sum() == 0:
        #             continue
        #         lcc_id = counts.argmax()
        #         M = torch.from_numpy((lab == lcc_id).astype(np.float32)).to(pred_logits.device)
        #         M_lcc[b,0] = M

        # 1) “Islands” penalty: probability not in the single component
        # L_islands = (pred_prob * (1 - M_lcc)).mean()
        L_islands = 0

        # 2) Mass re-centering: reward pouring mass *into* the LCC
        #    (this gives gradients inside the LCC region)
        # L_fill = (1 - pred_prob) * M_lcc
        # L_fill = L_fill.mean()
        L_fill = 0

        loss = (loss_l1 + loss_bce) + (loss_surface_l1 + loss_ci) + 0.1 * loss_geometry + loss_eki +  (L_fill + L_islands)

        if engine.pca_enabled:
            # Extract colon prediction (prob or logit; prob is fine) and flatten
            colon_pred = pred_prob[:, 0, ...]         # [B, H, W, D]
            # Optional: if your PCA was trained on binary masks, you can threshold softly/hard here:
            # colon_pred = (colon_pred >= 0.5).float()
            B = colon_pred.shape[0]
            # Ensure spatial matches PCA shape; assert once (cheap)
            if colon_pred.shape[1:] != engine.pca_shape:
                raise RuntimeError(f"PCA shape {engine.pca_shape} != pred {tuple(colon_pred.shape[1:])}")
            X = colon_pred.reshape(B, -1)                                 # [B, N]
            # Center by PCA mean
            Xc = X - engine.pca_mu.unsqueeze(0)                           # [B, N]

            assert engine.pca_U.shape[1] == engine.pca_inv_lam.numel(), "PCA U/λ mismatch after slicing"

            # Project to PCA coords: z = Xc * U   (U columns = PCs)
            # (Assumes U has unit-norm columns; from eigendecomp this is standard.)
            Z = Xc @ engine.pca_U                                         # [B, k]
            # Mahalanobis per sample: sum_j z_j^2 / lambda_j
            maha = (Z.pow(2) * engine.pca_inv_lam.unsqueeze(0)).sum(dim=1)  # [B]
            # Average over batch; multiply by weight
            loss_pca = maha.mean()
            loss =  loss + loss_pca

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
            # "loss_l2": loss_l2,
            "loss_ci": loss_ci,
            "loss_pca": loss_pca if engine.pca_enabled else torch.tensor(0.0),
            "loss_surface_l1": loss_surface_l1,
            "surface_area": surface_area.mean() if not isinstance(surface_area, int) else torch.tensor(0.0),
            "loss_geometry": loss_geometry,
            "loss_eki": loss_eki,
            "volume": volume.mean() if not isinstance(volume, int) else torch.tensor(0.0),
            "loss_islands": L_islands,
            "loss_fill": L_fill,
            # "loss_lcc_dice": loss_lcc_dice
            # "loss_topo": loss_topo
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

    trainer.pca_enabled = bool(config.model.pca_likelihood.enabled)
    if trainer.pca_enabled:
        pca_path = getattr(config.model.pca_likelihood, "pca_model_path", None)
        assert pca_path is not None and os.path.isfile(
            pca_path
        ), "Set training.pca_model_path to colon_pca_cov_eig.pt"
        pca_pack = torch.load(pca_path, weights_only=False, map_location=idist.device())
        # Tensors to device
        trainer.pca_mu = pca_pack["mean_flat"].to(idist.device()).float()  # [N]
        trainer.pca_U = (
            pca_pack["U"].to(idist.device()).float()
        )  # [N, k] (columns ~ PCs)
        trainer.pca_lam = pca_pack["eigenvalues"].to(idist.device()).float()  # [k]
        trainer.pca_shape = tuple(pca_pack["volume_shape"])  # e.g. (96,96,96)

        k_cfg = getattr(
            config.model.pca_likelihood, "k", 50
        )  # optional: allow override
        k = min(int(k_cfg), trainer.pca_U.shape[1])
        if k <= 0:
            raise ValueError("Requested PCA components k must be >= 1")
        trainer.pca_U = trainer.pca_U[:, :k].contiguous()  # [N, k]
        trainer.pca_lam = trainer.pca_lam[:k].contiguous()  # [k]

        # Precompute inverse lambdas for Mahalanobis
        eps = 1e-8
        trainer.pca_inv_lam = 1.0 / (trainer.pca_lam + eps)

        trainer.pca_k = k
        if idist.get_rank() == 0:
            logging.info(f"[Rank 0] PCA PCs kept: {k}")
            logging.info("[Rank 0] Loaded PCA model for shape likelihood...")

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
            )
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

                # with torch.no_grad():
                #     # pred_prob = torch.sigmoid(pred * 10.0)
                #     pred_mask = (pred >= 0.5).float()

                #     # Count the number of connected components in the predicted mask
                #     B = pred_mask.shape[0]
                #     binv = (pred_mask > 0.5).float()
                #     n_components = []
                #     for b in range(B):
                #         binb = binv[b, 0].detach().cpu().numpy()
                #         lab = cc3d.connected_components(binb, connectivity=6)
                #         counts = np.bincount(lab.ravel())
                #         fg_counts = counts[1:] if counts.size > 1 else np.array([], dtype=counts.dtype)  # Ignore background count
                #         n_comp = int((fg_counts >= 10).sum())

                #         # if there is only one component, use that as the grouth truth
                #         if n_comp == 1:
                #             # Replace corresponding x_start with pred_xstart
                #             engine.state.output[Keys.LABEL][b] = pred_mask[b]

                engine.state.output[Keys.PRED] = pred

                # img = engine.state.output[Keys.IMAGE]
                # lbl = engine.state.output[Keys.LABEL]
                # pred = engine.state.output[Keys.PRED]

                # log stats
                # logging.info(
                #     f"Image shape {img.shape} stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}"
                # )
                # logging.info(
                #     f"Label shape {lbl.shape} stats: min={lbl.min():.4f}, max={lbl.max():.4f}, mean={lbl.mean():.4f}"
                # )
                # logging.info(
                #     f"Pred shape {pred.shape} stats: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}"
                # )

        # engine.state.output[Keys.PRED] = (engine.state.output[Keys.PRED] + 1) / 2.0
        # engine.state.output[Keys.PRED] = torch.sigmoid(engine.state.output[Keys.PRED])

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
    train_loader, val_loader = get_dataloaders(cfg, aim_logger)
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
        stage_name="training", config=cfg, to_load=savables, rank=rank
    )

    attach_handlers(
        trainer=trainer,
        val_evaluator=evaluator,
        objects_to_save=savables,
        cfg=cfg,
        aim_logger=aim_logger,
        rank=rank,
        stage_name="training",
        aim_log_items=[
            (trainer, Events.ITERATION_COMPLETED, "Iter Loss", Keys.LOSS),
            *[
                (trainer, Events.EPOCH_COMPLETED, f"Epoch {key}", key)
                for key in [
                    "loss",
                    "loss_bce",
                    "loss_l1",
                    # "loss_l2",
                    "loss_ci",
                    "loss_eki",
                    # "loss_pca",
                    "loss_surface_l1",
                    "surface_area",
                    "loss_geometry",
                    "volume",
                    "loss_islands",
                    "loss_fill",
                    # "loss_lcc_dice",
                    # "d_loss",
                    # "g_adv_loss",
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

    # for data in iter(train_loader):
    #     # Go through the training loader once to make everything is cached
    #     continue
    # for data in iter(val_loader):
    #     # Go through the validation loader once to make everything is cached
    #     continue
    # if rank == 0:
    #     logging.info("Cache warm-up complete")

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
        parts = [cfg.experiment.name, cfg.constraint, cfg.task]

        cfg.experiment.name = "-".join(parts)
        # drop the version itself, leave tags for Aim
        cfg.experiment.tags.extend(parts[2:])

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
            cfg.training.resume = None  # Don't resume from any previous run

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
        config_path="/home/yb107/cvpr2025/DukeDiffSeg/configs/diffunet_v4",
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
