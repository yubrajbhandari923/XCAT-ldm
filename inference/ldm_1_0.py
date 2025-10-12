# region imports
from typing import Dict, Any, Tuple, Optional
import functools
import json
import logging
import os
import sys
import glob
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
from aim.pytorch_ignite import AimLogger

import torch
import torch.multiprocessing as tmp_mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel
from ignite.utils import setup_logger


from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import global_step_from_engine
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim
# from ignite.utils import setup_logger

import monai
from monai import transforms
from monai.data import list_data_collate
from monai.handlers import MeanDice, StatsHandler
from monai.handlers.utils import from_engine, stopping_fn_from_metric
from monai.inferers import LatentDiffusionInferer
from monai.losses.dice import DiceLoss, DiceCELoss
from monai.engines.utils import IterationEvents
from monai.networks.nets import AutoencoderKL, Discriminator, PatchDiscriminator
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.engines import SupervisedTrainer
from monai.utils import set_determinism, AdversarialIterationEvents, AdversarialKeys
from monai.utils.enums import CommonKeys as Keys
from monai.losses import PatchAdversarialLoss, HausdorffDTLoss
from monai.engines.utils import DiffusionPrepareBatch
from monai.data import decollate_batch

from utils.monai_helpers import (
    AimIgnite2DImageHandler,
    AimIgniteGIFHandler,
    AimIgnite3DImageHandler
)
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

tmp_mp.set_sharing_strategy("file_system")
torch.serialization.add_safe_globals([monai.utils.enums.CommonKeys])
# stash the original loader
_torch_load = torch.load

# override so all loads are unguarded
torch.load = lambda f, **kwargs: _torch_load(f, weights_only=False, **kwargs)
# endregion


# region Logging and Config Handling :
def log_config(config, rank):
    if rank == 0:
        logging.info(f"Config: \n{OmegaConf.to_yaml(config)}")
        logging.info(f"MONAI version:  \n{monai.__version__}")
        logging.info(f"PyTorch version: \n{torch.__version__}")
        monai.config.print_config()


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
    if config.training.save_config_yaml:
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


# endregion


# region Data Loading and Preprocessing


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


def remove_labels(x: torch.Tensor, labels: list) -> torch.Tensor:
    """Remove the specified labels from the label tensor."""
    for label in labels:
        x[x == label] = 0
    return x


def transform_labels(x: torch.Tensor, label_map: dict) -> torch.Tensor:
    """Transform labels in the tensor according to the provided label_map."""
    for old_label, new_label in label_map.items():
        x[x == old_label] = new_label
    return x


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
    elif "a_grade_colons_not_in_refined_by_md" in pathname:
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


def list_from_jsonl(jsonl_path, image_key="image", label_key="label"):
    """Pure function: read a .jsonl and return a list of dicts for MONAI."""
    files = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line)
            files.append({Keys.IMAGE: d[image_key], Keys.LABEL: d[label_key]})
    return files


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

    custom_transforms = [
        transforms.LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
        transforms.EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
        transforms.Spacingd(
            keys=[Keys.IMAGE, Keys.LABEL],
            pixdim=config.data.pixdim,
            mode=["nearest", "nearest"],
        ),
        transforms.Orientationd(
            keys=[Keys.IMAGE, Keys.LABEL],
            axcodes=config.data.orientation,
        ),
        transforms.CropForegroundd(
            keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.IMAGE
        ),
        transforms.Resized(
            keys=[Keys.IMAGE, Keys.LABEL],
            spatial_size=config.data.roi_size,
            mode=["nearest", "nearest"],
        ),
        transforms.Lambda(add_spacing),  # Add spacing tensor to the sample
        # transforms.SpatialPadd(
        #     keys=[Keys.IMAGE, Keys.LABEL],
        #     spatial_size=config.data.roi_size,
        #     mode=["constant", "constant"],
        #     constant_values=[0, 0],
        # ),
        transforms.Lambdad(
            keys=[Keys.IMAGE, Keys.LABEL],
            func=functools.partial(
                dataset_depended_transform_labels,
                # transform_labels,
                # label_map={
                #     14: 13,
                #     15: 13,
                #     16: 13,
                #     17: 13,
                #     18: 13,
                #     19: 13,
                #     20: 13,
                #     21: 13,
                #     22: 13,
                # },
            ),
        ),
        transforms.Lambdad(
            keys=[Keys.LABEL],
            func=functools.partial(
                remove_labels,
                labels=range(4, 14),  # Assuming labels 4-13 are the organs,
            ),
        ),
    ]

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

    if config.task.target == "colon_bowel":
        # Remove rectum (2) and transform small bowel (3) to rectum (2)
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
    elif config.task.target == "colon":
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
                    transform_labels, label_map={1: 0}
                ),  # Remove colon and small bowel from Image
            ),
        )

    if config.constraint.target == "binary":
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    binary_mask_labels,
                    labels=range(1, 14),  # Assuming labels 1-13 are the organs,
                ),
            ),
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

    custom_transforms.extend(
        [
            transforms.AsDiscreted(
                keys=[Keys.IMAGE, Keys.LABEL],
                to_onehot=[
                    config.model.params.in_channels,
                    config.model.params.out_channels,
                ],
            ),
            transforms.ToTensord(
                keys=[Keys.IMAGE, Keys.LABEL, "spacing_tensor"],
            ),
        ]
    )
    return transforms.Compose(custom_transforms)


# @profile_block("get_dataloaders")
def get_dataloaders(config, aim_logger):
    train_files = list_from_jsonl(
        config.data.train_jsonl, image_key="mask", label_key="mask"
    )
    val_files = list_from_jsonl(
        config.data.val_jsonl, image_key="mask", label_key="mask"
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

    if len(val_files) > config.evaluation.validation_max_num_samples:
        # Randomly sample validation slices to limit the number of samples
        logging.info(f"Validation files length before sampling: {len(val_files)}")
        np.random.seed(config.seed)
        np.random.shuffle(val_files)
        val_files = val_files[: config.evaluation.validation_max_num_samples]
        logging.info(f"Validation files length after sampling: {len(val_files)}")

    # create a training data loader
    if config.data.cache_dir is not None:
        # train_ds = monai.data.PersistentDataset(
        train_ds = monai.data.CacheNTransDataset(
            data=train_files,
            transform=get_transforms(config, train=True),
            cache_dir=config.data.cache_dir,
            # cache_dir=os.path.join(config.data.cache_dir, f"{config.constraint.target}_{config.task.target}"),
            cache_n_trans=9,
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
        val_ds = monai.data.CacheNTransDataset(
            data=val_files,
            transform=get_transforms(config, train=False),
            cache_dir=config.data.cache_dir,
            # cache_dir=os.path.join(config.data.cache_dir, f"{config.constraint.target}_{config.task.target}"),
            cache_n_trans=9,
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


# region Condition encoder (for cross-attn tokens)
# ---------------------------
# Condition encoder (for cross-attn tokens)
# ---------------------------
class ConditionEncoder3D(nn.Module):
    """Encodes conditioning image (surrounding organs one-hot) + spacing maps into N_ctx x D tokens."""

    def __init__(self, in_channels: int, embed_dim: int = 256, num_tokens: int = 8):
        super().__init__()
        ch = [in_channels, 64, 128, 256]
        self.conv = nn.Sequential(
            nn.Conv3d(ch[0], ch[1], 3, padding=1),
            nn.GroupNorm(8, ch[1]),
            nn.SiLU(),
            nn.Conv3d(ch[1], ch[2], 3, stride=2, padding=1),
            nn.GroupNorm(8, ch[2]),
            nn.SiLU(),
            nn.Conv3d(ch[2], ch[3], 3, stride=2, padding=1),
            nn.GroupNorm(8, ch[3]),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(ch[-1], embed_dim * num_tokens)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, x):
        # x: (B, C_cond, 96, 96, 96)
        h = self.conv(x)
        g = self.pool(h).flatten(1)  # (B, ch[-1])
        ctx = self.proj(g).view(x.size(0), self.num_tokens, self.embed_dim)
        return ctx


# endregion


# region Stage 1: Build (AE + Discriminator) and trainer
# ---------------------------
# Stage 1: Build (AE + Discriminator) and trainer
# ---------------------------
def build_stage1(cfg, rank: int):
    """
    Contract with dataloader:
      - batch[Keys.LABEL] : (B, 2, H, W, D) -> target masks (colon, small bowel)
      - batch["spacing_maps"] : (B, 3, H, W, D) -> spacing maps (dx,dy,dz)
    AE input = concat(LABEL, spacing_maps) with 5 channels.
    """
    spatial_dims = 3
    target_ch = int(cfg.stage1.target_channels) + 1
    spacing_ch = int(cfg.data.spacing_channels) if cfg.data.use_spacing_maps else 0
    ae_in_out = target_ch + spacing_ch

    # --- networks ---
    autoencoder = AutoencoderKL(
        spatial_dims=spatial_dims,
        in_channels=ae_in_out,
        out_channels=ae_in_out,
        channels=tuple(cfg.stage1.ae_channels),
        latent_channels=int(cfg.stage1.latent_channels),
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=tuple(cfg.stage1.attn_levels),
    )

    if str(cfg.stage1.disc_type).lower() == "patch":
        disc = PatchDiscriminator(
            spatial_dims=spatial_dims,
            in_channels=ae_in_out,
            out_channels=1,
            num_layers_d=int(cfg.stage1.disc_layers),
            channels=int(cfg.stage1.disc_channels),
            norm="INSTANCE"
        )
    else:
        disc = Discriminator(
            in_shape= (ae_in_out, *cfg.data.roi_size),
            channels= tuple(cfg.stage1.disc_channels),
            strides= tuple(cfg.stage1.disc_strides),
            
        )

    if rank == 0:
        logging.info(
            f"[Rank {rank}] AE params: {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)/1e6:.2f}M"
        )
        logging.info(
            f"[Rank {rank}] D  params: {sum(p.numel() for p in disc.parameters() if p.requires_grad)/1e6:.2f}M"
        )

    # --- optimizers ---
    g_opt = optim.AdamW(
        autoencoder.parameters(),
        lr=float(cfg.stage1.g_lr),
        weight_decay=float(cfg.stage1.weight_decay),
        betas=tuple(cfg.stage1.betas),
    )
    d_opt = optim.AdamW(
        disc.parameters(),
        lr=float(cfg.stage1.d_lr),
        weight_decay=float(cfg.stage1.weight_decay),
        betas=tuple(cfg.stage1.betas),
    )

    # turn into auto-optimizers and auto-models
    g_opt = auto_optim(g_opt)
    d_opt = auto_optim(d_opt)
    autoencoder = auto_model(autoencoder)
    disc = auto_model(disc)

    # --- losses ---
    if str(cfg.stage1.recon_loss).lower() == "mse":
        recon_loss = nn.MSELoss()
    elif str(cfg.stage1.recon_loss).lower() == "dice":
        recon_loss = DiceLoss(sigmoid=True)
    else:
        recon_loss = nn.L1Loss()

    # KL helper (β-VAE style)
    def kld(mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Generator adv loss (least-squares for PatchGAN)

    g_adv = PatchAdversarialLoss(criterion="least_squares")
    d_adv = PatchAdversarialLoss(criterion="least_squares")

    lambda_gan_state = {"v": 0.0}
    

    def g_loss(fake_logits):
        # Generator wants to fool the discriminator, so it wants real logits
        g_adv_loss = lambda_gan_state["v"] * g_adv(fake_logits, target_is_real=True, for_discriminator=False)
        return g_adv_loss

    def d_loss(real_logits, fake_logits):
        d_adv_real = d_adv(real_logits, target_is_real=True, for_discriminator=True)
        d_adv_fake = d_adv(fake_logits, target_is_real=False, for_discriminator=True)
        total_loss = 0.5 * (d_adv_real + d_adv_fake)
        return total_loss

    # prepare_batch: build AE input on-the-fly to avoid relying on external transforms
    def ae_prepare_batch(batch, device=None, non_blocking=True):
        target = batch[Keys.LABEL].to(device, non_blocking=non_blocking).float()  # (B, 2, ...)
        if cfg.data.use_spacing_maps:
            sp = batch["spacing_maps"].to(device, non_blocking=non_blocking).float()  # (B, 3, ...)
            x = torch.cat([target, sp], dim=1)
        else:
            x = target
        return x, x  # inputs, recon target

    # Loss wrapper combining recon + KL for AE; AdversarialTrainer will add GAN term
    def recon_and_kld_loss(g_outputs, targets):
        # AutoencoderKL forward returns: recon, z_mu, z_log_var
        if isinstance(g_outputs, (tuple, list)):
            recon, z_mu, z_log_var = g_outputs
        else:
            recon = g_outputs
            # If AE variant doesn't return KL terms, set zeros
            z_mu = torch.zeros_like(recon[:, :1])
            z_log_var = torch.zeros_like(recon[:, :1])
        # Only compute recon on first target_ch channels (ignore spacing maps)
        return (
            float(cfg.stage1.lambda_recon) * recon_loss(recon[:, :target_ch], targets[:, :target_ch])
            + float(cfg.stage1.lambda_kl) * kld(z_mu, z_log_var)
        )

    # Build AdversarialTrainer
    from monai.engines import AdversarialTrainer

    ae_trainer = AdversarialTrainer(
        device=idist.device(),
        max_epochs=int(cfg.stage1.epochs),
        train_data_loader=None,  # set later
        g_network=autoencoder,
        g_optimizer=g_opt,
        g_loss_function=g_loss,
        recon_loss_function=recon_and_kld_loss,
        d_network=disc,
        d_optimizer=d_opt,
        d_loss_function=d_loss,  # Discriminator term
        prepare_batch=ae_prepare_batch,
        amp=bool(cfg.amp.enabled),
    )

    def set_fake_images(engine):
        g_out = engine.state.output[AdversarialKeys.FAKES]
        recon = g_out if not isinstance(g_out, (tuple, list)) else g_out[0]
        engine.state.output[AdversarialKeys.FAKES] = recon[:, :target_ch]

    def unset_fake_images(engine):
        engine.state.output[AdversarialKeys.FAKES] = engine.state.output[Keys.PRED]

    ae_trainer.add_event_handler(
        AdversarialIterationEvents.GENERATOR_FORWARD_COMPLETED,
        set_fake_images
    )
    ae_trainer.add_event_handler(
        AdversarialIterationEvents.GENERATOR_DISCRIMINATOR_FORWARD_COMPLETED,
        unset_fake_images
    )
    ae_trainer.add_event_handler(
        AdversarialIterationEvents.DISCRIMINATOR_REALS_FORWARD_COMPLETED,
        set_fake_images,
    )
    ae_trainer.add_event_handler(
        AdversarialIterationEvents.DISCRIMINATOR_FAKES_FORWARD_COMPLETED,
        unset_fake_images
    )
    
    @ae_trainer.on(Events.EPOCH_STARTED)
    def ramp_gan_weight(engine):
        e = engine.state.epoch
        if cfg.stage1.gan_warmup_epochs <= 1:
            w = cfg.stage1.lambda_gan
        elif e <= cfg.stage1.gan_warmup_epochs:
            # epoch 1 -> 0, epoch warmup_epochs -> target
            w = cfg.stage1.lambda_gan * (e - 1) / (cfg.stage1.gan_warmup_epochs - 1)
        else:
            w = cfg.stage1.lambda_gan

        lambda_gan_state["v"] = float(w)
        if rank == 0:
            logging.info(f"[GAN warmup] epoch {e}: lambda_gan={w:.4g}")

    return autoencoder, disc, ae_trainer, g_opt, d_opt


def build_stage1_only_ae(cfg, rank: int, train_loader=None):
    """
    AE-only Stage 1 (no adversarial training)

    Contract with dataloader:
      - batch[Keys.LABEL] : (B, 2, H, W, D) -> target masks (colon, small bowel)
      - batch["spacing_maps"] : (B, 3, H, W, D) -> spacing maps (dx,dy,dz)
    AE input = concat(LABEL, spacing_maps) with (2 + 3) = 5 channels when spacing maps are used.
    """
    # --- shapes/channels ---
    spatial_dims = 3
    target_ch = int(cfg.stage1.target_channels) + 1
    spacing_ch = int(cfg.data.spacing_channels) if cfg.data.use_spacing_maps else 0
    ae_in_out = target_ch + spacing_ch

    # --- network ---
    autoencoder = AutoencoderKL(
        spatial_dims=spatial_dims,
        in_channels=ae_in_out,
        out_channels=ae_in_out,
        channels=tuple(cfg.stage1.ae_channels),
        latent_channels=int(cfg.stage1.latent_channels),
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=tuple(cfg.stage1.attn_levels),
    )

    if rank == 0:
        logging.info(
            f"[Rank {rank}] AE params: {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)/1e6:.2f}M"
        )
        logging.info(f"[Rank {rank}] D  params: 0.00M (AE-only)")

    # --- optimizer (G only) ---
    g_opt = optim.AdamW(
        autoencoder.parameters(),
        lr=float(cfg.stage1.g_lr),
        weight_decay=float(cfg.stage1.weight_decay),
        betas=tuple(cfg.stage1.betas),
    )

    # turn into auto-optimizers and auto-models
    g_opt = auto_optim(g_opt)
    autoencoder = auto_model(autoencoder)

    def _parse_loss_specs(spec: str):
        out = []
        for token in (s.strip() for s in spec.split(",") if s.strip()):
            if ":" in token:
                name, w = token.split(":", 1)
                out.append((name.strip().lower(), float(w)))
            else:
                out.append((token.lower(), 1.0))
        return out

    loss_specs = _parse_loss_specs(str(cfg.stage1.recon_loss))

    # --- recon loss ---
    recon_losses = []
    recon_weights = []

    for name, w in loss_specs:
        if name == "mse":
            fn = nn.MSELoss()
        elif name == "l1":
            fn = nn.L1Loss()
        elif name == "dice":
            fn = DiceLoss(
                sigmoid=True,
                squared_pred=True,  # stabilizes gradients
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                include_background=False,
                reduction="mean",
            )
        elif name == "dicece":
            fn = DiceCELoss(
                sigmoid=True,
                include_background=False,
                # softmax=False is default; keep logits + sigmoid
            )
        elif name == "hausdorff":
            fn = HausdorffDTLoss(
                sigmoid=True,
                include_background=False,
                alpha=2.0,  # common stable choice
            )
        else:
            raise ValueError(f"Unknown recon loss: {name}")
        recon_losses.append(fn)
        recon_weights.append(float(w))

    # KL helper (β-VAE style)
    def kld(mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # --- batch prep ---
    def ae_prepare_batch(batch, device=None, non_blocking=True):
        target = batch[Keys.LABEL].to(device, non_blocking=non_blocking).float()  # (B, 2, ...)
        if cfg.data.use_spacing_maps:
            sp = batch["spacing_maps"].to(device, non_blocking=non_blocking).float()  # (B, 3, ...)
            x = torch.cat([target, sp], dim=1)
        else:
            x = target
        return x, x  # inputs, recon target


    # --- combined AE loss (recon + optional KL) ---
    def recon_and_kld_loss(g_outputs, targets):
        # AutoencoderKL forward returns: recon, z_mu, z_log_var
        if isinstance(g_outputs, (tuple, list)):
            recon, z_mu, z_log_var = g_outputs
        else:
            recon = g_outputs
            z_mu = torch.zeros_like(recon[:, :1])
            z_log_var = torch.zeros_like(recon[:, :1])

        # Only compute recon on first target_ch channels (ignore spacing maps)
        pred_logits = recon[:, :target_ch]
        tgt = targets[:, :target_ch]

        # keep VAE numerically tame
        z_log_var = z_log_var.clamp(-30, 20)

        # compute weighted sum of recon losses
        terms = []
        for w, fn in zip(recon_weights, recon_losses):
            val = fn(pred_logits, tgt)  # should be a scalar tensor
            # if a loss returns per-channel or per-batch, reduce it:
            if val.dim() != 0:
                val = val.mean()
            terms.append(w * val)

        recon_term = sum(terms)  # <-- python sum over scalar tensors

        return float(cfg.stage1.lambda_recon) * recon_term + float(
            cfg.stage1.lambda_kl
        ) * kld(z_mu, z_log_var)
        
    # --- trainer (supervised, no GAN) ---

    trainer = SupervisedTrainer(
        device=idist.device(),
        max_epochs=int(cfg.stage1.epochs),
        train_data_loader=train_loader,   # set later
        network=autoencoder,
        optimizer=g_opt,
        loss_function=recon_and_kld_loss,
        prepare_batch=ae_prepare_batch,
        amp=bool(cfg.amp.enabled),
    )

    # Make evaluator-friendly: set Keys.PRED to only the mask channels
    # def _set_pred_mask_channels(engine):
    #     y_pred = engine.state.output[Keys.PRED]
    #     if isinstance(y_pred, (tuple, list)):
    #         recon = y_pred[0]
    #     else:
    #         recon = y_pred
    #     engine.state.output[Keys.PRED] = recon[:, :target_ch]

    # trainer.add_event_handler(IterationEvents.FORWARD_COMPLETED, _set_pred_mask_channels)

    # Optional KL warmup if you have cfg.stage1.kl_warmup_epochs
    if hasattr(cfg.stage1, "kl_warmup_epochs") and int(cfg.stage1.kl_warmup_epochs) > 1:
        final_kl = float(cfg.stage1.lambda_kl)
        warmup = int(cfg.stage1.kl_warmup_epochs)

        @trainer.on(Events.EPOCH_STARTED)
        def _kl_ramp(engine):
            e = engine.state.epoch
            if e <= warmup:
                w = final_kl * (e - 1) / (warmup - 1)
            else:
                w = final_kl
            # monkey-patch by closure capture: update lambda in loss via cfg
            cfg.stage1.lambda_kl = w
            if rank == 0:
                logging.info(f"[KL warmup] epoch {e}: lambda_kl={w:.3e}")

    # Return tuple compatible with your existing calling code
    disc = None
    d_opt = None
    return autoencoder, disc, trainer, g_opt, d_opt


# ---------------------------
# Stage 1: Evaluator
# ---------------------------
def build_stage1_evaluator(cfg, autoencoder, val_loader, device):
    target_ch = int(cfg.stage1.target_channels) + 1

    def ae_inferer(inputs, network):
        out = network(inputs)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        return recon[:, :target_ch]

    post = transforms.Compose(
        [
            transforms.Lambdad(keys=Keys.PRED, func=lambda x: torch.sigmoid(x)),
            transforms.AsDiscreted(keys=Keys.PRED, argmax=True),
            transforms.AsDiscreted(keys=Keys.LABEL, argmax=True),
        ]
    )

    metrics = {
        "Mean Dice": MeanDice(
            include_background=False,
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=target_ch,
        )
    }

    # prepare_batch: build AE input on-the-fly to avoid relying on external transforms
    def ae_prepare_batch(batch, device=None, non_blocking=True):
        target = batch[Keys.LABEL].to(device, non_blocking=non_blocking).float()  # (B, 2, ...)
        if cfg.data.use_spacing_maps:
            sp = batch["spacing_maps"].to(device, non_blocking=non_blocking).float()  # (B, 3, ...)
            x = torch.cat([target, sp], dim=1)
        else:
            x = target
        return x, x  # inputs, recon target
    
    from monai.engines import SupervisedEvaluator

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=autoencoder.eval(),
        prepare_batch=ae_prepare_batch,
        inferer=ae_inferer,
        postprocessing=post,
        key_val_metric=metrics,
        amp=bool(cfg.amp.enabled),
    )
    return evaluator

# endregion


# region Stage 2: Build (UNet + conditioner + scheduler + trainer)
# ---------------------------
# Stage 2: Build (UNet + conditioner + scheduler + trainer)
# ---------------------------
def build_stage2(cfg, rank: int, autoencoder: AutoencoderKL, cond_in_channels: int):
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=int(cfg.stage1.latent_channels),
        out_channels=int(cfg.stage1.latent_channels),
        channels=tuple(cfg.stage2.channels),
        num_res_blocks=int(cfg.stage2.num_res_blocks),
        attention_levels=tuple(cfg.stage2.attn_levels),
        num_head_channels=tuple(cfg.stage2.num_head_channels),
        with_conditioning=True,
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg.stage2.num_train_timesteps),
        schedule=str(cfg.stage2.beta_schedule),
        beta_start=float(cfg.stage2.beta_start),
        beta_end=float(cfg.stage2.beta_end),
    )

    conditioner = ConditionEncoder3D(
        in_channels=cond_in_channels,
        embed_dim=int(cfg.stage2.ctx_dim),
        num_tokens=int(cfg.stage2.num_ctx_tokens),
    )

    inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=float(cfg.stage2.scale_factor))

    opt = optim.AdamW(
        list(unet.parameters()) + list(conditioner.parameters()),
        lr=float(cfg.stage2.lr),
        weight_decay=float(cfg.stage2.weight_decay),
        betas=tuple(cfg.stage2.betas),
    )

    loss_fn = nn.MSELoss()

    # turn into auto-optimizers and auto-models
    opt = auto_optim(opt)
    unet = auto_model(unet)
    conditioner = auto_model(conditioner)

    # Diffusion prepare-batch util: samples noise and time-steps
    # try:

    dpb = DiffusionPrepareBatch(num_train_timesteps=int(cfg.stage2.num_train_timesteps))

    def ldm_prepare_batch(batch, device=None, non_blocking=True):
        target = batch[Keys.LABEL].to(device, non_blocking=non_blocking).float()                      # (B, 2, ...)
        organs = batch[Keys.IMAGE].to(device, non_blocking=non_blocking).float()                      # (B, C_cond, ...)
        sp_maps = batch["spacing_maps"].to(device, non_blocking=non_blocking).float()                # (B, 3, ...)
        cond_img = torch.cat([organs, sp_maps], dim=1)

        inputs, target_eps, kw = dpb((target, None))  # inputs=target image; target_eps=noise; kw={timesteps, noise}
        ctx = conditioner(cond_img)

        # classifier-free guidance: randomly drop condition
        drop = torch.rand((ctx.shape[0],), device=ctx.device) < float(cfg.stage2.p_uncond)
        ctx = torch.where(drop.view(-1, 1, 1), torch.zeros_like(ctx), ctx)

        kw.update(dict(conditioning=ctx, mode="crossattn"))
        return inputs.to(device), target_eps.to(device), kw

    from monai.engines import SupervisedTrainer

    trainer = SupervisedTrainer(
        device=idist.device(),
        max_epochs=int(cfg.stage2.epochs),
        train_data_loader=None,
        network=unet,
        optimizer=opt,
        loss_function=loss_fn,
        prepare_batch=ldm_prepare_batch,
        inferer=inferer,
        amp=bool(cfg.amp.enabled),
    )

    # Freeze AE for stage2 and stash references for evaluation/sampling
    for p in autoencoder.parameters():
        p.requires_grad = False
    trainer.autoencoder = autoencoder.eval()
    trainer.conditioner = conditioner
    trainer.scheduler = scheduler

    return unet, conditioner, scheduler, inferer, trainer


# ---------------------------
# Helper: latent shape probe & decoding
# ---------------------------
def _get_latent_shape(ae: AutoencoderKL, target_masks: torch.Tensor, spacing_maps: torch.Tensor, latent_channels: int) -> Tuple[int, int, int, int, int]:
    with torch.no_grad():
        x = torch.cat([target_masks, spacing_maps], dim=1)
        out = ae(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            z_mu = out[1]
            return z_mu.shape
    # fallback heuristic: /8
    B, _, H, W, D = target_masks.shape
    f = 8
    return (B, latent_channels, H // f, W // f, D // f)


def _cfg_sample_latents(unet, conditioner, scheduler, cond_img, latent_shape, guidance_scale: float, steps: int, device: torch.device):
    ctx = conditioner(cond_img)
    ctx0 = torch.zeros_like(ctx)

    scheduler.set_timesteps(steps)
    z = torch.randn(latent_shape, device=device)

    for t in scheduler.timesteps:
        eps_c = unet(z, t, conditioning=ctx, mode="crossattn")
        eps_u = unet(z, t, conditioning=ctx0, mode="crossattn")
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        z = scheduler.step(model_output=eps, timestep=t, sample=z).prev_sample
    return z


def _ae_decode_masks(autoencoder: AutoencoderKL, latents: torch.Tensor, target_channels: int):
    if hasattr(autoencoder, "decode"):
        recon = autoencoder.decode(latents)
    elif hasattr(autoencoder, "decode_stage_2_inputs"):
        recon = autoencoder.decode_stage_2_inputs(latents)
    else:
        recon = autoencoder(latents, decode=True)  # type: ignore
    return recon[:, :target_channels]


# ---------------------------
# Stage 2: Evaluator
# ---------------------------
def build_stage2_evaluator(cfg, unet, conditioner, autoencoder, scheduler, val_loader, device):
    target_ch = int(cfg.stage1.target_channels)
    guidance_scale = float(cfg.stage2.guidance_scale)
    num_steps = int(cfg.stage2.eval_num_steps)

    post = transforms.Compose([
        transforms.Lambdad(keys=Keys.PRED, func=lambda x: torch.sigmoid(x)),
        transforms.AsDiscreted(keys=Keys.PRED, threshold=0.5),
        transforms.AsDiscreted(keys=Keys.LABEL, threshold=0.5),
    ])

    metrics = {
        "Mean Dice": MeanDice(
            include_background=False,
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=target_ch,
        )
    }

    from monai.engines import Evaluator

    def eval_step(engine, batch):
        organs = batch[Keys.IMAGE].to(device).float()
        target = batch[Keys.LABEL].to(device).float()
        sp = batch["spacing_maps"].to(device).float()
        cond_img = torch.cat([organs, sp], dim=1)

        latent_shape = _get_latent_shape(autoencoder, target, sp, int(cfg.stage1.latent_channels))
        latents = _cfg_sample_latents(
            unet=unet.eval(),
            conditioner=conditioner.eval(),
            scheduler=scheduler,
            cond_img=cond_img,
            latent_shape=latent_shape,
            guidance_scale=guidance_scale,
            steps=num_steps,
            device=device,
        )
        logits = _ae_decode_masks(autoencoder.eval(), latents, target_channels=target_ch)
        engine.state.output = {Keys.IMAGE: organs, Keys.LABEL: target, Keys.PRED: logits}
        return engine.state.output

    evaluator = Evaluator(
        device=device,
        val_data_loader=val_loader,
        iteration_update=eval_step,
        postprocessing=post,
        key_val_metric=metrics,
        amp=bool(cfg.amp.enabled),
    )

    @evaluator.on(Events.STARTED)
    def _set_eval_modes(_):
        autoencoder.eval()
        unet.eval()
        conditioner.eval()

    return evaluator

# endregion

# region Resume helpers
# =====================================================================
# Resume helpers
# =====================================================================

def _is_set(x: Optional[str]) -> bool:
    return isinstance(x, str) and x.strip() != "" and x.strip().lower() != "null"


def _latest_ckpt_path(save_dir: str, stage_name: str, prefix: Optional[str] = None) -> Optional[str]:
    ckpt_dir = os.path.join(save_dir, "checkpoints", stage_name)
    if not os.path.isdir(ckpt_dir):
        return None
    if prefix is None:
        prefix = f"*{stage_name}_latest*"
    candidates = glob.glob(os.path.join(ckpt_dir, f"{prefix}.pt")) or \
                 glob.glob(os.path.join(ckpt_dir, f"{prefix}*"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def _align_module_prefix(sd: Dict[str, Any], target_keys) -> Dict[str, Any]:
    # only touch real model state dicts (string keys with .weight/.bias, etc.)
    if not sd or not isinstance(sd, dict):
        return sd
    str_keys = [k for k in sd.keys() if isinstance(k, str)]
    if not str_keys:
        return sd

    tgt_has_module = any(k.startswith("module.") for k in target_keys)
    sd_has_module  = any(k.startswith("module.") for k in str_keys)

    if tgt_has_module and not sd_has_module:
        return {f"module.{k}": v for k, v in sd.items()}
    if not tgt_has_module and sd_has_module:
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd

def _load_one(obj: Any, state: Dict[str, Any], strict: bool):
    if obj is None or state is None:
        return
    try:
        if hasattr(obj, "load_state_dict"):
            if isinstance(obj, torch.optim.Optimizer):
                obj.load_state_dict(state)
            else:
                # Align 'module.' prefix if needed
                try:
                    tgt_keys = obj.state_dict().keys()
                    state = _align_module_prefix(state, tgt_keys)
                except Exception:
                    pass
                try:
                    obj.load_state_dict(state, strict=strict)
                except TypeError:
                    obj.load_state_dict(state)
        else:
            raise AttributeError
    except Exception as e:
        logging.exception(f"Resume: failed loading into {type(obj).__name__}: {e}")


def resume_from_checkpoint(
    *,
    stage_name: str,
    config,
    to_load: Dict[str, Any],  # mapping name -> live object
    rank: int,
):
    """Load a checkpoint bundle if configured.

    Expected YAML structure under `stageX.resume`:
      path: "auto" | "/path/to/ckpt.pt"
      strict: true/false
      restore_optimizer: true/false
      restore_scheduler: true/false
      restore_ema: true/false
    """
    rcfg = getattr(getattr(config, stage_name), "resume", None)
    if rcfg is None:
        return False

    ckpt_path = rcfg.path if _is_set(getattr(rcfg, "path", None)) else None
    if ckpt_path is None:
        return False
    if ckpt_path.strip().lower() == "auto":
        ckpt_path = _latest_ckpt_path(config.training.save_dir, stage_name)
        if ckpt_path is None:
            if rank == 0:
                logging.warning(f"[{stage_name}] auto-resume requested but no checkpoint found")
            return False

    if rank == 0:
        logging.info(f"[{stage_name}] Resuming from: {ckpt_path}")
    bundle = torch.load(ckpt_path, map_location=idist.device())

    # Always try to load networks and trainer first
    _load_one(to_load.get("trainer"), bundle.get("trainer"), strict=False)
    _load_one(to_load.get("autoencoder"), bundle.get("autoencoder"), strict=getattr(rcfg, "strict", True))
    _load_one(to_load.get("unet"), bundle.get("unet"), strict=getattr(rcfg, "strict", True))
    _load_one(to_load.get("conditioner"), bundle.get("conditioner"), strict=getattr(rcfg, "strict", True))
    _load_one(to_load.get("discriminator"), bundle.get("discriminator"), strict=getattr(rcfg, "strict", True))

    # Optionals per flags
    if getattr(rcfg, "restore_optimizer", True):
        for k in ("g_optimizer", "d_optimizer", "optimizer"):
            _load_one(to_load.get(k), bundle.get(k), strict=False)
    if getattr(rcfg, "restore_scheduler", True):
        _load_one(to_load.get("lr_scheduler"), bundle.get("lr_scheduler"), strict=False)
    if getattr(rcfg, "restore_ema", False):
        _load_one(to_load.get("ema_model"), bundle.get("ema_model"), strict=False)
    # Grad scaler if present (AMP)
    _load_one(to_load.get("scaler"), bundle.get("scaler"), strict=False)

    # Log where we are
    tr = to_load.get("trainer")
    if tr is not None and hasattr(tr, "state"):
        if rank == 0:
            logging.info(f"[{stage_name}] Trainer state after resume: epoch={getattr(tr.state, 'epoch', 'NA')}")
    return True


# endregion


# region Generic handlers
# ---------------------------
# Generic handlers
# ---------------------------

def _maybe(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if v is not None}


def attach_checkpoint_handler(trainer, val_evaluator, objects_to_save: dict, cfg, rank: int, stage_name: str, metric_name: str | None = "Mean Dice", n_saved_best: int = 5, n_saved_latest: int = 5):
    if rank != 0:
        return
    ckpt_dir = os.path.join(cfg.training.save_dir, "checkpoints", stage_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt = ModelCheckpoint(
        dirname=ckpt_dir,
        filename_prefix=f"{cfg.experiment.name}_{stage_name}_latest",
        n_saved=n_saved_latest,
        require_empty=False,
        create_dir=True,
        global_step_transform=global_step_from_engine(trainer),
        save_on_rank=0,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), latest_ckpt, _maybe(objects_to_save))

    if val_evaluator is not None and metric_name:
        best_metric_ckpt = ModelCheckpoint(
            dirname=ckpt_dir,
            filename_prefix=f"{cfg.experiment.name}_{stage_name}_best",
            n_saved=n_saved_best,
            score_name=metric_name,
            score_function=stopping_fn_from_metric(metric_name),
            require_empty=False,
            create_dir=True,
            global_step_transform=global_step_from_engine(trainer),
            save_on_rank=0,
        )
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_metric_ckpt, _maybe(objects_to_save))

    logging.info(f"[Rank {rank}] ({stage_name}) Checkpoint handlers attached")


def attach_ema_update(trainer, ema_model, rank: int, stage_name: str):
    if ema_model is None:
        return
    def update_ema(engine):
        ema_model.update_parameters(engine.network)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1), update_ema)
    if rank == 0:
        logging.info(f"[Rank {rank}] ({stage_name}) EMA update attached")


def attach_validation(trainer, val_evaluator, cfg, rank: int, stage_name: str):
    if val_evaluator is None:
        return
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=int(cfg.evaluation.validation_interval)),
        lambda engine: val_evaluator.run(),
    )
    if rank == 0:
        logging.info(f"[Rank {rank}] ({stage_name}) Validation every {cfg.evaluation.validation_interval} epochs")


def attach_stats_handlers(trainer, val_evaluator, cfg, rank: int, stage_name: str, metric_name: str | None = "Mean Dice"):
    if rank != 0:
        return

    def _loss_from_output(out):
        if isinstance(out, dict):
            if "loss" in out:
                return out["loss"]
            for k, v in out.items():
                if "loss" in str(k).lower():
                    return v
            return None
        return out

    StatsHandler(
        name=f"{stage_name}_training_logger",
        output_transform=_loss_from_output,
        global_epoch_transform=lambda epoch: trainer.state.epoch,
        iteration_log=False,
        tag_name=f"{stage_name}/Loss",
    ).attach(trainer)

    if val_evaluator is not None and metric_name:
        StatsHandler(
            name=f"{stage_name}_training_logger",
            output_transform=lambda x: None,
            global_epoch_transform=lambda epoch: trainer.state.epoch,
            iteration_log=False,
            tag_name=f"{stage_name}/{metric_name}",
        ).attach(val_evaluator)


def attach_early_stopping(val_evaluator, trainer, cfg, rank: int, stage_name: str, metric_name: str = "Mean Dice"):
    if rank != 0:
        return
    if not bool(cfg.evaluation.early_stopping.enabled) or val_evaluator is None:
        logging.info(f"[Rank {rank}] ({stage_name}) Early stopping disabled or no evaluator")
        return
    from ignite.handlers import EarlyStopping
    stopper = EarlyStopping(
        patience=int(cfg.evaluation.early_stopping.patience),
        score_function=stopping_fn_from_metric(metric_name),
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopper)
    logging.info(f"[Rank {rank}] ({stage_name}) Early stopping on '{metric_name}' attached")


def attach_stage1_val_saver(evaluator, cfg):
    so = getattr(getattr(cfg.evaluation, "stage1", None), "save_outputs", None)
    if not (so and bool(so.enabled)):
        return

    outdir = str(so.dir)
    os.makedirs(outdir, exist_ok=True)

    # We’ll save predictions using filenames/affines from the LABEL meta
    saver_pred = transforms.SaveImaged(
        keys="pred",
        # meta_keys="pred_meta_dict",
        output_dir=outdir,
        output_postfix=str(so.pred_postfix),
        separate_folder=False,
        resample=False,
    )
    saver_lab = (
        transforms.SaveImaged(
            keys="lab",
            # meta_keys="lab_meta_dict",
            output_dir=outdir,
            output_postfix=str(so.label_postfix),
            separate_folder=False,
            resample=False,
        )
        if bool(so.save_inputs)
        else None
    )
    
    @evaluator.on(Events.ITERATION_COMPLETED)
    def _save_batch(engine):
        """
        engine.state.output[Keys.PRED] is postprocessed by the evaluator's postprocessing
        (sigmoid->argmax). We borrow LABEL meta from engine.state.batch to name files.
        """
        output = engine.state.output
        batch  = engine.state.batch

        preds = output[0][Keys.PRED]  # postprocessed preds
        labels = output[0][Keys.LABEL]  # postprocessed by postprocessing too

        # Save each item; reuse LABEL meta to build filenames
        for p, l in zip(preds, labels):
            # prediction
            d_pred = {"pred": p, "pred_meta_dict": l.meta}
            saver_pred(d_pred)

            # optional ground-truth copy
            if saver_lab is not None:
                d_lab = {"lab": l, "lab_meta_dict": l.meta}
                saver_lab(d_lab)


def attach_aim_handlers(trainer, val_evaluator, aim_logger, rank: int, cfg, stage_name: str, postprocess=None, metric_name: str | None = "Mean Dice"):
    if rank != 0 or aim_logger is None:
        return
    
    
    aim_logs = {
        "stage1": [
            (trainer, Events.ITERATION_COMPLETED, "Iter Generator Loss", AdversarialKeys.GENERATOR_LOSS),
            (trainer, Events.ITERATION_COMPLETED, "Iter Reconstruction Loss", AdversarialKeys.RECONSTRUCTION_LOSS),
            (trainer, Events.ITERATION_COMPLETED, "Iter Discriminator Loss", AdversarialKeys.DISCRIMINATOR_LOSS),
            (trainer, Events.EPOCH_COMPLETED, "Epoch Generator Loss", AdversarialKeys.GENERATOR_LOSS),
            (trainer, Events.EPOCH_COMPLETED, "Epoch Reconstruction Loss", AdversarialKeys.RECONSTRUCTION_LOSS),
            (trainer, Events.EPOCH_COMPLETED, "Epoch Discriminator Loss", AdversarialKeys.DISCRIMINATOR_LOSS),            
        ] if bool(cfg.stage1.adversarial_train) else [
            (trainer, Events.ITERATION_COMPLETED, "Iter Loss", Keys.LOSS),
            (trainer, Events.EPOCH_COMPLETED, "Epoch Loss", Keys.LOSS),
        ]
    }
    
    for stage_name, log_items in aim_logs.items():
        for eng, event, tag, key in log_items:
            aim_logger.attach_output_handler(
                eng,
                event_name=event,
                tag=f"{stage_name} {tag}",
                output_transform=from_engine([key], first=True),
                global_step_transform=global_step_from_engine(trainer),
            )
    
    # Metric
    if val_evaluator is not None and metric_name:
        aim_logger.attach_output_handler(
            val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=f"{stage_name} Validation {metric_name}",
            metric_names=[metric_name],
            global_step_transform=global_step_from_engine(trainer),
        )

    # Optional image logging
    if val_evaluator is not None and bool(cfg.evaluation.visualize) and AimIgnite3DImageHandler is not None:
        aim_logger.attach(
            val_evaluator,
            log_handler=AimIgnite3DImageHandler(
                f"{stage_name}/Prediction",
                output_transform=from_engine([Keys.IMAGE, Keys.LABEL, Keys.PRED]),
                global_step_transform=global_step_from_engine(trainer),
                postprocess=postprocess,
            ),
            event_name=Events.ITERATION_COMPLETED(
                every=(1 if bool(cfg.experiment.debug) else int(cfg.evaluation.visualize_every_iter))
            ),
        )

    logging.info(f"[Rank {rank}] ({stage_name}) Aim handlers attached")


def attach_handlers(trainer, val_evaluator, objects_to_save: dict, cfg, aim_logger, rank: int, stage_name: str, metric_name: str | None = "Mean Dice", ema_model=None, postprocess=None, step_lr: bool = True):
    if rank == 0:
        logging.info(f"[Rank {rank}] Attaching handlers for {stage_name}")

    attach_checkpoint_handler(trainer, val_evaluator, objects_to_save, cfg, rank, stage_name, metric_name)
    attach_ema_update(trainer, ema_model, rank, stage_name)
    attach_validation(trainer, val_evaluator, cfg, rank, stage_name)
    attach_stats_handlers(trainer, val_evaluator, cfg, rank, stage_name, metric_name)
    if metric_name:
        attach_early_stopping(val_evaluator, trainer, cfg, rank, stage_name, metric_name)
    attach_aim_handlers(trainer, val_evaluator, aim_logger, rank, cfg, stage_name, postprocess, metric_name)

    if step_lr and getattr(cfg, "lr_scheduler", None) and cfg.lr_scheduler.name == "LinearWarmupCosineAnnealingLR":
        def _step_lr(engine):
            if hasattr(engine, "lr_scheduler") and engine.lr_scheduler is not None:
                engine.lr_scheduler.step()
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _step_lr)

    # trainer.add_event_handler(Events.STARTED, lambda e: e.optimizer.zero_grad(set_to_none=True))

# endregion


# TODO: For stage 2, cache latents on-the-fly to avoid AE forward pass every time?
# This could potentially be done by modifying the data loader to include the latent representations
# or by creating a wrapper around the autoencoder that caches the latents.

# ---------------------------
# Distributed run
# ---------------------------
def _distributed_run(rank: int, cfg):
    device = idist.device()
    world_size = idist.get_world_size()

    setup_logger(name="training_logger", level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s: %(message)s", reset=True)
    logging.info(f"[Rank {rank}] Running on device: {device}, world size: {world_size}")

    set_determinism(seed=int(cfg.seed))

    # Optional Aim/profiler
    aim_logger = get_aim_logger(cfg) if (rank == 0 and get_aim_logger is not None and bool(cfg.logging.use_aim)) else None

    # Data
    train_loader, val_loader = get_dataloaders(cfg, aim_logger)
    logging.info(f"[Rank {rank}] Train/Val loaders ready")

    if bool(getattr(getattr(cfg.evaluation, "stage1", None), "validation_only", False)):
        # Build AE (no GAN path to keep it simple)
        ae, disc, ae_trainer, g_opt, d_opt = build_stage1_only_ae(cfg, rank, train_loader)

        # Try to load weights (prefer resume config; fallback to pretrained_path)
        stage1_savables = {"autoencoder": ae}
        resumed = resume_from_checkpoint(
            stage_name="stage1", config=cfg, to_load=stage1_savables, rank=rank
        )
        if (not resumed) and getattr(cfg.stage1, "pretrained_path", None):
            sd = torch.load(cfg.stage1.pretrained_path, map_location=idist.device())
            ae.load_state_dict(sd)
            if rank == 0:
                logging.info("[stage1] Loaded pretrained AE from path")

        ae.eval()
        stage1_val = build_stage1_evaluator(cfg, ae, val_loader, idist.device())
        attach_stage1_val_saver(stage1_val, cfg)

        if rank == 0:
            logging.info("[stage1] >>> Running validation-only")
        stage1_val.run()

        # Skip everything else (no stage 2)
        return

    # Stage 1
    if bool(cfg.pipeline.train_stage1):
        if cfg.stage1.adversarial_train:
            ae, disc, ae_trainer, g_opt, d_opt = build_stage1(cfg, rank)
        else:
            ae, disc, ae_trainer, g_opt, d_opt = build_stage1_only_ae(cfg, rank, train_loader)
        ae_trainer.data_loader = train_loader

        stage1_val = build_stage1_evaluator(cfg, ae, val_loader, idist.device())

        # (optional) EMA for stage1
        ema_model_stage1 = None
        if bool(cfg.ema.stage1.enable):
            ema_model_stage1 = AveragedModel(ae, avg_fn=lambda avg_p, new_p, _: avg_p.mul_(float(cfg.ema.stage1.ema_rate)).add_(new_p, alpha=1 - float(cfg.ema.stage1.ema_rate)))

        stage1_savables = {
            "trainer": ae_trainer,
            "autoencoder": ae,
            "discriminator": disc,
            "g_optimizer": ae_trainer.optimizer if hasattr(ae_trainer, "optimizer") else g_opt,
            "d_optimizer": getattr(ae_trainer, "d_optimizer", None) or d_opt,
            "lr_scheduler": getattr(ae_trainer, "lr_scheduler", None),
            "scaler": getattr(ae_trainer, "scaler_", None),
            "ema_model": ema_model_stage1,
        }

        # Resume from checkpoint if specified
        resume_from_checkpoint(stage_name="stage1", config=cfg, to_load=stage1_savables, rank=rank)

        attach_handlers(
            trainer=ae_trainer,
            val_evaluator=stage1_val,
            objects_to_save=stage1_savables,
            cfg=cfg,
            aim_logger=aim_logger,
            rank=rank,
            stage_name="stage1",
            metric_name="Mean Dice",
            ema_model=ema_model_stage1,
            step_lr=False,  # no LR scheduler wired for AdversarialTrainer by default
        )

        idist.utils.barrier()
        if rank == 0:
            logging.info(f"[Rank {rank}] >>> Stage 1 training for {cfg.stage1.epochs} epochs")
        ae_trainer.run()
        idist.utils.barrier()

        if rank == 0:
            os.makedirs(os.path.join(cfg.training.save_dir, "checkpoints"), exist_ok=True)
            torch.save(ae.state_dict(), os.path.join(cfg.training.save_dir, "checkpoints", "stage1_autoencoder.pth"))

        # Freeze AE for stage2
        for p in ae.parameters():
            p.requires_grad = False
        ae.eval()
    else:
        # Load a pretrained AE if specified
        ae = AutoencoderKL(spatial_dims=3, in_channels=5, out_channels=5, channels=tuple(cfg.stage1.ae_channels), latent_channels=int(cfg.stage1.latent_channels), num_res_blocks=1, norm_num_groups=16, attention_levels=tuple(cfg.stage1.attn_levels))
        if cfg.stage1.pretrained_path:
            sd = torch.load(cfg.stage1.pretrained_path, map_location=device)
            ae.load_state_dict(sd)
        ae.eval()

    # Stage 2
    if bool(cfg.pipeline.train_stage2):
        cond_ch = len(cfg.data.condition_labels) + (int(cfg.data.spacing_channels) if bool(cfg.data.use_spacing_maps) else 0)
        unet, conditioner, scheduler, inferer, dm_trainer = build_stage2(cfg, rank, ae, cond_ch)
        dm_trainer.data_loader = train_loader

        # LR scheduler for Stage 2
        if getattr(cfg, "lr_scheduler", None) and cfg.lr_scheduler.name == "LinearWarmupCosineAnnealingLR":
            dm_trainer.lr_scheduler = LinearWarmupCosineAnnealingLR(dm_trainer.optimizer, warmup_epochs=int(cfg.lr_scheduler.warmup_epochs), max_epochs=int(cfg.lr_scheduler.max_epochs))
            dm_trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda e: e.lr_scheduler.step())

        stage2_val = build_stage2_evaluator(cfg, unet, conditioner, ae, scheduler, val_loader, idist.device())

        ema_model_stage2 = None
        if bool(cfg.ema.stage2.enable):
            ema_model_stage2 = AveragedModel(unet, avg_fn=lambda avg_p, new_p, _: avg_p.mul_(float(cfg.ema.stage2.ema_rate)).add_(new_p, alpha=1 - float(cfg.ema.stage2.ema_rate)))

        attach_handlers(
            trainer=dm_trainer,
            val_evaluator=stage2_val,
            objects_to_save={"unet": unet, "conditioner": conditioner},
            cfg=cfg,
            aim_logger=aim_logger,
            rank=rank,
            stage_name="stage2",
            metric_name="Mean Dice",
            ema_model=ema_model_stage2,
            step_lr=True,
        )

        idist.utils.barrier()
        if rank == 0:
            logging.info(f"[Rank {rank}] >>> Stage 2 training for {cfg.stage2.epochs} epochs")
        dm_trainer.run()

        if rank == 0:
            os.makedirs(os.path.join(cfg.training.save_dir, "checkpoints"), exist_ok=True)
            torch.save(unet.state_dict(), os.path.join(cfg.training.save_dir, "checkpoints", "stage2_unet.pth"))


# ---------------------------
# Hydra entry-point
# ---------------------------


if __name__ == "__main__":

    def derive_experiment_metadata(cfg: DictConfig) -> None:
        parts = [cfg.experiment.name, cfg.constraint.target, cfg.task.target]

        if cfg.loss.type != "default":
            parts.append(cfg.loss.type)

        cfg.experiment.name = "-".join(parts)
        # drop the version itself, leave tags for Aim
        cfg.experiment.tags.extend(parts[2:])

        cfg.training.save_dir = os.path.join(
            cfg.training.save_dir, cfg.experiment.name.lower(), f"{cfg.experiment.version}"
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
        config_path="/home/yb107/cvpr2025/DukeDiffSeg/configs/ldm",
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
        backend = "nccl" if (str(cfg.training.device).startswith("cuda") and torch.cuda.is_available()) else "gloo"

        with idist.Parallel(backend=backend, nproc_per_node=nproc) as parallel:
            parallel.run(_distributed_run, cfg)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    main()
