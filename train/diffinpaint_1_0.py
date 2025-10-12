# region imports
import functools
import json
import logging
import os
import sys


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
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel
from ignite.utils import setup_logger
from torch.nn.parallel import DistributedDataParallel, DataParallel


from ignite.engine import Events
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim
# from ignite.utils import setup_logger
from aim.pytorch_ignite import AimLogger


import monai
from monai import transforms
from monai.data import list_data_collate
from monai.handlers import MeanDice, StatsHandler, from_engine
from monai.inferers import LatentDiffusionInferer, DiffusionInferer
from monai.engines.utils import IterationEvents
from monai.networks.nets import DiffusionModelUNet, BasicUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.engines import SupervisedTrainer, SupervisedEvaluator, Evaluator
from monai.utils import set_determinism, AdversarialIterationEvents, AdversarialKeys
from monai.utils.enums import CommonKeys as Keys
from monai.losses import PatchAdversarialLoss, HausdorffDTLoss
from monai.engines.utils import DiffusionPrepareBatch
from monai.data import decollate_batch

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

# from model.diffInpaint import BasicUNetEncoder, DiffusionModelUNet

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils import log_config,  _prepare_batch_factory
from utils.data import add_spacing, binary_mask_labels, remove_labels, transform_labels, list_from_jsonl, dataset_depended_transform_labels
from utils.resume import resume_from_checkpoint
from utils.handlers import attach_handlers, attach_inference_saver

tmp_mp.set_sharing_strategy("file_system")
torch.serialization.add_safe_globals([monai.utils.enums.CommonKeys])
# stash the original loader
_torch_load = torch.load

# override so all loads are unguarded
# torch.load = lambda f, **kwargs: _torch_load(f, weights_only=False, **kwargs)
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

    if config.task == "colon_bowel":
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
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.IMAGE],
                func=functools.partial(
                    transform_labels,
                    label_map={2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10}, 
                ),
            ),
        )
    elif config.task == "colon":
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
                # to_onehot=[
                #     config.model.params.in_channels,
                #     config.model.params.out_channels,
                # ],
                to_onehot=[
                    2, 2,
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
        logging.info("Debug mode is on, using a small subset of the data.")
        logging.info(f"Training files length: {len(train_files)}")
        logging.info(f"Validation files length: {len(val_files)}")

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
            batch_size=config.data.val_batch_size * config.training.num_gpus,
            num_workers=config.data.val_num_workers * config.training.num_gpus,
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

# ----------------------------
# wrapper: denoiser UNet with cross-attn conditioning
# ----------------------------
# class ConditionedDenoiser(nn.Module):
#     """
#     Wrap DiffusionModelUNet and a condition encoder.
#     Cross-attention is used inside the UNet; we pass encoded tokens via `conditioning=...`.
#     """

#     def __init__(
#         self,
#         target_channels: int,  # colon + small bowel = 2
#         cond_channels: int,  # surrounding organ channels
#         unet_channels=(256, 256, 512, 512),
#         attn_levels=(False, True, True, True),
#         num_res_blocks=2,
#         num_head_channels=8,  # per MONAI API: channels per attention head
#         crossattn_dim=256,  # must match CondEncoder3D embed_dim
#     ):
#         super().__init__()
#         # self.cond_encoder = CondEncoder3D(cond_channels, crossattn_dim)
#         self.cond_encoder = BasicUNetEncoder(
#             spatial_dims=3,
#             in_channels=cond_channels,
#             out_channels=crossattn_dim,
#             channels=tuple(unet_channels),
#             num_res_blocks=num_res_blocks,
#         )

#         # Denoising UNet with spatial transformers (cross-attention) enabled.
#         # NOTE: API names below follow MONAI >=1.4. If your version differs,
#         # you may need to adapt (e.g., `with_conditioning=True`, `context_dim=...`).
#         self.unet = DiffusionModelUNet(
#             spatial_dims=3,
#             in_channels=target_channels,  # predict noise for 2-channel targets
#             out_channels=target_channels,
#             num_res_blocks=num_res_blocks,
#             channels=tuple(unet_channels),
#             attention_levels=tuple(attn_levels),
#             num_head_channels=num_head_channels,
#             with_conditioning=True,  # enable cross-attention path
#             cross_attention_dim=crossattn_dim,  # condition token channel-dim
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         timesteps: torch.Tensor,
#         context: torch.Tensor = None,
#         mode: str = "crossattn",
#     ):
#         """
#         x:        noisy sample (B, C=target_channels, H, W, D) (the inferer adds noise)
#         timesteps: integer timesteps (B,)
#         context: surrounding organ masks (B, C_cond, H, W, D)
#         mode:     "crossattn" (default) or "concat" (fallback)
#         """
#         if context is not None and mode == "crossattn":
#             ctx = self.cond_encoder(context)  # (B, N_tokens, crossattn_dim)
#             # MONAI DiffusionModelUNet accepts `conditioning=` for cross-attn context
#             return self.unet(x, timesteps=timesteps, context=ctx)

#         if context is not None and mode == "concat":
#             x_in = torch.cat(
#                 [x, context], dim=1
#             )  # requires re-init UNet in/out if you use this path
#             return self.unet(x_in, timesteps=timesteps)

#         return self.unet(x, timesteps=timesteps)
# # endregion


# region Models and Eval
def _prepare_batch(batch, device=None, non_blocking=False, verbose=False, num_train_timesteps=1000, predict_noise=True):
    x_clean = batch[Keys.LABEL].to(device=device, non_blocking=non_blocking)
    cond = batch[Keys.IMAGE].to(device=device, non_blocking=non_blocking)

    x_clean = x_clean * 2.0 - 1.0  # scale to [-1, 1]
    cond = cond * 2.0 - 1.0  # scale to [-1, 1]
    
    if verbose:
        logging.info(
            f"x_clean (labels) shape: {x_clean.shape}, dtype: {x_clean.dtype}, min: {x_clean.min()}, max: {x_clean.max()}"
        )
        logging.info(
            f"cond (images) shape: {cond.shape}, dtype: {cond.dtype}, min: {cond.min()}, max: {cond.max()}"
        )

    # Remove the background channel from cond
    # cond = cond[:, 1:, ...].to(device=device, non_blocking=non_blocking).to(torch.float32)

    b = x_clean.shape[0]
    timesteps = torch.randint(
        low=0,
        high=num_train_timesteps,
        size=(b,),
        device=x_clean.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(x_clean)

    # labels = noise (epsilon prediction)
    # kwargs consumed by DiffusionInferer: it will build x_noisy internally via scheduler.add_noise(...)
    kwargs = {
        "noise": noise,
        "timesteps": timesteps,
        "condition": cond,
        "mode": "concat",
    }
    if predict_noise:
        # model predicts noise (epsilon)
        return x_clean, noise, [], kwargs
    else:
        # model predicts x0 (clean image)
        return x_clean, [x_clean, cond], [], kwargs


def build_model(cfg, train_loader):
    # Build the diffusion model
    # model = ConditionedDenoiser(
    #     target_channels=3,
    #     cond_channels=11,
    #     unet_channels=cfg.model.params.channels,
    #     attn_levels=cfg.model.params.attn_levels,
    #     num_res_blocks=cfg.model.params.num_res_blocks,
    #     num_head_channels=cfg.model.params.num_head_channels,
    #     crossattn_dim=cfg.model.params.crossattn_dim,
    # )

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=cfg.model.params.in_channels,  # Colon mask + Forbidden region mask
        out_channels=cfg.model.params.out_channels, # predict noise for colon mask
        num_res_blocks=cfg.model.params.num_res_blocks,
        channels=tuple(cfg.model.params.channels),
        attention_levels=tuple(cfg.model.params.attn_levels),
        num_head_channels=cfg.model.params.num_head_channels,
        with_conditioning=False,  # enable cross-attention path
        cross_attention_dim=None,  # condition token channel-dim
    )

    # Build the diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule="linear_beta",
        prediction_type=cfg.model.diffusion.prediction_type,  # "epsilon" or "sample"
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2)

    model = auto_model(model)
    optimizer = auto_optim(optimizer)

    inferer = DiffusionInferer(scheduler=scheduler)
    # loss_fn = nn.functional.mse_loss  # or monai.losses.DiffusionLoss() if you want v-pred, etc.
    # loss_fn = monai.losses.DiceLoss(
    #     include_background=False,  # class 0 is background
    #     to_onehot_y=False,  # y already one-hot (B,3,H,W,D)
    #     softmax=True,  # apply softmax to model output before Dice
    #     squared_pred=True,  # common stabilization
    # )
    # loss_fn = monai.losses.DiceCELoss(
    #     include_background=False,  # class 0 is background
    #     to_onehot_y=False,  # y already one-hot (B,3,H,W
    #     softmax=True,  # apply softmax to model output before Dice
    #     squared_pred=True,  # common stabilization
    # )
    dice_loss = monai.losses.DiceLoss(
        include_background=False,  # class 0 is background
        to_onehot_y=False,  # y already one-hot (B,3,H,W,D)
        sigmoid=True,  # apply sigmoid to model output before Dice
        squared_pred=True,  # common stabilization
    )
    def _loss_fn(y_pred, targets):
        y, cond = targets
        y = (y + 1.0) / 2.0  # scale back to [0, 1]
        cond = (cond + 1.0) / 2.0  # scale back to [0, 1]
        loss_bce = nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        loss_dice = dice_loss(y_pred, y)
        
        y_pred = torch.sigmoid(y_pred)
        loss_mse = nn.functional.mse_loss(y_pred, y)
        
        loss_reverse_dice = 1.0 - dice_loss(y, cond)

        return loss_bce + loss_dice + loss_mse + loss_reverse_dice

    # prepare_batch = _prepare_batch_factory(num_train_timesteps=1000, condition_key=Keys.IMAGE, verbose=False)
    prepare_batch = functools.partial(
        _prepare_batch,
        num_train_timesteps=1000,
        predict_noise=False,
        verbose=False,
    )

    trainer = SupervisedTrainer(
        device=idist.device(),
        max_epochs=cfg.training.epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=_loss_fn,
        inferer=inferer,
        prepare_batch=prepare_batch,
        amp=False,
    )

    return model, optimizer, scheduler, trainer


# --- tiny helper: CFG-capable inferer (re-uses MONAI's scheduler math) ---
class DiffusionInfererCFG(DiffusionInferer):
    @torch.no_grad()
    def sample_cfg(
        self,
        input_noise: torch.Tensor,  # (B, C, H, W, D)
        diffusion_model,  # ConditionedUNetWrapper
        scheduler=None,
        *,
        conditioning_tokens: torch.Tensor | None,  # (B, N_ctx, Cctx)
        cfg_scale: float = 0.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling with classifier-free guidance, using model.forward(x,t) twice (cond/uncond)."""
        sch = scheduler or self.scheduler
        device = input_noise.device
        x = input_noise

        # prepare steps
        sch.set_timesteps(len(sch.timesteps))

        # run denoising steps
        for t in sch.timesteps:
            t_in = torch.full((x.size(0),), int(t), device=device, dtype=torch.long)

            # conditional pass
            # diffusion_model.ctx = conditioning_tokens
            eps_c = diffusion_model(x, t_in, conditioning_tokens)

            # unconditional pass
            if cfg_scale != 0.0:
                eps_u = diffusion_model(x, t_in, torch.zeros_like(conditioning_tokens))
                eps = eps_u + cfg_scale * (eps_c - eps_u)
            else:
                eps = eps_c

            # one DDIM step
            out = sch.step(model_output=eps, timestep=int(t), sample=x, eta=eta)
            x = out.prev_sample if hasattr(out, "prev_sample") else out[0]
            
        # diffusion_model.ctx = None  # clean up
        return x


def get_evaluator(cfg, model, val_loader):
    target_ch = cfg.model.params.out_channels 
    
    # Build the evaluator
    post = transforms.Compose(
        [
            # transforms.Lambdad(keys=Keys.PRED, func=torch.softmax),
            # transforms.AsDiscreted(keys=Keys.PRED, threshold=0.5),
            # transforms.AsDiscreted(keys=Keys.LABEL, threshold=0.5),
            transforms.AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=target_ch),
            transforms.AsDiscreted(keys=Keys.LABEL, argmax=True, to_onehot=target_ch),
        ]
    )

    metrics = {
        "Mean Dice": MeanDice(
            include_background=False,
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            num_classes=target_ch,
        )
    }
    
    ddim_scheduler = DDIMScheduler(prediction_type=cfg.model.diffusion.prediction_type)
    ddim_scheduler.set_timesteps(100)

    inferer = DiffusionInfererCFG(scheduler=ddim_scheduler)

    cfg_scale = cfg.model.diffusion.cfg_scale  # CFG scale; 0.0 = no guidance, 1.0 = normal, >1.0 = stronger guidance
    ddim_eta = cfg.model.diffusion.ddim_eta  # DDIM eta; 0.0 = no noise, 1.0 = full noise

    # --- One evaluation step: sample conditioned predictions ---
    @torch.no_grad()
    def _eval_step(engine, batch):
        device = idist.device()
        
        model.eval()

        cond_img = batch[Keys.IMAGE].to(device).float()  # (B, C_cond, H, W, D)
        target = batch[Keys.LABEL].to(device).float()  # (B, 3,      H, W, D)
        B, _, H, W, D = target.shape

        # start from pure noise in target space
        noise = torch.randn((B, target_ch, H, W, D), device=device)

        # sample with CFG
        # preds = inferer.sample_cfg(
        preds = inferer.sample(
            input_noise=noise,
            diffusion_model=model,  # ConditionedUNetWrapper
            scheduler=ddim_scheduler,
            conditioning=cond_img,
            mode="concat",
            # cfg_scale=cfg_scale,
            # eta=ddim_eta,
        )

        engine.state.output = {
            Keys.IMAGE: cond_img,
            Keys.LABEL: target,
            Keys.PRED: preds,
        }
        
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
    )
    return evaluator
# endregion

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

    model, optimizer, scheduler, trainer = build_model(cfg, train_loader)
    evaluator = get_evaluator(cfg, model, val_loader)

    ema_model = None
    if bool(cfg.ema.enable):
        ema_model = AveragedModel(model, avg_fn=lambda avg_p, new_p, _: avg_p.mul_(float(cfg.ema.rate)).add_(new_p, alpha=1 - float(cfg.ema.rate)))
    
    savables = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer": trainer,
        "ema_model": ema_model
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
                (trainer, Events.EPOCH_COMPLETED, "Epoch Loss", Keys.LOSS),
            ],
            metric_name="Mean Dice",
            ema_model=ema_model,
            step_lr=False,  # no LR scheduler wired for AdversarialTrainer by default
    )

    if bool(cfg.training.inference_mode):        
        if rank == 0:
            logging.info("Running inference only")

        evaluator.run()
        return
        
    idist.utils.barrier()
    if rank == 0:
        logging.info(f"[Rank {rank}] >>> Stage 1 training for {cfg.training.epochs} epochs")

    trainer.run()
    idist.utils.barrier()

    if rank == 0:
        os.makedirs(os.path.join(cfg.training.save_dir, "checkpoints"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg.training.save_dir, "checkpoints", "final_unet.pth"))


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
        config_path="/home/yb107/cvpr2025/DukeDiffSeg/configs/diffinpaint",
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
