# region imports
import functools
import json
import logging
import os
import sys
import time

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
from utils import log_config,  _prepare_batch_factory
from utils.data import (
    add_spacing,
    binary_mask_labels,
    remove_labels,
    transform_labels,
    list_from_jsonl,
    dataset_depended_transform_labels,
)
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
    spacing = batch["spacing_tensor"].squeeze()

    # Convert the small bowel label to convex hull mask
    # small_bowel_label = 2  # Assuming small bowel is labeled as 2
    # small_bowel_label_mask = labels[:, small_bowel_label, ...]
    # small_bowel_convex_hull_mask = convex_hull_mask_3d(small_bowel_label_mask, spacing=spacing.clone().detach().cpu().numpy())
    # Replace the small bowel label with the convex hull mask
    # labels[:, small_bowel_label, ...] = small_bowel_convex_hull_mask

    spacing = spacing.to(device, non_blocking=non_blocking)
    # logging.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels, spacing


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

    discriminator = monai.networks.nets.PatchDiscriminator(
        in_channels=config.model.params.out_channels,
        out_channels=config.model.params.out_channels,
        spatial_dims=3,
        num_layers_d=3,
        channels=32,
    )

    optimizer = optim.AdamW(
        net.parameters(),
        config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    d_optimizer = optim.AdamW(
        discriminator.parameters(),
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
    discriminator = auto_model(discriminator)
    d_optimizer = auto_optim(d_optimizer)

    def train_step(engine, batchdata):
        accum = config.training.accumulate_grad_steps

        images, labels, spacing_tensor = engine.prepare_batch(
            batchdata, engine.state.device, engine.non_blocking
        )
        images = images.float()
        labels_1hot = labels.float()
        structure_mask = (images.sum(dim=1, keepdim=True) > 0).float()

        engine.network.train()
        if engine.state.iteration == 1:
            engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
            engine.d_optimizer.zero_grad(set_to_none=True)

        x_start = (labels_1hot) * 2 - 1
        # 2) in-place ops for speed
        # x_start = labels_1hot.mul_(2).sub_(1)  # in-place

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
        # pred_noise, pred_xstart = engine.network(
        #     x=x_t, step=t, image=images, pred_type="noise", spacing_tensor=spacing_tensor
        # )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)

        with torch.no_grad():
            fake_colon =torch.sigmoid(pred_xstart[:, 1, ...])  # Assuming colon is channel 1
            fake_colon = fake_colon.unsqueeze(1)
            real_colon = labels_1hot[:, 1, ...].unsqueeze(1)

        engine.discriminator.train()
        d_real = engine.discriminator(real_colon)
        d_fake = engine.discriminator(fake_colon)

        d_loss = 0.5 * (
            engine.adv_loss(d_real, target_is_real=True, for_discriminator=True)
            + engine.adv_loss(d_fake, target_is_real=False, for_discriminator=True)
        )
        scaler.scale(d_loss).backward() if engine.amp else d_loss.backward()

        num_pos = labels_1hot.sum()
        num_neg = labels_1hot.numel() - num_pos
        beta = (num_neg + 1) / (num_pos + 1)

        # labels_1hot = labels
        # pred_xstart = (pred_xstart + 1) / 2 # Convert to [0, 1] range
        loss_dice = engine.dice_loss(pred_xstart, labels_1hot)
        loss_bce = engine.bce(pred_xstart, labels_1hot, pos_weight=beta)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = engine.mse(pred_xstart, labels_1hot)
        # get channel 1 for colon
        colon_pred = pred_xstart[:, 1, ...]  # Assuming colon is channel 1
        colon_pred = colon_pred.unsqueeze(1)  # Add channel dimension back
        loss_reverse_dice = 1 - engine.reverse_dice_loss(colon_pred, structure_mask)
        # loss_overlap = (pred_xstart[:, 1, ...] * structure_mask).sum()

        d_fake_for_g = engine.discriminator(colon_pred)
        g_adv_loss = engine.adv_loss(
            d_fake_for_g, target_is_real=True, for_discriminator=False
        )

        # loss_cc = single_component_penalty(torch.sigmoid(pred_xstart[:, 1, ...]), tau=0.5)

        # pred_xstart = torch.sigmoid(pred_xstart)

        # pred_noise = torch.sigmoid(pred_noise)
        # loss_mse = engine.mse(pred_noise, noise)

        loss = (loss_dice + loss_bce + loss_mse + loss_reverse_dice + g_adv_loss) / accum
        # loss = (loss_mse + loss_reverse_dice) / accum
        # loss = loss_mse / accum
        # loss = (loss_dice + loss_bce) / accum

        engine.fire_event(IterationEvents.LOSS_COMPLETED)

        # backward
        scaler.scale(loss).backward() if engine.amp else loss.backward()

        # optimizer step on the last micro-step
        if engine.state.iteration % accum == 0:
            if engine.amp:
                scaler.step(optimizer)
                scaler.step(engine.d_optimizer)
                scaler.update()
            else:
                optimizer.step()
                engine.d_optimizer.step()

            optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
            d_optimizer.zero_grad(set_to_none=True)

        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return {
            "loss": loss * accum,
            "loss_dice": loss_dice,
            "loss_bce": loss_bce,
            "loss_mse": loss_mse,
            "loss_reverse_dice": loss_reverse_dice,
            # "loss_overlap": loss_overlap,
            "d_loss" : d_loss,
            "g_adv_loss": g_adv_loss,
            # "label": labels,
        }

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
    trainer.discriminator = discriminator
    trainer.optimizer = optimizer
    trainer.d_optimizer = d_optimizer
    trainer.lr_scheduler = lr_scheduler
    trainer.scaler_ = scaler
    trainer.config = config

    trainer.ce = nn.CrossEntropyLoss()
    trainer.mse = nn.MSELoss()
    trainer.bce = nn.functional.binary_cross_entropy_with_logits
    trainer.dice_loss = monai.losses.DiceLoss(sigmoid=True, include_background=False, reduction="mean")
    trainer.reverse_dice_loss = monai.losses.DiceLoss(
        sigmoid=False, include_background=False, reduction="mean"
    )
    trainer.adv_loss = monai.losses.PatchAdversarialLoss(criterion="least_squares")

    trainer.optim_set_to_none = config.optimizer.set_to_none

    return net, optimizer, lr_scheduler, trainer

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

        engine.state.output = {Keys.IMAGE: image, Keys.LABEL: masks}

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
                pred = engine.inferer(image, model_fn)
                engine.state.output[Keys.PRED] = pred

        engine.state.output[Keys.PRED] = (engine.state.output[Keys.PRED] + 1) / 2.0
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

    evaluator = get_evaluator(cfg, ema_model if bool(cfg.ema.enable) else model, val_loader)

    savables = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "trainer": trainer,
        "ema_model": ema_model,
    }
    resumed = resume_from_checkpoint(
        stage_name="training", config=cfg, to_load=savables, rank=rank
    )

    adverserial_savables = {
        "discriminator": trainer.discriminator,
        "d_optimizer": trainer.d_optimizer,
        "model": model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "trainer": trainer,
        "ema_model": ema_model,
    }

    attach_handlers(
        trainer=trainer,
        val_evaluator=evaluator,
        objects_to_save=adverserial_savables,
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
                    "loss_dice",
                    "loss_bce",
                    "loss_mse",
                    "loss_reverse_dice",
                    "d_loss",
                    "g_adv_loss",
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
