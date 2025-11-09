from typing import Dict, Any
import os, logging, sys

from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import global_step_from_engine
from ignite.engine import Events
from monai.handlers.utils import from_engine, stopping_fn_from_metric
from monai.handlers import StatsHandler
from monai import transforms
from monai.utils.enums import CommonKeys as Keys
from monai.utils import AdversarialKeys

from omegaconf import OmegaConf

from utils.monai_helpers import (
    AimIgnite3DImageHandler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

def _maybe(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if v is not None}


def attach_checkpoint_handler(
    trainer,
    val_evaluator,
    objects_to_save: dict,
    cfg,
    rank: int,
    stage_name: str,
    metric_name: str | None = "Mean Dice",
    n_saved_best: int = 5,
    n_saved_latest: int = 5,
):
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
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), latest_ckpt, _maybe(objects_to_save)
    )

    if val_evaluator is not None and metric_name and cfg.evaluation.save_checkpoints:
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
        val_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, best_metric_ckpt, _maybe(objects_to_save)
        )

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
        logging.info(
            f"[Rank {rank}] ({stage_name}) Validation every {cfg.evaluation.validation_interval} epochs"
        )


def attach_stats_handlers(
    trainer,
    val_evaluator,
    cfg,
    rank: int,
    stage_name: str,
    metric_name: str | None = "Mean Dice",
):
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


def attach_early_stopping(
    val_evaluator,
    trainer,
    cfg,
    rank: int,
    stage_name: str,
    metric_name: str = "Mean Dice",
):
    if rank != 0:
        return
    if not bool(cfg.evaluation.early_stopping.enabled) or val_evaluator is None:
        logging.info(
            f"[Rank {rank}] ({stage_name}) Early stopping disabled or no evaluator"
        )
        return
    from ignite.handlers import EarlyStopping

    stopper = EarlyStopping(
        patience=int(cfg.evaluation.early_stopping.patience),
        score_function=stopping_fn_from_metric(metric_name),
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopper)
    logging.info(
        f"[Rank {rank}] ({stage_name}) Early stopping on '{metric_name}' attached"
    )


def attach_inference_saver(evaluator, cfg):
    so = cfg.evaluation.save_outputs

    outdir = str(so.dir)
    if getattr(so, "dir_postfix", None):
        outdir = f"{outdir}_{so.dir_postfix}"

    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Output saver will write to: {outdir}")
    
    if getattr(so, "save_config_yaml", False):
        cfg_outpath = os.path.join(outdir, "config.yaml")
        with open(cfg_outpath, "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        logging.info(f"Saved config yaml to: {cfg_outpath}")

    # We’ll save predictions using filenames/affines from the LABEL meta
    pre_save_trans = transforms.Compose(
        [ # orientation
            transforms.EnsureTyped(keys=["pred"], track_meta=True),
            transforms.ToDeviced(keys=["pred"], device="cpu"),
            transforms.Orientationd(keys=["pred"], axcodes=cfg.data.orientation),
        ])
    
    saver_pred = transforms.SaveImaged(
        keys="pred",
        # meta_keys="pred_meta_dict",
        output_dir=outdir,
        output_postfix=str(so.pred_postfix),
        separate_folder=False,
        resample=False,
    )
    saver_img = (
        transforms.SaveImaged(
            keys="img",
            # meta_keys="lab_meta_dict",
            output_dir=outdir,
            output_postfix=str(so.input_postfix),
            separate_folder=False,
            resample=False,
        )
        if bool(so.save_inputs)
        else None
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

        preds = output[0][Keys.PRED]  # postprocessed preds
        # preds = output[0]["SDF"]  # postprocessed preds
        labels = output[0][Keys.LABEL]  # not postprocessed
        image = output[0][Keys.IMAGE]  # not postprocessed

        # logging.info(f"Shapes - preds: {preds.shape}, labels: {labels.shape}, image: {image.shape}")
        # preds = transforms.AsDiscrete(argmax=True)(preds)
        # labels = transforms.AsDiscrete(argmax=True)(labels)
        # image = transforms.AsDiscrete(argmax=True)(image)
        
        # Save each item; reuse LABEL meta to build filenames
        for p, l, img in zip(preds, labels, image):
            # prediction
            d_pred = {"pred": p, "pred_meta_dict": l.meta}
            d_pred = pre_save_trans(d_pred)
            saver_pred(d_pred)

            # optional ground-truth copy
            if saver_lab is not None:
                d_lab = {"lab": l, "lab_meta_dict": l.meta}
                saver_lab(d_lab)
            if saver_img is not None:
                d_img = {"img": img, "img_meta_dict": img.meta}
                saver_img(d_img)
        

    # @evaluator.on(Events.ITERATION_COMPLETED)
    # def _save_batch(engine):
    #     """
    #     engine.state.output[Keys.PRED] is postprocessed by the evaluator's postprocessing
    #     (sigmoid->argmax). We borrow LABEL meta from engine.state.batch to name files.
    #     """
    #     output = engine.state.output
    #     batch = engine.state.batch

    #     preds = output[0][Keys.PRED]  # postprocessed preds
    #     labels = output[0][Keys.LABEL]  # postprocessed by postprocessing too
    #     image = output[0][Keys.IMAGE]  # not postprocessed

    #     # Save each item; reuse LABEL meta to build filenames
    #     for p, l, img in zip(preds, labels, image):
    #         # prediction
    #         d_pred = {"pred": p, "pred_meta_dict": l.meta}
    #         saver_pred(d_pred)

    #         # optional ground-truth copy
    #         if saver_lab is not None:
    #             d_lab = {"lab": l, "lab_meta_dict": l.meta}
    #             saver_lab(d_lab)
    #             d_lab = {"lab": img, "lab_meta_dict": img.meta}
    #             saver_lab(d_lab)


def attach_aim_handlers(
    trainer,
    val_evaluator,
    aim_logger,
    rank: int,
    cfg,
    log_items: list[tuple],
    postprocess=None,
    metric_name: str | None = "Mean Dice",
):
    if rank != 0 or aim_logger is None:
        return

    for eng, event, tag, key in log_items:
        aim_logger.attach_output_handler(
            eng,
            event_name=event,
            tag=f"{tag}",
            output_transform=from_engine([key], first=True),
            global_step_transform=global_step_from_engine(trainer),
        )
        logging.info(f"[Rank {rank}] Aim handler attached: {tag}")

    # Metric
    if val_evaluator is not None and metric_name:
        aim_logger.attach_output_handler(
            val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=f"Validation {metric_name}",
            metric_names=[metric_name],
            global_step_transform=global_step_from_engine(trainer),
        )
        # Log per-batch volume differences
        aim_logger.attach_output_handler(
            val_evaluator,
            event_name=Events.ITERATION_COMPLETED,
            tag="Validation Volume Difference",
            output_transform=lambda x: x[0]["Predicted Volume Difference"].abs().mean().item(),
            global_step_transform=global_step_from_engine(trainer),
        )

    # Optional image logging
    if (
        val_evaluator is not None
        and bool(cfg.evaluation.visualize)
        and AimIgnite3DImageHandler is not None
    ):
        aim_logger.attach(
            val_evaluator,
            log_handler=AimIgnite3DImageHandler(
                f"Validation Prediction",
                output_transform=from_engine([Keys.IMAGE, Keys.LABEL, Keys.PRED]),
                global_step_transform=global_step_from_engine(trainer),
                postprocess=postprocess,
            ),
            event_name=Events.ITERATION_COMPLETED(
                every=(
                    1
                    if bool(cfg.experiment.debug)
                    else int(cfg.evaluation.visualize_every_iter)
                )
            ),
        )

    logging.info(f"[Rank {rank}] Aim handlers attached")


def attach_handlers(
    trainer,
    val_evaluator,
    objects_to_save: dict,
    cfg,
    aim_logger,
    rank: int,
    stage_name: str,
    aim_log_items: list[tuple],
    metric_name: str | None = "Mean Dice",
    ema_model=None,
    postprocess=None,
    step_lr: bool = True,
):
    if rank == 0:
        logging.info(f"[Rank {rank}] Attaching handlers for {stage_name}")

    attach_checkpoint_handler(
        trainer, val_evaluator, objects_to_save, cfg, rank, stage_name, metric_name
    )
    attach_ema_update(trainer, ema_model, rank, stage_name)
    attach_validation(trainer, val_evaluator, cfg, rank, stage_name)
    attach_stats_handlers(trainer, val_evaluator, cfg, rank, stage_name, metric_name)
    if metric_name:
        attach_early_stopping(
            val_evaluator, trainer, cfg, rank, stage_name, metric_name
        )
    attach_aim_handlers(
        trainer,
        val_evaluator,
        aim_logger,
        rank,
        cfg,
        aim_log_items,
        postprocess,
        metric_name,
    )
    if cfg.evaluation.save_outputs.enabled:
        attach_inference_saver(val_evaluator, cfg)

    if (
        step_lr
        and getattr(cfg, "lr_scheduler", None)
        and cfg.lr_scheduler.name == "LinearWarmupCosineAnnealingLR"
    ):

        def _step_lr(engine):
            if hasattr(engine, "lr_scheduler") and engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        trainer.add_event_handler(Events.EPOCH_COMPLETED, _step_lr)

    # trainer.add_event_handler(Events.STARTED, lambda e: e.optimizer.zero_grad(set_to_none=True))
