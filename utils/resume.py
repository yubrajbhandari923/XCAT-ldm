from typing import Dict, Any, Tuple, Optional
import os, torch, glob, sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
import ignite.distributed as idist


def _is_set(x: Optional[str]) -> bool:
    return isinstance(x, str) and x.strip() != "" and x.strip().lower() != "null"


def _latest_ckpt_path(
    save_dir: str, stage_name: str, prefix: Optional[str] = None
) -> Optional[str]:
    ckpt_dir = os.path.join(save_dir, "checkpoints", stage_name)
    if not os.path.isdir(ckpt_dir):
        return None
    if prefix is None:
        prefix = f"*{stage_name}_latest*"
    candidates = glob.glob(os.path.join(ckpt_dir, f"{prefix}.pt")) or glob.glob(
        os.path.join(ckpt_dir, f"{prefix}*")
    )
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
    sd_has_module = any(k.startswith("module.") for k in str_keys)

    if tgt_has_module and not sd_has_module:
        return {f"module.{k}": v for k, v in sd.items()}
    if not tgt_has_module and sd_has_module:
        return {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in sd.items()
        }
    return sd


def _load_one(obj: Any, state: Dict[str, Any], strict: bool):
    """Load state into obj if possible, with some heuristics."""
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
                logging.warning(
                    f"[{stage_name}] auto-resume requested but no checkpoint found"
                )
            return False

    if rank == 0:
        logging.info(f"[{stage_name}] Resuming from: {ckpt_path}")
    bundle = torch.load(ckpt_path, map_location=idist.device())

    # Always try to load networks and trainer first
    _load_one(to_load.get("trainer"), bundle.get("trainer"), strict=False)
    _load_one(
        to_load.get("autoencoder"),
        bundle.get("autoencoder"),
        strict=getattr(rcfg, "strict", True),
    )
    _load_one(
        to_load.get("unet"), bundle.get("unet"), strict=getattr(rcfg, "strict", True)
    )
    _load_one(
        to_load.get("model"), bundle.get("model"), strict=getattr(rcfg, "strict", True)
    )
    _load_one(
        to_load.get("conditioner"),
        bundle.get("conditioner"),
        strict=getattr(rcfg, "strict", True),
    )
    
    if getattr(rcfg, "restore_discriminator", False):
        _load_one(
            to_load.get("discriminator"),
            bundle.get("discriminator"),
            strict=getattr(rcfg, "strict", True),
        )
        _load_one(
            to_load.get("d_optimizer"), bundle.get("d_optimizer"), strict=False
        )

    # Optionals per flags
    if getattr(rcfg, "restore_optimizer", True):
        for k in ("g_optimizer", "optimizer"):
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
            logging.info(
                f"[{stage_name}] Trainer state after resume: epoch={getattr(tr.state, 'epoch', 'NA')}"
            )
    return True
