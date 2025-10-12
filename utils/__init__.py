import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion
import logging
from aim.pytorch_ignite import AimLogger
from omegaconf import OmegaConf
from monai.utils.enums import CommonKeys as Keys
import monai
import torch
import os

logging.basicConfig(level=logging.INFO)


def convex_hull_mask_3d(
    mask,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    method="centers",
    qhull_options=None,
):
    """
    Return a boolean mask equal to the convex hull of a 3D binary mask.

    Parameters
    ----------
    mask : (Z, Y, X) bool/0-1 ndarray
    spacing : (sx, sy, sz) voxel size in world units
    origin : (ox, oy, oz) world coords of grid origin at index (0,0,0) *corner*
    method : "corners" (exact hull of union of cubes) or "centers" (approx hull)
    qhull_options : str or None, e.g. "QJ" to joggle if precision issues

    Returns
    -------
    hull_mask : (Z, Y, X) bool ndarray
    """
    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # sx, sy, sz = map(float, spacing)
    # sx, sy, sz = spacing[0,0], spacing[1,1], spacing[2,2]  # spacing is a tensor of shape (3,3)
    sx, sy, sz = (1,1,1)  # spacing is a tensor of shape (3,3)
    ox, oy, oz = map(float, origin)

    # --- choose points to hull ---
    if method == "corners":
        # keep only surface voxels before expanding to corners to reduce points
        shell = mask & ~binary_erosion(
            mask, structure=np.ones((3, 3, 3), bool), border_value=0
        )
        zyx = np.argwhere(shell)
        if zyx.size == 0:
            zyx = np.argwhere(mask)  # fall back if everything eroded away

        # 8 cube-corner offsets in (z,y,x) index space
        offs = (
            np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"))
            .reshape(3, -1)
            .T
        )  # (8,3)
        # unique corners -> convert to (x,y,z) index space
        pts_idx_xyz = np.unique(
            (zyx[:, None, :] + offs[None, :, :]).reshape(-1, 3), axis=0
        )[:, ::-1]
    elif method == "centers":
        # voxel centers = indices + 0.5
        zyx = np.argwhere(mask)
        pts_idx_xyz = zyx[:, ::-1] + 0.5
    else:
        raise ValueError("method must be 'corners' or 'centers'")

    # indices -> world coords
    pts_world = np.empty_like(pts_idx_xyz, dtype=float)
    pts_world[:, 0] = pts_idx_xyz[:, 0] * sx + ox
    pts_world[:, 1] = pts_idx_xyz[:, 1] * sy + oy
    pts_world[:, 2] = pts_idx_xyz[:, 2] * sz + oz

    # convex hull in world space
    hull = ConvexHull(pts_world, qhull_options=qhull_options)
    A = hull.equations[:, :3]  # outward normals
    b = hull.equations[:, 3]  # offsets; inside satisfies A @ p + b <= 0

    # --- voxelize the hull back onto the original grid (centers test) ---
    Z, Y, X = mask.shape

    # bbox in index space to limit computation
    mins = np.floor(pts_idx_xyz.min(axis=0)).astype(int)
    maxs = np.ceil(pts_idx_xyz.max(axis=0)).astype(int)

    x0, x1 = np.clip([mins[0], maxs[0]], 0, X)
    y0, y1 = np.clip([mins[1], maxs[1]], 0, Y)
    z0, z1 = np.clip([mins[2], maxs[2]], 0, Z)

    # voxel centers in that bbox (index space -> world)
    zs = np.arange(z0, z1)
    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    Zg, Yg, Xg = np.meshgrid(zs + 0.5, ys + 0.5, xs + 0.5, indexing="ij")  # (z,y,x)

    Xw = Xg * sx + ox
    Yw = Yg * sy + oy
    Zw = Zg * sz + oz
    P = np.stack([Xw, Yw, Zw], axis=-1).reshape(-1, 3)  # (N,3), xyz

    inside = np.all(P @ A.T + b <= 1e-9, axis=1)  # tolerance
    hull_mask = np.zeros_like(mask, dtype=bool)
    inside = inside.reshape(Zg.shape)
    hull_mask[z0:z1, y0:y1, x0:x1] = inside
    return hull_mask


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


def _prepare_batch_factory(num_train_timesteps: int, condition_key: str, verbose: bool= False, predict_noise: bool= True):
    """
    Returns a prepare_batch callable that:
      - pulls the clean target masks as images (B, 2, ...)
      - samples timesteps and Gaussian noise
      - returns (images, noise) as (x, y)
      - passes kwargs to inferer: noise, timesteps, condition, mode='crossattn'
    Contract with dataloader:
      batch[Keys.LABEL]   : (B, 2, H, W, D)  # colon + small bowel masks (clean target)
      batch[condition_key]: (B, Cc, H, W, D) # surrounding organs condition
    """

    def _prepare_batch(batch, device=None, non_blocking=False):
        x_clean = (
            batch[Keys.LABEL]
            .to(device=device, non_blocking=non_blocking).to(torch.float32)
        )
        cond = (
            batch[condition_key]
            .to(non_blocking=non_blocking)
            .to(device=device)
            .to(torch.float32)
        )
        
        if verbose:
            logging.info(f"x_clean (labels) shape: {x_clean.shape}, dtype: {x_clean.dtype}, min: {x_clean.min()}, max: {x_clean.max()}")
            logging.info(f"cond (images) shape: {cond.shape}, dtype: {cond.dtype}, min: {cond.min()}, max: {cond.max()}")

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
            "mode": "crossattn",
        }
        if predict_noise:
            # model predicts noise (epsilon)
            return x_clean, noise, [], kwargs
        else:
            # model predicts x0 (clean image)
            return x_clean, x_clean, [], kwargs

    return _prepare_batch
