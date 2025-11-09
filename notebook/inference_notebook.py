# ============================================================================
# MINIMAL INFERENCE SETUP FOR JUPYTER NOTEBOOK
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

import monai
from monai import transforms
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler

# Import your custom utilities (adjust paths as needed)
from utils.data import MaskToSDFd, sdf_to_mask
from utils.monai_transforms import (
    HarmonizeLabelsd,
    AddSpacingTensord,
    FilterAndRelabeld,
    EnsureAllTorchd,
    CropForegroundAxisd
)

from monai.transforms import Transform

class ProbeTransform(Transform):
    def __init__(self, message="ProbeTransform called"):
        super().__init__()
        self.message = message

    def __call__(self, data):
        print(self.message)
        return data

# ============================================================================
# 1. CONFIGURATION
# ============================================================================


class InferenceConfig:
    # Model params
    spatial_dims = 3
    in_channels = 2  # image SDF + conditioning
    out_channels = 1  # target organ SDF
    features = [32, 64, 64, 128, 256]  # adjust based on your trained model
    attention_levels = [False, False, False, False, False]
    num_head_channels = [0, 0, 0, 64, 64]
    with_conditioning = True
    cross_attention_dim = 128  # adjust based on your trained model

    # Diffusion params
    diffusion_steps = 1000
    ddim_steps = 20
    beta_schedule = "scaled_linear_beta"
    model_mean_type = "sample"  # or "sample"
    guidance_scale = 1.0  # CFG scale
    condition_drop_prob = 0.1

    # Data params
    pixdim = (1.5, 1.5, 2.0)
    orientation = "RAS"
    roi_size = (128, 128, 128)

    # Paths
    checkpoint_path = "/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-iterative/6.6/checkpoints/liver/DiffUnet-binary-iterative_liver_best_checkpoint_1710_MeanDice0.7490.pt"
    device = "cuda:7" if torch.cuda.is_available() else "cpu"


config = InferenceConfig()

# ============================================================================
# 2. ORGAN MAPPING (from your script)
# ============================================================================

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


def get_conditioning_organs(generation_order, target_organ_index):
    """Get list of organs to condition on"""
    if target_organ_index not in generation_order:
        raise ValueError(f"Target organ {target_organ_index} not in order")
    pos = generation_order.index(target_organ_index)
    return generation_order[:pos]


# ============================================================================
# 3. BUILD PREPROCESSING TRANSFORM
# ============================================================================




def build_inference_transform(config, target_organ="liver", generation_order=None):
    """Simplified transform for single-sample inference"""

    target_organ_index = NAME_TO_INDEX.get(target_organ)
    if generation_order is None:
        generation_order = [5, 6, 7, 9, 3, 1, 2, 4, 10, 11, 12]  # default order

    conditioning_organs = get_conditioning_organs(generation_order, target_organ_index)

    data_keys = ["image", "label", "body_filled_channel"]

    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=data_keys),
            transforms.EnsureChannelFirstd(keys=data_keys),
            transforms.Spacingd(keys=data_keys, pixdim=config.pixdim, mode="nearest"),
            transforms.Orientationd(keys=data_keys, axcodes=config.orientation),
            ProbeTransform(message="🐔 After Orientationd"),
            # transforms.KeepLargestConnectedComponentd(keys=data_keys),
            # ProbeTransform(message="🐸 After KeepLargestConnectedComponentd"),
            HarmonizeLabelsd(keys=["image", "label"], kidneys_same_index=True),
            CropForegroundAxisd(
                keys=data_keys,
                source_key="image",
                axis=2,
                margin=5,
            ),
            transforms.CropForegroundd(keys=data_keys, source_key="body_filled_channel", margin=5),
            ProbeTransform(message="🐢 After CropForegroundd"),
            transforms.Resized(
                keys=data_keys, spatial_size=config.roi_size, mode="nearest"
            ),
            AddSpacingTensord(ref_key="image"),
            FilterAndRelabeld(
                image_key="image",
                label_key="label",
                conditioning_organs=conditioning_organs,
                target_organ=target_organ_index,
            ),
            ProbeTransform(message="🐍 After FilterAndRelabeld"),
            MaskToSDFd(
                keys=data_keys,
                spacing_key="spacing_tensor",
                device=torch.device("cpu"),
            ),
            ProbeTransform(message="🐙 After MaskToSDFd"),
            EnsureAllTorchd(print_changes=False),
            transforms.EnsureTyped(
                keys=data_keys + ["spacing_tensor"],
                track_meta=True,
            ),
        ]
    )

    return transform


# ============================================================================
# 4. BUILD MODEL
# ============================================================================


def build_model(config):
    """Create and load model"""

    model = DiffusionModelUNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels
        + config.out_channels,  # concat image during inference
        out_channels=config.out_channels,
        channels=config.features,
        attention_levels=config.attention_levels,
        num_res_blocks=1,
        transformer_num_layers=0,
        num_head_channels=config.num_head_channels,
        with_conditioning=config.with_conditioning,
        cross_attention_dim=config.cross_attention_dim,
    )

    # Load checkpoint
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(config.device)
    model.eval()

    print(f"✓ Model loaded from {config.checkpoint_path}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    return model


# ============================================================================
# 5. BUILD SCHEDULER
# ============================================================================


def build_scheduler(config):
    """Create DDIM scheduler for inference"""

    scheduler = DDIMScheduler(
        num_train_timesteps=config.diffusion_steps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule=config.beta_schedule,
        clip_sample=False,
        prediction_type=config.model_mean_type,
    )
    scheduler.set_timesteps(num_inference_steps=config.ddim_steps)

    return scheduler


# ============================================================================
# 6. INFERENCE FUNCTION
# ============================================================================


@torch.no_grad()
def run_inference(model, scheduler, image_sdf, body_filled_sdf, config):
    """
    Run DDIM sampling to generate organ mask

    Args:
        model: DiffusionModelUNet
        scheduler: DDIMScheduler
        image_sdf: conditioning image SDF [B, 1, H, W, D]
        config: InferenceConfig

    Returns:
        pred_mask: binary mask [B, 1, H, W, D]
        pred_sdf: signed distance field [B, 1, H, W, D]
    """

    device = config.device
    image_sdf = image_sdf.to(device).float()
    pred = torch.randn_like(image_sdf) * 2.0
    
    # Initialize with random noise
    if body_filled_sdf is not None:
        body_filled_sdf = body_filled_sdf.to(device).float()
        image = torch.cat([image_sdf, body_filled_sdf], dim=1)
    
    # Get all timesteps
    all_next_timesteps = torch.cat(
        [scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype)]
    )

    # DDIM sampling loop
    for i, (t, next_t) in enumerate(zip(scheduler.timesteps, all_next_timesteps)):
        pred_before = pred.clone()
        # Concatenate conditioning
        model_input = torch.cat([pred, image], dim=1)

        # Predict
        t_tensor = torch.full((image.shape[0],), t, device=device).long()
        model_output = model(x=model_input, timesteps=t_tensor)

        # Classifier-free guidance (if scale != 1.0)
        if config.guidance_scale != 1.0:
            image_uncond = torch.zeros_like(image)
            model_input_uncond = torch.cat([pred, image_uncond], dim=1)
            uncond_output = model(x=model_input_uncond, timesteps=t_tensor)
            model_output = uncond_output + config.guidance_scale * (
                model_output - uncond_output
            )

        # DDIM step
        pred, _ = scheduler.step(model_output, t, pred)
        
        change = (pred - pred_before).abs().mean().item()
        print(f"Step {i}: t={t}, change={change:.6f}, pred_mean={pred.mean():.4f}")

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{len(scheduler.timesteps)}")

    # Convert SDF to mask
    pred_sdf = pred.clone()
    pred_mask = sdf_to_mask(pred * 10.0)  # scale factor from your training

    return pred_mask, pred_sdf

# ============================================================================
# 7. MAIN INFERENCE PIPELINE
# ============================================================================


def inference_pipeline(
    image_path,
    label_path,  # ground truth for comparison (optional)
    config,
    target_organ="liver",
    generation_order=None,
):
    """
    Full inference pipeline from file paths

    Returns:
        dict with keys: image, label, pred_mask, pred_sdf, spacing
    """

    # 1. Preprocess
    print("📦 Preprocessing data...")
    transform = build_inference_transform(config, target_organ, generation_order)

    data_dict = {"image": image_path, "label": label_path}
    data_dict = transform(data_dict)

    image_sdf = data_dict["image"].unsqueeze(0)  # [1, 1, H, W, D]
    label_sdf = data_dict["label"].unsqueeze(0)
    spacing = data_dict["spacing_tensor"]

    # 2. Build model & scheduler
    print("🏗️  Building model...")
    model = build_model(config)
    scheduler = build_scheduler(config)

    # 3. Run inference
    print("🎨 Running DDIM sampling...")
    pred_mask, pred_sdf = run_inference(model, scheduler, image_sdf, config)

    # 4. Return results
    return {
        "image_sdf": image_sdf.cpu(),
        "label_sdf": label_sdf.cpu(),
        "label_mask": sdf_to_mask(label_sdf).cpu(),
        "pred_mask": pred_mask.cpu(),
        "pred_sdf": pred_sdf.cpu(),
        "spacing": spacing.cpu(),
    }


# ============================================================================
# 8. EXAMPLE USAGE IN JUPYTER
# ============================================================================

# if __name__ == "__main__":

#     # Configure
#     config = InferenceConfig()
#     config.checkpoint_path = "/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-iterative/6.6/checkpoints/liver/DiffUnet-binary-iterative_liver_best_checkpoint_1710_MeanDice0.7490.pt"
#     config.device = "cuda:1"  # match your script
#     config.ddim_steps = 20
#     config.guidance_scale = 1.0

#     # Run inference
#     results = inference_pipeline(
#         image_path="/path/to/your/image.nii.gz",
#         label_path="/path/to/your/label.nii.gz",
#         config=config,
#         target_organ="liver",
#         generation_order=[5, 6, 7, 9, 3, 1, 2, 4, 10, 11, 12],
#     )

#     # Access results
#     pred_mask = results["pred_mask"]  # [1, 1, H, W, D]
#     pred_sdf = results["pred_sdf"]

#     print(f"✓ Prediction shape: {pred_mask.shape}")
#     print(f"✓ Unique values: {pred_mask.unique()}")
