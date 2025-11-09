import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import functools
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from monai import transforms
from monai.data import Dataset, PersistentDataset, DataLoader
from monai.transforms import MapTransform
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.spatial.distance import pdist, squareform

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Your existing transform setup
def is_dist():
    return dist.is_available() and dist.is_initialized()


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def make_cached_dataset(data, transform, cache_rate=1.0, cache_dir=None):
    # cache_rate=1.0 if it fits; else try 0.2–0.5; PersistentDataset if you want on-disk cache
    return Dataset(
        data=data, transform=transform
    )


class Keys:
    IMAGE = "mask"
    LABEL = "label"
    MASK = "mask"
    BODY_FILLED_MASK = "body_filled_mask"


def dataset_depended_transform_labels(data, label_mapping=None):
    """
    Your existing label transformation function.
    Maps labels to standardized organ labels:
      1: colon
      2: rectum
      3: small_bowel
      etc.
    """
    # Placeholder - implement your actual transform logic here
    # This should handle the label mapping from your multi-organ segmentation
    return data


def get_data_dicts_from_jsonl(
    jsonl_path: str, dataset_type: str = "training"
) -> List[Dict]:
    """
    Load data from JSONL and create MONAI-compatible data dictionaries.

    Args:
        jsonl_path: Path to JSONL file
        dataset_type: "training", "c_grade_original", or "c_grade_predicted"

    Returns:
        List of data dictionaries
    """
    data_dicts = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            if dataset_type == "training":
                # Training data has mask and body_filled_mask
                # Need to infer image path from mask path
                mask_path = data["mask"]
                image_path = mask_path.replace("/masks/", "/images/").replace(
                    "_mask", ""
                )

                data_dict = {
                    Keys.IMAGE: mask_path,
                    Keys.LABEL: mask_path,
                    Keys.MASK: mask_path,
                    "dataset_type": dataset_type,
                    "patient_id": os.path.basename(mask_path).replace(".nii.gz", ""),
                }

            elif dataset_type == "c_grade_original":
                # C-grade data has image, mask, and body_filled_mask
                data_dict = {
                    Keys.IMAGE: data["mask"],
                    Keys.LABEL: data["mask"],
                    Keys.MASK: data["mask"],
                    "dataset_type": dataset_type,
                    "patient_id": os.path.basename(data["mask"]).replace(".nii.gz", ""),
                }

            data_dicts.append(data_dict)

    return data_dicts


def get_c_grade_prediction_dicts(jsonl_path: str, pred_dir: str) -> List[Dict]:
    """
    Create data dicts for C-grade predictions.

    Args:
        jsonl_path: Path to original C-grade JSONL
        pred_dir: Directory containing _pred.nii.gz files

    Returns:
        List of data dictionaries with predictions
    """
    data_dicts = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            patient_id = os.path.basename(data["mask"]).replace(".nii.gz", "")
            pred_path = os.path.join(pred_dir, f"{patient_id}_pred.nii.gz")

            if os.path.exists(pred_path):
                # For predictions, we use the pred as the label (binary mask)
                data_dict = {
                    Keys.IMAGE: data["image"],
                    Keys.LABEL: pred_path,  # Use prediction as label
                    Keys.MASK: pred_path,
                    "dataset_type": "c_grade_predicted",
                    "patient_id": patient_id,
                    "is_binary_pred": True,  # Flag to skip label transformation
                }
                data_dicts.append(data_dict)

    return data_dicts


def get_transforms_for_analysis(is_binary_pred=False):
    """
    Get MONAI transforms for analysis (modified from your get_transforms).

    Args:
        is_binary_pred: If True, skip label transformation for binary predictions

    Returns:
        Composed transforms
    """
    data_keys = [Keys.LABEL]

    custom_transforms = [
        transforms.LoadImaged(keys=data_keys, image_only=False),
        transforms.EnsureChannelFirstd(keys=data_keys),
        transforms.Spacingd(
            keys=data_keys,
            pixdim=[1.5, 1.5, 2.0],
            mode=("nearest"),
        ),
        transforms.Orientationd(
            keys=data_keys,
            axcodes="RAS",
        ),
    ]

    # Only apply label transformation and connected component for non-binary predictions
    if not is_binary_pred:
        custom_transforms.extend(
            [
                transforms.Lambdad(
                    keys=[Keys.LABEL],
                    func=functools.partial(dataset_depended_transform_labels),
                ),
                # transforms.KeepLargestConnectedComponentd(
                #     keys=[Keys.LABEL],
                #     applied_labels=[2],  # Only for colon label
                # ),
            ]
        )
    else:
        # For binary predictions, just ensure it's binary
        custom_transforms.append(
            transforms.Lambdad(
                keys=[Keys.LABEL],
                func=lambda x: (x > 0).astype(np.float32),
            )
        )

    return transforms.Compose(custom_transforms)


class ColonLengthMetricsd(MapTransform):
    """
    MONAI transform to compute colon length and volume metrics.
    """

    def __init__(
        self,
        keys: List[str],
        colon_label: int = 1,
        spacing: List[float] = [1.5, 1.5, 2.0],
        is_binary: bool = False,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: Keys to compute metrics for (typically [Keys.LABEL])
            colon_label: Label value for colon (1 by default)
            spacing: Physical spacing in mm
            is_binary: If True, treat as binary mask instead of multi-label
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.colon_label = colon_label
        self.spacing = np.array(spacing)
        self.is_binary = is_binary

    def _compute_skeleton_length(self, binary_mask: np.ndarray) -> float:
        """
        Compute colon length using 3D skeletonization.

        Args:
            binary_mask: Binary colon mask

        Returns:
            Length in millimeters
        """
        if binary_mask.sum() == 0:
            return 0.0

        # Ensure 3D array (remove channel dimension if present)
        if binary_mask.ndim == 4:
            binary_mask = binary_mask[0]

        # Skeletonize
        skeleton = skeletonize(binary_mask.astype(np.uint8), method="lee")

        # Get skeleton coordinates
        skeleton_points = np.argwhere(skeleton > 0)

        if len(skeleton_points) < 2:
            return 0.0

        # Method 1: Simple skeleton voxel count scaled by average spacing
        # This is fast but less accurate
        # length_mm = skeleton.sum() * np.mean(self.spacing)

        # Method 2: Sum of distances between consecutive points
        # Sort points to create a path (simplified - not optimal path)
        # For better results, you'd use minimum spanning tree or graph algorithms

        # Calculate pairwise distances in physical space
        skeleton_points_physical = skeleton_points * self.spacing

        # Use minimum spanning tree approach for better path estimation
        length_mm = self._compute_path_length_mst(skeleton_points_physical)

        return length_mm

    def _compute_path_length_mst(self, points: np.ndarray) -> float:
        """
        Compute path length using Minimum Spanning Tree approach.
        This provides a better approximation of the actual colon path.

        Args:
            points: Skeleton points in physical coordinates (N x 3)

        Returns:
            Total path length in mm
        """
        if len(points) < 2:
            return 0.0

        # For very long paths, sample points to avoid memory issues
        max_points = 5000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        # Compute pairwise distances
        try:
            distances = pdist(points, metric="euclidean")
            dist_matrix = squareform(distances)

            # Simple MST using Prim's algorithm
            n_points = len(points)
            visited = np.zeros(n_points, dtype=bool)
            visited[0] = True
            total_length = 0.0

            for _ in range(n_points - 1):
                min_dist = np.inf
                min_idx = -1

                for i in range(n_points):
                    if visited[i]:
                        for j in range(n_points):
                            if not visited[j] and dist_matrix[i, j] < min_dist:
                                min_dist = dist_matrix[i, j]
                                min_idx = j

                if min_idx != -1:
                    visited[min_idx] = True
                    total_length += min_dist

            return total_length

        except MemoryError:
            # Fallback: simple consecutive point distances
            sorted_indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
            sorted_points = points[sorted_indices]
            diffs = np.diff(sorted_points, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            return np.sum(distances)

    def _compute_centerline_length(self, binary_mask: np.ndarray) -> float:
        """
        Alternative method: Compute length using distance transform and centerline.
        More accurate for tubular structures.

        Args:
            binary_mask: Binary colon mask

        Returns:
            Length in millimeters
        """
        if binary_mask.sum() == 0:
            return 0.0

        if binary_mask.ndim == 4:
            binary_mask = binary_mask[0]

        # Compute distance transform
        distance = ndimage.distance_transform_edt(binary_mask, sampling=self.spacing)

        # Find centerline by skeletonizing the mask
        skeleton = skeletonize(binary_mask.astype(np.uint8), method="lee")

        # Get skeleton coordinates
        skeleton_coords = np.argwhere(skeleton > 0)

        if len(skeleton_coords) < 2:
            return 0.0

        # Convert to physical coordinates
        skeleton_physical = skeleton_coords * self.spacing

        # Compute path length
        length_mm = self._compute_path_length_mst(skeleton_physical)

        return length_mm

    def _compute_volume(self, binary_mask: np.ndarray) -> float:
        """
        Compute colon volume in milliliters.

        Args:
            binary_mask: Binary colon mask

        Returns:
            Volume in milliliters
        """
        if binary_mask.ndim == 4:
            binary_mask = binary_mask[0]

        voxel_volume_mm3 = np.prod(self.spacing)
        volume_ml = binary_mask.sum() * voxel_volume_mm3 / 1000.0

        return float(volume_ml)

    def _compute_surface_area(self, binary_mask: np.ndarray) -> float:
        """
        Compute approximate surface area using voxel counting.

        Args:
            binary_mask: Binary colon mask

        Returns:
            Surface area in cm²
        """
        if binary_mask.ndim == 4:
            binary_mask = binary_mask[0]

        # Count boundary voxels (simple approach)
        # More sophisticated: use marching cubes for actual surface
        eroded = ndimage.binary_erosion(binary_mask)
        boundary = binary_mask & ~eroded

        # Approximate surface area
        voxel_face_area = np.mean(
            [
                self.spacing[i] * self.spacing[j]
                for i in range(3)
                for j in range(i + 1, 3)
            ]
        )
        surface_area_cm2 = boundary.sum() * voxel_face_area / 100.0

        return float(surface_area_cm2)

    def __call__(self, data: Dict) -> Dict:
        """
        Apply the transform to compute metrics.

        Args:
            data: Data dictionary

        Returns:
            Data dictionary with added metrics
        """
        d = dict(data)

        for key in self.key_iterator(d):
            label_array = d[key]

            # Convert to numpy if tensor
            if torch.is_tensor(label_array):
                label_array = label_array.numpy()

            # Extract colon mask
            if self.is_binary:
                colon_mask = (label_array > 0).astype(np.uint8)
            else:
                colon_mask = (label_array == self.colon_label).astype(np.uint8)

            # Compute metrics
            length_skeleton = self._compute_skeleton_length(colon_mask)
            length_centerline = self._compute_centerline_length(colon_mask)
            volume = self._compute_volume(colon_mask)
            surface_area = self._compute_surface_area(colon_mask)

            # Add metrics to data dictionary
            d["colon_length_skeleton_mm"] = length_skeleton
            d["colon_length_centerline_mm"] = length_centerline
            d["colon_length_cm"] = length_centerline / 10.0  # Use centerline as primary
            d["colon_volume_ml"] = volume
            d["colon_surface_area_cm2"] = surface_area

            # Additional derived metrics
            if volume > 0 and length_centerline > 0:
                d["colon_avg_diameter_mm"] = np.sqrt(
                    volume * 1000 / (np.pi * length_centerline)
                )
            else:
                d["colon_avg_diameter_mm"] = 0.0

        return d


# def process_dataset_with_monai(
#     data_dicts: List[Dict],
#     batch_size: int = 1,
#     num_workers: int = 4,
#     is_binary_pred: bool = False,
# ) -> pd.DataFrame:
#     """
#     Process dataset using MONAI pipeline to compute colon metrics.

#     Args:
#         data_dicts: List of data dictionaries
#         batch_size: Batch size for processing
#         num_workers: Number of workers for data loading
#         is_binary_pred: Whether this is binary prediction data

#     Returns:
#         DataFrame with all computed metrics
#     """
#     # Get transforms
#     base_transforms = get_transforms_for_analysis(is_binary_pred=is_binary_pred)

#     # Add metric computation transform
#     metric_transform = ColonLengthMetricsd(
#         keys=[Keys.LABEL],
#         colon_label=1,
#         spacing=[1.5, 1.5, 2.0],
#         is_binary=is_binary_pred,
#     )

#     # Combine transforms
#     all_transforms = transforms.Compose([base_transforms, metric_transform])

#     # Create dataset
#     dataset = Dataset(data=data_dicts, transform=all_transforms)

#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         collate_fn=list,  # Return list of dicts instead of batched dict
#     )

#     # Process all data
#     results = []

#     logging.info(f"Processing {len(data_dicts)} cases...")
#     for batch_idx, batch_data in enumerate(dataloader):
#         for item in batch_data:
#             result = {
#                 "patient_id": item["patient_id"],
#                 "dataset_type": item["dataset_type"],
#                 "length_skeleton_mm": item["colon_length_skeleton_mm"],
#                 "length_centerline_mm": item["colon_length_centerline_mm"],
#                 "length_cm": item["colon_length_cm"],
#                 "volume_ml": item["colon_volume_ml"],
#                 "surface_area_cm2": item["colon_surface_area_cm2"],
#                 "avg_diameter_mm": item["colon_avg_diameter_mm"],
#             }
#             results.append(result)

#         if (batch_idx + 1) % 10 == 0:
#             logging.info(f"  Processed {(batch_idx + 1) * batch_size}/{len(data_dicts)} cases")

#     return pd.DataFrame(results)


def process_dataset_with_monai(
    data_dicts: List[Dict],
    batch_size: int = 1,
    num_workers: int = 4,
    is_binary_pred: bool = False,
    cache_rate: float = 1.0,
    cache_dir: str = None,
) -> pd.DataFrame:

    base_transforms = get_transforms_for_analysis(is_binary_pred=is_binary_pred)

    metric_transform = ColonLengthMetricsd(
        keys=[Keys.LABEL],
        colon_label=1,
        spacing=[1.5, 1.5, 2.0],
        is_binary=is_binary_pred,
    )

    all_transforms = transforms.Compose([base_transforms, metric_transform])

    # ---- Dataset (cached) ----
    dataset = make_cached_dataset(
        data=data_dicts,
        transform=all_transforms,
        cache_rate=cache_rate,
        cache_dir=cache_dir,
    )

    # ---- Sampler per rank ----
    sampler = (
        DistributedSampler(dataset, shuffle=False, drop_last=False)
        if is_dist()
        else None
    )

    # ---- DataLoader tuned for throughput ----
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else False,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=list,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ---- Per-rank processing ----
    if sampler is not None:
        sampler.set_epoch(0)

    local_results = []
    if is_main_process():
        logging.info(
            f"Processing {len(data_dicts)} cases (world_size={get_world_size()})..."
        )

    for batch_idx, batch_data in enumerate(dataloader):
        for item in batch_data:
            local_results.append(
                {
                    "patient_id": item["patient_id"],
                    "dataset_type": item["dataset_type"],
                    "length_skeleton_mm": item["colon_length_skeleton_mm"],
                    "length_centerline_mm": item["colon_length_centerline_mm"],
                    "length_cm": item["colon_length_cm"],
                    "volume_ml": item["colon_volume_ml"],
                    "surface_area_cm2": item["colon_surface_area_cm2"],
                    "avg_diameter_mm": item["colon_avg_diameter_mm"],
                }
            )

        if is_main_process() and ((batch_idx + 1) % max(1, (16 // batch_size)) == 0):
            logging.info(f"  Rank 0 progress: {(batch_idx + 1) * batch_size}")

    # ---- Gather results across ranks ----
    if is_dist():
        all_results = [None for _ in range(get_world_size())]
        dist.all_gather_object(all_results, local_results)
        merged = []
        for r in all_results:
            merged.extend(r)
        return pd.DataFrame(merged)
    else:
        return pd.DataFrame(local_results)


def create_comprehensive_plots(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Create comprehensive visualization plots.

    Args:
        df: DataFrame with all measurements
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Plot 1: Length distribution comparison (box + violin)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot
    df.boxplot(column="length_cm", by="dataset_type", ax=axes[0])
    axes[0].set_title(
        "Colon Length Distribution (Box Plot)", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Dataset", fontsize=12)
    axes[0].set_ylabel("Colon Length (cm)", fontsize=12)
    axes[0].get_figure().suptitle("")  # Remove automatic title
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha="right")

    # Violin plot
    sns.violinplot(data=df, x="dataset_type", y="length_cm", ax=axes[1])
    axes[1].set_title(
        "Colon Length Distribution (Violin Plot)", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Dataset", fontsize=12)
    axes[1].set_ylabel("Colon Length (cm)", fontsize=12)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "01_length_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Histogram overlays
    plt.figure(figsize=(12, 6))
    for dataset in df["dataset_type"].unique():
        subset = df[df["dataset_type"] == dataset]["length_cm"].dropna()
        plt.hist(
            subset, bins=30, alpha=0.5, label=dataset, edgecolor="black", linewidth=0.5
        )

    plt.xlabel("Colon Length (cm)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(
        "Colon Length Distribution - Histogram Overlay", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "02_length_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Paired comparison for C-grade (original vs predicted)
    c_orig = df[df["dataset_type"] == "c_grade_original"].set_index("patient_id")
    c_pred = df[df["dataset_type"] == "c_grade_predicted"].set_index("patient_id")

    common_patients = c_orig.index.intersection(c_pred.index)

    if len(common_patients) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        orig_lengths = c_orig.loc[common_patients, "length_cm"].values
        pred_lengths = c_pred.loc[common_patients, "length_cm"].values

        # Scatter plot
        axes[0].scatter(
            orig_lengths,
            pred_lengths,
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )

        # Perfect agreement line
        min_val = min(orig_lengths.min(), pred_lengths.min())
        max_val = max(orig_lengths.max(), pred_lengths.max())
        axes[0].plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Agreement",
            alpha=0.7,
        )

        # Add correlation coefficient
        correlation = np.corrcoef(orig_lengths, pred_lengths)[0, 1]
        axes[0].text(
            0.05,
            0.95,
            f"r = {correlation:.3f}",
            transform=axes[0].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        axes[0].set_xlabel("Original C-Grade Length (cm)", fontsize=12)
        axes[0].set_ylabel("Predicted Length (cm)", fontsize=12)
        axes[0].set_title(
            "Original vs Predicted: Correlation", fontsize=14, fontweight="bold"
        )
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_aspect("equal", adjustable="box")

        # Bland-Altman plot
        mean_lengths = (orig_lengths + pred_lengths) / 2
        diff_lengths = pred_lengths - orig_lengths
        mean_diff = np.mean(diff_lengths)
        std_diff = np.std(diff_lengths)

        axes[1].scatter(
            mean_lengths,
            diff_lengths,
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[1].axhline(
            mean_diff,
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Mean Diff: {mean_diff:.2f} cm",
        )
        axes[1].axhline(
            mean_diff + 1.96 * std_diff,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"+1.96 SD: {mean_diff + 1.96*std_diff:.2f} cm",
        )
        axes[1].axhline(
            mean_diff - 1.96 * std_diff,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"-1.96 SD: {mean_diff - 1.96*std_diff:.2f} cm",
        )
        axes[1].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)

        axes[1].set_xlabel("Mean Length (cm)", fontsize=12)
        axes[1].set_ylabel("Difference (Predicted - Original) cm", fontsize=12)
        axes[1].set_title("Bland-Altman Plot", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "03_c_grade_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 4: Individual patient improvement bars
        differences = pred_lengths - orig_lengths

        plt.figure(figsize=(14, 6))
        colors = ["green" if d > 0 else "red" for d in differences]
        bars = plt.bar(
            range(len(differences)),
            differences,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        plt.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        plt.axhline(
            y=mean_diff,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_diff:.2f} cm",
        )

        plt.xlabel("Patient Index", fontsize=12)
        plt.ylabel("Length Difference (Predicted - Original) cm", fontsize=12)
        plt.title(
            "Per-Patient Length Change: Improvement Analysis",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "04_improvement_bars.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 5: Volume vs Length relationship
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for dataset in df["dataset_type"].unique():
            subset = df[df["dataset_type"] == dataset]
            axes[0].scatter(
                subset["length_cm"],
                subset["volume_ml"],
                label=dataset,
                alpha=0.6,
                s=100,
                edgecolors="black",
                linewidth=0.5,
            )

        axes[0].set_xlabel("Colon Length (cm)", fontsize=12)
        axes[0].set_ylabel("Colon Volume (ml)", fontsize=12)
        axes[0].set_title(
            "Length vs Volume Relationship", fontsize=14, fontweight="bold"
        )
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Average diameter distribution
        for dataset in df["dataset_type"].unique():
            subset = df[df["dataset_type"] == dataset]["avg_diameter_mm"].dropna()
            axes[1].hist(
                subset,
                bins=20,
                alpha=0.5,
                label=dataset,
                edgecolor="black",
                linewidth=0.5,
            )

        axes[1].set_xlabel("Average Colon Diameter (mm)", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title(
            "Average Diameter Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "05_volume_diameter_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    logging.info(f"\nAll plots saved to '{output_dir}/' directory")


def generate_statistical_summary(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Generate comprehensive statistical summary and tests.

    Args:
        df: DataFrame with measurements
        output_dir: Output directory
    """
    from scipy import stats

    logging.info("\n" + "=" * 80)
    logging.info("COMPREHENSIVE STATISTICAL ANALYSIS")
    logging.info("=" * 80)

    # Overall summary statistics
    summary = (
        df.groupby("dataset_type")
        .agg(
            {
                "length_cm": ["count", "mean", "std", "min", "median", "max"],
                "volume_ml": ["mean", "std", "median"],
                "avg_diameter_mm": ["mean", "std", "median"],
            }
        )
        .round(2)
    )

    logging.info("\n1. DESCRIPTIVE STATISTICS")
    logging.info("-" * 80)
    logging.info(summary)

    # Statistical tests for C-grade comparison
    c_orig = df[df["dataset_type"] == "c_grade_original"].set_index("patient_id")
    c_pred = df[df["dataset_type"] == "c_grade_predicted"].set_index("patient_id")

    common_patients = c_orig.index.intersection(c_pred.index)

    if len(common_patients) > 0:
        orig_lengths = c_orig.loc[common_patients, "length_cm"].values
        pred_lengths = c_pred.loc[common_patients, "length_cm"].values

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(pred_lengths, orig_lengths)

        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_p_value = stats.wilcoxon(pred_lengths, orig_lengths)

        # Effect size (Cohen's d)
        differences = pred_lengths - orig_lengths
        cohens_d = np.mean(differences) / np.std(differences)

        logging.info("\n2. PAIRED COMPARISON: C-GRADE ORIGINAL vs PREDICTED")
        logging.info("-" * 80)
        logging.info(f"Number of paired samples: {len(common_patients)}")
        logging.info(f"Mean difference (Pred - Orig): {np.mean(differences):.2f} cm")
        logging.info(f"Std of differences: {np.std(differences):.2f} cm")
        logging.info(f"\nPaired t-test:")
        logging.info(f"  t-statistic: {t_stat:.4f}")
        logging.info(f"  p-value: {p_value:.4f}")
        logging.info(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        logging.info(f"\nWilcoxon signed-rank test:")
        logging.info(f"  W-statistic: {w_stat:.4f}")
        logging.info(f"  p-value: {w_p_value:.4f}")
        logging.info(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

        # Correlation
        correlation = np.corrcoef(orig_lengths, pred_lengths)[0, 1]
        logging.info(f"Pearson correlation: {correlation:.4f}")

        # Improvement metrics
        improved = (differences > 0).sum()
        worsened = (differences < 0).sum()
        unchanged = (differences == 0).sum()

        logging.info(f"\n3. IMPROVEMENT ANALYSIS")
        logging.info("-" * 80)
        logging.info(
            f"Cases with increased length: {improved} ({100*improved/len(differences):.1f}%)"
        )
        logging.info(
            f"Cases with decreased length: {worsened} ({100*worsened/len(differences):.1f}%)"
        )
        logging.info(f"Cases unchanged: {unchanged} ({100*unchanged/len(differences):.1f}%)")

        # Mean absolute error and relative error
        mae = np.mean(np.abs(differences))
        mape = np.mean(np.abs(differences) / orig_lengths) * 100

        logging.info(f"\nMean Absolute Error: {mae:.2f} cm")
        logging.info(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # ANOVA across all groups
    groups = [
        df[df["dataset_type"] == dt]["length_cm"].dropna().values
        for dt in df["dataset_type"].unique()
    ]

    if len(groups) > 2:
        f_stat, anova_p = stats.f_oneway(*groups)

        logging.info(f"\n4. ONE-WAY ANOVA (ALL GROUPS)")
        logging.info("-" * 80)
        logging.info(f"F-statistic: {f_stat:.4f}")
        logging.info(f"p-value: {anova_p:.4f}")
        logging.info(f"Significant at α=0.05: {'Yes' if anova_p < 0.05 else 'No'}")

    logging.info("\n" + "=" * 80 + "\n")

    # Save summary to CSV
    summary.to_csv(os.path.join(output_dir, "statistical_summary.csv"))

    # Save detailed comparison if available
    if len(common_patients) > 0:
        comparison_df = pd.DataFrame(
            {
                "patient_id": common_patients,
                "original_length_cm": orig_lengths,
                "predicted_length_cm": pred_lengths,
                "difference_cm": differences,
                "percent_change": (differences / orig_lengths) * 100,
            }
        )
        comparison_df.to_csv(
            os.path.join(output_dir, "c_grade_paired_comparison.csv"), index=False
        )
        logging.info(
            f"Detailed comparison saved to '{output_dir}/c_grade_paired_comparison.csv'"
        )


# def main():
#     """
#     Main execution function integrating MONAI transforms.
#     """
#     # ==================== CONFIGURATION ====================
#     # UPDATE THESE PATHS TO YOUR ACTUAL DATA
#     # TRAINING_JSONL = "/path/to/training_data.jsonl"
#     # C_GRADE_JSONL = "/path/to/c_grade_data.jsonl"
#     # PRED_DIR = "/path/to/predictions"
#     # OUTPUT_DIR = "colon_analysis_results"

#     TRAINING_JSONL = "/home/yb107/cvpr2025/DukeDiffSeg/data/mobina_mixed_colon_dataset/mobina_mixed_colon_dataset_with_body_filled.jsonl"
#     C_GRADE_JSONL = "/home/yb107/cvpr2025/DukeDiffSeg/data/c_grade_colons/3d_vlsmv2_c_grade_colon_dataset_with_body_filled.jsonl"
#     PRED_DIR = "/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/5.1/inference_c_grade_550_gs_2.0_final_small_with_skeletonization"
#     OUTPUT_DIR = "/home/yb107/cvpr2025/DukeDiffSeg/notebook/length_analysis"

#     BATCH_SIZE = 4
#     NUM_WORKERS = 4
#     # =======================================================

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     logging.info("=" * 80)
#     logging.info("COLON LENGTH ANALYSIS USING MONAI TRANSFORMS")
#     logging.info("=" * 80)

#     # Step 1: Load data dictionaries
#     logging.info("\n[1/5] Loading data dictionaries...")

#     training_dicts = get_data_dicts_from_jsonl(TRAINING_JSONL, dataset_type="training")
#     logging.info(f"  Loaded {len(training_dicts)} training cases")

#     c_grade_dicts = get_data_dicts_from_jsonl(
#         C_GRADE_JSONL, dataset_type="c_grade_original"
#     )
#     logging.info(f"  Loaded {len(c_grade_dicts)} C-grade original cases")

#     c_pred_dicts = get_c_grade_prediction_dicts(C_GRADE_JSONL, PRED_DIR)
#     logging.info(f"  Loaded {len(c_pred_dicts)} C-grade prediction cases")

#     # Step 2: Process datasets with MONAI
#     logging.info("\n[2/5] Processing datasets with MONAI transforms...")

#     logging.info("\n  Processing training data...")
#     df_training = process_dataset_with_monai(
#         training_dicts,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         is_binary_pred=False,
#     )

#     logging.info("\n  Processing C-grade original data...")
#     df_c_grade = process_dataset_with_monai(
#         c_grade_dicts,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         is_binary_pred=False,
#     )

#     logging.info("\n  Processing C-grade predictions...")
#     df_c_pred = process_dataset_with_monai(
#         c_pred_dicts,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         is_binary_pred=True,
#     )

#     # Step 3: Combine results
#     logging.info("\n[3/5] Combining results...")
#     df_all = pd.concat([df_training, df_c_grade, df_c_pred], ignore_index=True)

#     # Save raw measurements
#     output_csv = os.path.join(OUTPUT_DIR, "colon_measurements_all.csv")
#     df_all.to_csv(output_csv, index=False)
#     logging.info(f"  Saved {len(df_all)} measurements to '{output_csv}'")

#     # Step 4: Generate visualizations
#     logging.info("\n[4/5] Generating visualizations...")
#     create_comprehensive_plots(df_all, output_dir=OUTPUT_DIR)

#     # Step 5: Statistical analysis
#     logging.info("\n[5/5] Performing statistical analysis...")
#     generate_statistical_summary(df_all, output_dir=OUTPUT_DIR)

#     logging.info("\n" + "=" * 80)
#     logging.info("ANALYSIS COMPLETE!")
#     logging.info(f"All results saved to '{OUTPUT_DIR}/' directory")
#     logging.info("=" * 80 + "\n")


# main()


# ======== NEW HELPERS (place near top of file) ========
def _safe_write_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _safe_write_parquet(df: pd.DataFrame, path: str):
    try:
        import pyarrow  # noqa: F401

        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:
        logging.info(f"Parquet not written ({e}); continuing with CSV only.")


def _load_df_if_exists(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logging.warning(f"Failed to read existing CSV '{csv_path}': {e}")
    return pd.DataFrame()


def _dedup_by_pid(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" in df.columns:
        # keep last occurrence (newly computed wins)
        return df.drop_duplicates(subset=["patient_id"], keep="last")
    return df


def _expected_ids_from_dicts(data_dicts: List[Dict]) -> List[str]:
    # robustly extract patient_ids
    ids = []
    for d in data_dicts:
        pid = d.get("patient_id")
        if pid is None:
            # fallback to basename of label/mask
            p = d.get("label") or d.get("mask") or d.get("LABEL") or d.get("MASK")
            if p:
                pid = os.path.basename(str(p)).replace(".nii.gz", "")
        if pid is None:
            continue
        ids.append(str(pid))
    return ids


def _bcast_obj(obj, src=0):
    if not is_dist():
        return obj
    obj_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def _filter_dicts_by_ids(data_dicts: List[Dict], keep_ids: set) -> List[Dict]:
    out = []
    for d in data_dicts:
        pid = d.get("patient_id")
        if pid in keep_ids:
            out.append(d)
    return out


def _resume_step(
    step_name: str,
    data_dicts: List[Dict],
    out_csv: str,
    out_parq: str,
    *,
    batch_size: int,
    num_workers: int,
    is_binary_pred: bool,
    cache_rate: float,
) -> pd.DataFrame:
    """
    Resume-aware processing for a single dataset step.
    - Loads existing CSV on rank-0
    - Finds missing patient_ids
    - Broadcasts missing_ids
    - Processes only missing
    - Merges & saves on rank-0
    - Returns full DataFrame (broadcast not needed; only used by rank-0 later)
    """
    rank = get_rank()
    world = get_world_size()

    # ---- rank-0: read existing & compute missing IDs ----
    if is_main_process():
        existing_df = _load_df_if_exists(out_csv)
        existing_ids = (
            set(existing_df["patient_id"].astype(str))
            if not existing_df.empty
            else set()
        )
        expected_ids = set(_expected_ids_from_dicts(data_dicts))
        missing_ids = sorted(list(expected_ids - existing_ids))
        logging.info(
            f"[{step_name}] total={len(expected_ids)} existing={len(existing_ids)} missing={len(missing_ids)}"
        )
    else:
        existing_df = pd.DataFrame()
        missing_ids = None

    # ---- broadcast missing_ids to all ranks ----
    missing_ids = _bcast_obj(missing_ids, src=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if is_dist():
        dist.barrier()

    # ---- if nothing missing, everyone returns (rank-0 returns full existing) ----
    if len(missing_ids) == 0:
        if is_main_process():
            logging.info(f"[{step_name}] up-to-date. Skipping compute.")
            return existing_df
        else:
            return pd.DataFrame()  # other ranks return dummy

    # ---- filter to missing only ----
    missing_set = set(missing_ids)
    dicts_missing = _filter_dicts_by_ids(data_dicts, missing_set)

    # ---- compute only missing subset ----
    df_missing = process_dataset_with_monai(
        dicts_missing,
        batch_size=batch_size,
        num_workers=num_workers,
        is_binary_pred=is_binary_pred,
        cache_rate=cache_rate,
    )

    # sync before write
    if is_dist():
        dist.barrier()

    # ---- rank-0 merges & saves ----
    if is_main_process():
        merged = existing_df
        if not df_missing.empty:
            merged = pd.concat([existing_df, df_missing], ignore_index=True)
        merged = _dedup_by_pid(merged)

        _safe_write_csv(merged, out_csv)
        _safe_write_parquet(merged, out_parq)
        logging.info(f"[{step_name}] wrote '{out_csv}' with {len(merged)} rows.")
        return merged
    else:
        return pd.DataFrame()


def main_distributed():
    setup_dist()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ==================== CONFIG ====================
    TRAINING_JSONL = "/home/yb107/cvpr2025/DukeDiffSeg/data/mobina_mixed_colon_dataset/mobina_mixed_colon_dataset_with_body_filled.jsonl"
    C_GRADE_JSONL = "/home/yb107/cvpr2025/DukeDiffSeg/data/c_grade_colons/3d_vlsmv2_c_grade_colon_dataset_with_body_filled.jsonl"
    PRED_DIR = "/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/5.1/inference_c_grade_550_gs_2.0_final_small_with_skeletonization"
    OUTPUT_DIR = "/home/yb107/cvpr2025/DukeDiffSeg/notebook/length_analysis"

    BATCH_SIZE = 4
    NUM_WORKERS = max(4, os.cpu_count() // max(1, world_size))
    CACHE_RATE = 1.0
    # ===============================================

    if is_main_process():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info("=" * 80)
        logging.info("COLON LENGTH ANALYSIS USING MONAI (Distributed, Resumable)")
        logging.info("=" * 80)
        logging.info("\n[1/5] Loading data dictionaries...")

    training_dicts = get_data_dicts_from_jsonl(TRAINING_JSONL, dataset_type="training")
    c_grade_dicts = get_data_dicts_from_jsonl(
        C_GRADE_JSONL, dataset_type="c_grade_original"
    )
    c_pred_dicts = get_c_grade_prediction_dicts(C_GRADE_JSONL, PRED_DIR)

    if is_main_process():
        logging.info(f"  Loaded {len(training_dicts)} training cases")
        logging.info(f"  Loaded {len(c_grade_dicts)} C-grade original cases")
        logging.info(f"  Loaded {len(c_pred_dicts)} C-grade prediction cases")
        logging.info("\n[2/5] Processing datasets with MONAI transforms (resumable)...")

    # ---- Step: training (resumable) ----
    train_csv = os.path.join(OUTPUT_DIR, "training_measurements.csv")
    train_parq = os.path.join(OUTPUT_DIR, "training_measurements.parquet")
    df_training = _resume_step(
        "training",
        training_dicts,
        train_csv,
        train_parq,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_binary_pred=False,
        cache_rate=CACHE_RATE,
    )

    # ---- Step: c-grade original (resumable) ----
    c_orig_csv = os.path.join(OUTPUT_DIR, "c_grade_original_measurements.csv")
    c_orig_parq = os.path.join(OUTPUT_DIR, "c_grade_original_measurements.parquet")
    df_c_grade = _resume_step(
        "c_grade_original",
        c_grade_dicts,
        c_orig_csv,
        c_orig_parq,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_binary_pred=False,
        cache_rate=CACHE_RATE,
    )

    # ---- Step: c-grade predicted (resumable) ----
    c_pred_csv = os.path.join(OUTPUT_DIR, "c_grade_predicted_measurements.csv")
    c_pred_parq = os.path.join(OUTPUT_DIR, "c_grade_predicted_measurements.parquet")
    df_c_pred = _resume_step(
        "c_grade_predicted",
        c_pred_dicts,
        c_pred_csv,
        c_pred_parq,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_binary_pred=True,
        cache_rate=CACHE_RATE,
    )

    # ---- Sync all ranks before final combine ----
    if is_dist():
        dist.barrier()

    # ---- Combine on rank-0, also resumable ----
    if is_main_process():
        logging.info("\n[3/5] Combining results (resumable)...")

        # If any of the per-step DFs is empty (because this is a non-main rank),
        # load from disk here to ensure we have them.
        if df_training is None or df_training.empty:
            df_training = _load_df_if_exists(train_csv)
        if df_c_grade is None or df_c_grade.empty:
            df_c_grade = _load_df_if_exists(c_orig_csv)
        if df_c_pred is None or df_c_pred.empty:
            df_c_pred = _load_df_if_exists(c_pred_csv)

        df_all = pd.concat([df_training, df_c_grade, df_c_pred], ignore_index=True)
        df_all = _dedup_by_pid(df_all)

        all_csv = os.path.join(OUTPUT_DIR, "colon_measurements_all.csv")
        all_parq = os.path.join(OUTPUT_DIR, "colon_measurements_all.parquet")

        # If exists & already current (same #rows and patient_ids), we can skip re-write.
        existing_all = _load_df_if_exists(all_csv)
        need_write = True
        if not existing_all.empty:
            try:
                s_new = set(df_all["patient_id"].astype(str))
                s_old = set(existing_all["patient_id"].astype(str))
                if s_new == s_old and len(existing_all) == len(df_all):
                    need_write = False
            except Exception:
                pass

        if need_write:
            _safe_write_csv(df_all, all_csv)
            _safe_write_parquet(df_all, all_parq)
            logging.info(f"  Saved {len(df_all)} combined measurements to '{all_csv}'")
        else:
            logging.info("  Combined file already up-to-date. Skipping write.")

        logging.info("\n[4/5] Generating visualizations...")
        create_comprehensive_plots(df_all, output_dir=OUTPUT_DIR)

        logging.info("\n[5/5] Performing statistical analysis...")
        generate_statistical_summary(df_all, output_dir=OUTPUT_DIR)

        logging.info("\n" + "=" * 80)
        logging.info("ANALYSIS COMPLETE (resumable)!")
        logging.info(f"All results saved to '{OUTPUT_DIR}/' directory")
        logging.info("=" * 80 + "\n")

    cleanup_dist()

if __name__ == "__main__":
    main_distributed()
