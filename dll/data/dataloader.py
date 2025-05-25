import os
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import logging

from dll.data.transforms import ITransform
from dll.data.augmentation import KeypointAugmentation
from dll.models.heatmap_head import generate_target_heatmap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KeypointDatasetError(Exception):
    """Custom exception for dataset-related errors."""
    pass

class ValidationError(KeypointDatasetError):
    """Specific validation error."""
    pass

@dataclass
class AnnotationData:
    """Structured annotation data for type safety and clarity."""
    keypoints: torch.Tensor  # Shape: [num_persons, num_keypoints, 2]
    visibilities: torch.Tensor  # Shape: [num_persons, num_keypoints]
    classes: torch.Tensor  # Shape: [num_persons]
    bboxes: torch.Tensor  # Shape: [num_persons, 4]

    def truncate(self, max_persons: int) -> 'AnnotationData':
        """Truncate to maximum number of persons."""
        return AnnotationData(
            keypoints=self.keypoints[:max_persons],
            visibilities=self.visibilities[:max_persons],
            classes=self.classes[:max_persons],
            bboxes=self.bboxes[:max_persons]
        )

    @property
    def num_persons(self) -> int:
        return self.keypoints.shape[0]

@dataclass
class ImageData:
    """Structured image data."""
    image: Image.Image
    orig_size: Tuple[int, int]

class LRUCache:
    """Simple LRU cache implementation."""
    def __init__(self, maxsize: int = 128):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: Any) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

class OptimizedKeypointsDataset(Dataset):
    """Optimized dataset class for loading images and keypoints for pose estimation."""

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        img_size: int = 512,
        grayscale: bool = False,
        num_keypoints: int = 17,
        heatmap_size: Tuple[int, int] = (56, 56),
        transform: Optional[ITransform] = None,
        augmentation: Optional[KeypointAugmentation] = None,
        max_persons: int = 10,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize the optimized dataset.

        Args:
            dataset_dir: Path to the dataset directory.
            split: Dataset split ("train", "val", or "test").
            img_size: Target image size (assumes square images).
            grayscale: Load images in grayscale if True.
            num_keypoints: Number of keypoints per person.
            heatmap_size: Size of the output heatmap (height, width).
            transform: Image transformation pipeline.
            augmentation: Keypoint augmentation pipeline (applied only for training).
            max_persons: Maximum number of persons to process per image.
            enable_caching: Enable annotation caching for performance.
            cache_size: Maximum cache size for images.
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.img_size = img_size
        self.grayscale = grayscale
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.max_persons = max_persons
        self.enable_caching = enable_caching

        # Initialize caches
        self._annotation_cache = {} if enable_caching else None
        self._image_cache = LRUCache(maxsize=cache_size) if enable_caching else None

        # Initialize transforms
        self.transform = transform or ITransform(img_size=img_size, clip_limit=1.5, tile_size=(8, 8))
        self.augmentation = augmentation if split == "train" else None

        # Validate and load dataset
        self._validate_dataset_structure()
        self.img_files, self.label_files = self._load_file_pairs()

        logger.info(f"Loaded {len(self.img_files)} images with valid labels for {split} split")

    def _validate_dataset_structure(self) -> None:
        """Validate dataset directory structure."""
        self.img_dir = self.dataset_dir / self.split / "images"
        self.label_dir = self.dataset_dir / self.split / "labels"

        if not self.img_dir.exists():
            raise KeypointDatasetError(f"Image directory not found: {self.img_dir}")
        if not self.label_dir.exists():
            raise KeypointDatasetError(f"Label directory not found: {self.label_dir}")

    def _load_file_pairs(self) -> Tuple[List[Path], List[Path]]:
        """Load and validate image and label file pairs."""
        img_extensions = {".png", ".jpg", ".jpeg"}
        img_files = sorted([
            f for f in self.img_dir.glob("*")
            if f.suffix.lower() in img_extensions
        ])

        valid_pairs = []
        for img_path in img_files:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
            else:
                logger.warning(f"No label file found for {img_path}")

        if not valid_pairs:
            raise KeypointDatasetError(f"No valid image-label pairs found in {self.dataset_dir}")

        return zip(*valid_pairs)

    def _get_file_paths(self, idx: int) -> Tuple[Path, Path]:
        """Get file paths for given index with bounds checking."""
        if idx >= len(self.img_files):
            raise KeypointDatasetError(f"Index {idx} out of range")
        return self.img_files[idx], self.label_files[idx]

    def _load_and_process_image(self, img_path: Path) -> ImageData:
        """Load and preprocess image with optional caching."""
        # Check cache first
        if self._image_cache:
            cached_data = self._image_cache.get(str(img_path))
            if cached_data is not None:
                return cached_data

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise KeypointDatasetError(f"Failed to load image {img_path}")

        orig_size = (img.shape[1], img.shape[0])
        if not self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_data = ImageData(
            image=Image.fromarray(img, mode="L" if self.grayscale else "RGB"),
            orig_size=orig_size
        )

        # Cache if enabled
        if self._image_cache:
            self._image_cache.put(str(img_path), image_data)

        return image_data

    def _parse_label_file_vectorized(self, label_path: Path) -> AnnotationData:
        """Optimized label parsing using numpy vectorization."""
        try:
            # Đọc từng dòng thay vì loadtxt
            lines = []
            max_cols = 0
            with open(label_path) as f:
                for line in f:
                    data = [float(x) for x in line.strip().split()]
                    lines.append(data)
                    max_cols = max(max_cols, len(data))

            if not lines:
                return self._create_empty_annotation()

            # Chuẩn hóa số cột cho mỗi dòng
            normalized_lines = []
            for line in lines:
                if len(line) < max_cols:
                    # Pad với 0 nếu thiếu
                    line.extend([0] * (max_cols - len(line)))
                normalized_lines.append(line)

            # Chuyển thành numpy array
            data_matrix = np.array(normalized_lines, dtype=np.float32)

            # Xử lý dữ liệu
            classes = data_matrix[:, 0].astype(np.int32)
            bboxes = data_matrix[:, 1:5]

            # Handle keypoint data
            keypoint_data = data_matrix[:, 5:]
            n_persons = len(data_matrix)

            # Ensure we have complete keypoint triplets
            expected_cols = self.num_keypoints * 3
            if keypoint_data.shape[1] < expected_cols:
                # Pad missing keypoints
                padding = np.zeros((n_persons, expected_cols - keypoint_data.shape[1]))
                keypoint_data = np.hstack([keypoint_data, padding])
            elif keypoint_data.shape[1] > expected_cols:
                # Truncate extra keypoints
                keypoint_data = keypoint_data[:, :expected_cols]

            # Reshape to [n_persons, num_keypoints, 3] for (x, y, visibility)
            keypoints_full = keypoint_data.reshape(n_persons, self.num_keypoints, 3)
            keypoints = keypoints_full[:, :, :2]
            visibilities = keypoints_full[:, :, 2].astype(np.int32)

            # Chuyển đổi bboxes thành list of tensors
            bboxes_tensor = torch.from_numpy(bboxes).float()
            bboxes_list = [bboxes_tensor]  # Wrap trong list

            return AnnotationData(
                keypoints=torch.from_numpy(keypoints).float().unsqueeze(0),
                visibilities=torch.from_numpy(visibilities).float().unsqueeze(0),
                classes=torch.from_numpy(classes).long(),
                bboxes=bboxes_list  # Trả về list thay vì tensor
            )

        except Exception as e:
            logger.error(f"Error parsing {label_path}: {str(e)}")
            return self._create_empty_annotation()

    def _create_empty_annotation(self) -> AnnotationData:
        """Create empty annotation data with correct format."""
        empty_bbox = torch.zeros(1, 4, dtype=torch.float32)
        return AnnotationData(
            keypoints=torch.zeros(1, 1, self.num_keypoints, 2, dtype=torch.float32),  # [1, P, K, 2]
            visibilities=torch.zeros(1, 1, self.num_keypoints, dtype=torch.float32),  # [1, P, K]
            classes=torch.zeros(1, dtype=torch.long),  # [P]
            bboxes=[empty_bbox]  # List containing tensor of shape [1, 4]
        )

    def _get_annotation_data(self, label_path: Path) -> AnnotationData:
        """Get annotation data with optional caching."""
        label_path_str = str(label_path)

        if self.enable_caching and label_path_str in self._annotation_cache:
            return self._annotation_cache[label_path_str]

        annotation_data = self._parse_label_file_vectorized(label_path)

        if self.enable_caching:
            self._annotation_cache[label_path_str] = annotation_data

        return annotation_data

    def _filter_valid_persons(self, ann_data: AnnotationData) -> AnnotationData:
        """Filter out persons without valid keypoint data."""
        # Find persons with visible keypoints and non-zero coordinates
        if ann_data.keypoints.dim() == 4:  # [B, P, K, 2]
            keypoints = ann_data.keypoints.squeeze(0)  # [P, K, 2]
            visibilities = ann_data.visibilities.squeeze(0)  # [P, K]
        else:
            keypoints = ann_data.keypoints  # [P, K, 2]
            visibilities = ann_data.visibilities  # [P, K]

        valid_mask = torch.zeros(keypoints.size(0), dtype=torch.bool, device=keypoints.device)

        for i in range(keypoints.size(0)):
            has_visible_kpts = visibilities[i].sum() > 0
            has_valid_coords = torch.any(keypoints[i] != 0, dim=(0, 1))
            valid_mask[i] = has_visible_kpts and has_valid_coords

        if not torch.any(valid_mask):
            return self._create_empty_annotation()

        # Xử lý bboxes dưới dạng list
        filtered_bboxes = []
        for bbox in ann_data.bboxes:
            # Áp dụng mask cho từng tensor trong list
            filtered_bbox = bbox[valid_mask] if bbox.size(0) > 1 else bbox
            filtered_bboxes.append(filtered_bbox)

        return AnnotationData(
            keypoints=keypoints[valid_mask].unsqueeze(0),
            visibilities=visibilities[valid_mask].unsqueeze(0),
            classes=ann_data.classes[valid_mask],
            bboxes=filtered_bboxes  # Giữ nguyên cấu trúc list
        )

    def _apply_transformations(self, image_data: ImageData, ann_data: AnnotationData) -> Tuple[torch.Tensor, AnnotationData]:
        """Apply augmentations and transforms."""
        image = image_data.image

        # Apply augmentations for training
        if self.augmentation and self.split == "train":
            image, keypoints, visibilities, bboxes = self.augmentation(
                image, ann_data.keypoints, ann_data.visibilities, ann_data.bboxes
            )
            ann_data = AnnotationData(keypoints, visibilities, ann_data.classes, bboxes)

        # Apply image transforms
        transformed_image = self.transform.transform(image)

        return transformed_image, ann_data

    def _generate_training_targets(self, ann_data: AnnotationData) -> torch.Tensor:
        """Generate heatmap targets for training."""
        heatmaps = generate_target_heatmap(
            keypoints=ann_data.keypoints,
            heatmap_size=self.heatmap_size,
            sigma=2.0
        )

        # Ensure consistent 4D shape [num_persons, num_keypoints, H, W]
        if heatmaps.dim() == 3:
            heatmaps = heatmaps.unsqueeze(0)

        return heatmaps

    def _validate_sample_data(self, ann_data: AnnotationData, heatmaps: torch.Tensor, img_path: Path) -> None:
        """Validate data format and shapes."""
        try:
            # Validate bboxes format
            if not isinstance(ann_data.bboxes, list):
                raise ValidationError(f"Invalid bboxes format: expected list, got {type(ann_data.bboxes)}")

            # Validate tensor shapes
            for bbox in ann_data.bboxes:
                if not isinstance(bbox, torch.Tensor):
                    raise ValidationError(f"Invalid bbox type: expected tensor, got {type(bbox)}")
                if bbox.dim() != 2 or bbox.size(1) != 4:
                    raise ValidationError(f"Invalid bbox shape: expected [N, 4], got {bbox.shape}")

            # Validate keypoints and visibilities shape
            if ann_data.keypoints.dim() != 4:  # [1, P, K, 2]
                raise ValidationError(f"Invalid keypoints shape: expected 4D, got {ann_data.keypoints.dim()}D")
            if ann_data.visibilities.dim() != 3:  # [1, P, K]
                raise ValidationError(f"Invalid visibilities shape: expected 3D, got {ann_data.visibilities.dim()}D")

        except Exception as e:
            logger.error(f"Validation error for sample {img_path}: {e}")
            raise

    def _create_sample_dict(self, image: torch.Tensor, ann_data: AnnotationData,
                       heatmaps: torch.Tensor, img_path: Path,
                       orig_size: Tuple[int, int]) -> Dict:
        """Create the final sample dictionary."""
        # Chuyển bboxes từ list sang tensor nếu cần
        bboxes = ann_data.bboxes[0] if isinstance(ann_data.bboxes, list) else ann_data.bboxes

        return {
            "image": image,
            "heatmaps": heatmaps,
            "visibilities": ann_data.visibilities,
            "bboxes": bboxes,  # Đã được chuyển thành tensor
            "keypoints": ann_data.keypoints,  # Thêm keypoints cho training
            "num_persons": ann_data.num_persons,
            "img_path": str(img_path),
            "orig_size": orig_size
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Tuple[int, int]]]:
        """Get a sample from the dataset - main orchestration method."""
        try:
            # Get file paths
            img_path, label_path = self._get_file_paths(idx)

            # Load and process image
            image_data = self._load_and_process_image(img_path)

            # Load and validate annotations
            ann_data = self._get_annotation_data(label_path)
            ann_data = self._filter_valid_persons(ann_data)

            # Limit number of persons
            if ann_data.num_persons > self.max_persons:
                logger.info(f"Truncating {ann_data.num_persons} persons to {self.max_persons} in {img_path}")
                ann_data = ann_data.truncate(self.max_persons)

            # Apply transformations
            transformed_image, ann_data = self._apply_transformations(image_data, ann_data)

            # Generate training targets
            heatmaps = self._generate_training_targets(ann_data)

            # Validate final data
            self._validate_sample_data(ann_data, heatmaps, img_path)

            return self._create_sample_dict(
                transformed_image, ann_data, heatmaps, img_path, image_data.orig_size
            )

        except Exception as e:
            logger.error(f"Error processing sample {idx} ({img_path if 'img_path' in locals() else 'unknown'}): {e}")
            raise

def efficient_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Memory-efficient collation ensuring correct output format."""
    try:
        # Filter out samples with no valid persons for training efficiency
        valid_samples = [sample for sample in batch if sample["num_persons"] > 0]

        if not valid_samples:
            # If no valid samples, return a dummy batch
            device = batch[0]["image"].device
            return {
                "image": torch.stack([sample["image"] for sample in batch]),
                "heatmaps": torch.zeros(len(batch), 1, 17, 56, 56, device=device),
                "visibilities": torch.zeros(len(batch), 1, 17, device=device),
                "bboxes": [torch.zeros(len(batch), 1, 4, device=device)],
                "num_persons": torch.zeros(len(batch), device=device, dtype=torch.long),
                "img_path": [s["img_path"] for s in batch],
                "orig_size": [s["orig_size"] for s in batch],
                "keypoints": torch.zeros(len(batch), 1, 17, 2, device=device)
            }

        device = valid_samples[0]["image"].device
        batch_images = torch.stack([sample["image"] for sample in valid_samples])
        batch_size = len(valid_samples)

        # Tìm max_persons cho mỗi batch
        max_persons = max(sample["num_persons"] for sample in valid_samples)

        # Khởi tạo tensors
        all_heatmaps = torch.zeros(batch_size, max_persons, 17, 56, 56, device=device)
        all_visibilities = torch.zeros(batch_size, max_persons, 17, device=device)
        all_bboxes = torch.zeros(batch_size, max_persons, 4, device=device)

        # Thêm keypoints cho training
        all_keypoints = torch.zeros(batch_size, max_persons, 17, 2, device=device)

        for batch_idx, sample in enumerate(valid_samples):
            num_persons = sample["num_persons"]

            if num_persons > 0:
                # Process heatmaps
                all_heatmaps[batch_idx, :num_persons] = sample["heatmaps"][:num_persons]

                # Process visibilities
                if sample["visibilities"].dim() == 3:  # [1, P, K]
                    vis = sample["visibilities"].squeeze(0)
                else:
                    vis = sample["visibilities"]
                all_visibilities[batch_idx, :num_persons] = vis[:num_persons]

                # Process bboxes
                bboxes = sample["bboxes"]
                if isinstance(bboxes, list):
                    bboxes = bboxes[0]  # Get first tensor
                if bboxes.dim() == 2:  # [P, 4]
                    all_bboxes[batch_idx, :num_persons] = bboxes[:num_persons]
                elif bboxes.dim() == 3:  # [1, P, 4]
                    all_bboxes[batch_idx, :num_persons] = bboxes.squeeze(0)[:num_persons]

                # Extract keypoints from heatmaps if available
                if "keypoints" in sample:
                    kpts = sample["keypoints"]
                    if kpts.dim() == 4:  # [1, P, K, 2]
                        kpts = kpts.squeeze(0)
                    all_keypoints[batch_idx, :num_persons] = kpts[:num_persons]

        return {
            "image": batch_images,
            "heatmaps": all_heatmaps,
            "visibilities": all_visibilities,
            "bboxes": [all_bboxes],  # Wrap trong list để phù hợp với model
            "num_persons": torch.tensor([s["num_persons"] for s in valid_samples], device=device),
            "img_path": [s["img_path"] for s in valid_samples],
            "orig_size": [s["orig_size"] for s in valid_samples],
            "keypoints": all_keypoints  # Thêm keypoints cho training
        }

    except Exception as e:
        logger.error(f"Error in efficient_collate_fn: {str(e)}")
        raise

def create_optimized_dataloader(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    split: str = "train",
    img_size: int = 224,
    grayscale: bool = True,
    num_keypoints: int = 17,
    heatmap_size: Tuple[int, int] = (56, 56),
    max_persons: int = 10,
    enable_caching: bool = True,
    cache_size: int = 1000
) -> DataLoader:
    """Create an optimized DataLoader for keypoint detection."""
    dataset = OptimizedKeypointsDataset(
        dataset_dir=dataset_dir,
        split=split,
        img_size=img_size,
        grayscale=grayscale,
        num_keypoints=num_keypoints,
        heatmap_size=heatmap_size,
        max_persons=max_persons,
        enable_caching=enable_caching,
        cache_size=cache_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        collate_fn=efficient_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )


class AdaptiveBatchSampler:
    """Advanced batch sampler that groups samples by number of persons for efficiency."""

    def __init__(self, dataset: OptimizedKeypointsDataset, batch_size: int = 32,
                 max_persons_per_batch: int = 50):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_persons_per_batch = max_persons_per_batch
        self.indices = list(range(len(dataset)))

        # Pre-compute person counts for adaptive batching
        self.person_counts = self._compute_person_counts()

    def _compute_person_counts(self) -> List[int]:
        """Pre-compute person counts for each sample."""
        person_counts = []
        logger.info("Computing person counts for adaptive batching...")

        for idx in range(len(self.dataset)):
            try:
                # Quick person count without full data loading
                _, label_path = self.dataset._get_file_paths(idx)
                ann_data = self.dataset._get_annotation_data(label_path)
                ann_data = self.dataset._filter_valid_persons(ann_data)
                person_counts.append(min(ann_data.num_persons, self.dataset.max_persons))
            except Exception as e:
                logger.warning(f"Error counting persons for sample {idx}: {e}")
                person_counts.append(1)  # Default to 1 person

        return person_counts

    def __iter__(self):
        """Generate adaptive batches based on person counts."""
        # Sort indices by person count for better batching
        sorted_indices = sorted(self.indices, key=lambda x: self.person_counts[x])

        current_batch = []
        current_person_count = 0

        for idx in sorted_indices:
            sample_person_count = self.person_counts[idx]

            # Check if adding this sample would exceed limits
            if (len(current_batch) >= self.batch_size or
                current_person_count + sample_person_count > self.max_persons_per_batch):

                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_person_count = 0

            current_batch.append(idx)
            current_person_count += sample_person_count

        # Yield remaining batch
        if current_batch:
            yield current_batch

    def __len__(self):
        """Approximate number of batches."""
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def create_adaptive_dataloader(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    split: str = "train",
    max_persons_per_batch: int = 50,
    **dataset_kwargs
) -> DataLoader:
    """Create a DataLoader with adaptive batching for optimal memory usage."""
    dataset = OptimizedKeypointsDataset(
        dataset_dir=dataset_dir,
        split=split,
        **dataset_kwargs
    )

    batch_sampler = AdaptiveBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        max_persons_per_batch=max_persons_per_batch
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=efficient_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )

