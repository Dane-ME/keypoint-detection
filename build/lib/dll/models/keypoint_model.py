"""
Multi-person Keypoint Detection Model
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from torchvision.ops import roi_align

from dll.configs.model_config import ModelConfig, BackboneConfig, KeypointHeadConfig, HeatmapHeadConfig
from dll.configs.training_config import TrainingConfig, OptimizerConfig
from dll.models.backbone import MobileNetV3Wrapper
from dll.models.heatmap_head import HeatmapHead
from dll.models.person_head import PERSON_HEAD
from dll.models.keypoint_head import KEYPOINT_HEAD
from dll.losses.keypoint_loss import KeypointLoss

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Ensure reduction preserves at least 1 channel
        reduced_channels = max(1, in_channels // reduction_ratio)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine both branches
        out = avg_out + max_out
        return torch.sigmoid(out)

class MultiPersonKeypointModel(nn.Module):
    """Multi-person keypoint detection model combining person detection and keypoint estimation."""

    def __init__(self, config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        self.training_config = training_config  # Store training config

        # Initialize components
        self.backbone = MobileNetV3Wrapper(config.backbone)
        self.person_detector = PERSON_HEAD(config.person_head)
        self.heatmap_head = HeatmapHead(config.heatmap_head)

        # Khởi tạo channel attention một lần
        self.channel_attention = ChannelAttention(
            in_channels=config.backbone.out_channels,
            reduction_ratio=16
        )

        self.loss_fn = KeypointLoss(
            num_keypoints=config.num_keypoints,
            config=training_config,
            device=torch.device('cuda')
        )

        self.num_keypoints = config.num_keypoints

    def forward(self, batch):
        """Forward pass of the model."""
        # Kiểm tra và lấy ảnh từ batch
        if isinstance(batch, dict):
            x = batch['image']  # Giả sử key của ảnh là 'image'
        else:
            x = batch

        # Đảm bảo x là tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a tensor or a dict with 'image' key containing a tensor")

        # Extract features from backbone
        all_features = self.backbone(x)
        features_full = all_features[0]
        batch_size = x.size(0)

        features = select_top_k_channels(features_full, self.channel_attention, k=64)

        # 1. Person Detection
        if isinstance(batch, dict) and 'bboxes' in batch:
            person_bboxes = batch['bboxes']
            # Xử lý format từ dataloader: List[Tensor[B, P, 4]]
            if isinstance(person_bboxes, list) and len(person_bboxes) > 0:
                bbox_tensor = person_bboxes[0]  # Lấy tensor đầu tiên từ list
                if bbox_tensor.dim() == 3:  # [B, P, 4]
                    # Chuyển thành list of tensors cho mỗi batch
                    person_bboxes = [bbox_tensor[i] for i in range(bbox_tensor.size(0))]
                elif bbox_tensor.dim() == 2:  # [P, 4] - single batch
                    person_bboxes = [bbox_tensor]
                else:
                    raise ValueError(f"Invalid bboxes tensor format: {bbox_tensor.shape}")
            elif isinstance(person_bboxes, torch.Tensor):
                # Nếu là tensor [B, P, 4], chuyển thành list of tensors
                if person_bboxes.dim() == 3 and person_bboxes.size(-1) == 4:
                    person_bboxes = [person_bboxes[i] for i in range(person_bboxes.size(0))]
                else:
                    raise ValueError(f"Invalid bboxes format: {person_bboxes.shape}")
            else:
                # Empty or invalid bboxes
                person_bboxes = [torch.zeros(0, 4, device=x.device) for _ in range(batch_size)]
        else:
            # Sử dụng person detector
            person_bboxes = self.person_detector(features, batch)
            # Đảm bảo person_bboxes là list
            if not isinstance(person_bboxes, list):
                person_bboxes = [person_bboxes]

        # 2. Keypoint Detection
        # Xử lý trường hợp không có người
        if not person_bboxes or all(len(boxes) == 0 for boxes in person_bboxes):
            if self.training:
                # Trong training mode, bỏ qua batch này
                return self._create_dummy_outputs(batch_size, x.device)
            else:
                # Trong inference mode, trả về kết quả rỗng
                empty_output = {
                    'keypoints': torch.zeros(batch_size, 1, self.num_keypoints, 2, device=x.device),
                    'visibilities': torch.zeros(batch_size, 1, self.num_keypoints, device=x.device),
                    'heatmap': torch.zeros(batch_size, 1, self.num_keypoints, 56, 56, device=x.device),
                    'boxes': person_bboxes if person_bboxes else [torch.zeros(0, 4, device=x.device) for _ in range(batch_size)]
                }
                return empty_output

        # Tìm số người tối đa trong batch
        max_persons = max(len(boxes) for boxes in person_bboxes)
        pred_kpts = []
        pred_heat = []
        pred_vis = []

        for batch_idx, boxes_in_batch in enumerate(person_bboxes):
            batch_heat = []
            batch_kpts = []
            batch_vis = []

            for box in boxes_in_batch:
                if torch.all(box == 0):
                    continue

                if box.shape[-1] != 4:
                    continue

                # Extract ROI features
                roi_features = self.extract_roi_features(features[batch_idx:batch_idx+1], box)

                # Generate heatmap
                heatmap, _ = self.heatmap_head(roi_features)  # Unpack tuple
                # Decode heatmap to get keypoints and visibility
                kpts, vis = self.decode_heatmap(heatmap)

                # Convert to original coordinates
                kpts = self.convert_to_original_coords(kpts, box)

                batch_heat.append(heatmap)
                batch_kpts.append(kpts)
                batch_vis.append(vis)

            # Pad if necessary
            if len(batch_kpts) == 0:
                # Không có người trong ảnh này
                dummy_kpts = torch.zeros(1, self.num_keypoints, 2, device=x.device)
                dummy_heat = torch.zeros(1, self.num_keypoints, 56, 56, device=x.device)
                dummy_vis = torch.zeros(1, self.num_keypoints, 3, device=x.device)  # 3-class visibility
                dummy_vis[:, :, 0] = 1.0  # Set to "not visible"
                batch_kpts.append(dummy_kpts)
                batch_heat.append(dummy_heat)
                batch_vis.append(dummy_vis)

            batch_kpts = pad_to_length(batch_kpts, max_persons)
            batch_heat = pad_to_length(batch_heat, max_persons)
            batch_vis = pad_to_length(batch_vis, max_persons)

            # Only stack if we have tensors
            if batch_kpts:
                pred_kpts.append(torch.stack(batch_kpts))
                pred_heat.append(torch.stack(batch_heat))
                pred_vis.append(torch.stack(batch_vis))
            else:
                # Create dummy tensors for empty batches
                dummy_kpts = torch.zeros(1, self.num_keypoints, 2, device=x.device)
                dummy_heat = torch.zeros(1, self.num_keypoints, 56, 56, device=x.device)
                dummy_vis = torch.zeros(1, self.num_keypoints, 3, device=x.device)
                dummy_vis[:, :, 0] = 1.0  # Set to "not visible"

                pred_kpts.append(dummy_kpts)
                pred_heat.append(dummy_heat)
                pred_vis.append(dummy_vis)

        outputs = {
            'heatmap': torch.stack(pred_heat).squeeze(2),  # Remove extra dimension
            'keypoints': torch.stack(pred_kpts),
            'visibilities': torch.stack(pred_vis),
            'boxes': person_bboxes
        }

        if isinstance(batch, dict) and 'keypoints' in batch and 'visibilities' in batch:
            outputs = self._compute_loss_and_metrics(outputs, batch)
        return outputs

    def extract_roi_features(self, features: torch.Tensor, box: torch.Tensor,
                            output_size: Tuple[int, int] = (56, 56)):
        """Extract ROI features using ROI Align.

        Args:
            features: Feature maps from backbone [1, C, H, W]
            box: Bounding box coordinates [center_x, center_y, width, height]
            output_size: Size of ROI output features

        Returns:
            ROI features [1, C, output_size[0], output_size[1]]
        """
        B, C, H, W = features.shape
        corners = box_center_to_corners(box)
        corners_pix = corners * torch.tensor([W, H, W, H], device=features.device)
        roi_boxes = torch.cat([torch.zeros(1, device=features.device), corners_pix]).unsqueeze(0)
        return roi_align(features, roi_boxes, output_size)

    def convert_to_original_coords(self, keypoints, box):
        """Convert normalized keypoint coordinates within ROI to original image coordinates."""
        original_shape = keypoints.shape
        if keypoints.dim() == 3:
            keypoints = keypoints.view(-1, 2)  # Flatten to [N*K, 2]

        # Box information (normalized)
        center_x, center_y, width, height = box  # Normalized in [0,1]
        # Convert keypoints from ROI space to image space
        keypoints_x = (keypoints[:, 0]) * width + (center_x - width / 2)
        keypoints_y = (keypoints[:, 1]) * height + (center_y - height / 2)
        # Ensure keypoints remain within [0,1]
        keypoints_x = torch.clamp(keypoints_x, 0, 1)
        keypoints_y = torch.clamp(keypoints_y, 0, 1)

        original_keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)
        # Reshape back to original shape
        original_keypoints = original_keypoints.view(original_shape)
        return original_keypoints

    def decode_heatmap(self, heatmap, threshold=0.1):
        """
        Decodes heatmap to (keypoints, visibilities) with soft-argmax for sub-pixel accuracy.
        heatmap: [B, K, H, W]
        Returns: keypoints [B, K, 2], visibilities [B, K, 3] (3-class: not visible, occluded, visible)
        """
        B, K, H, W = heatmap.shape
        device = heatmap.device

        # Use soft-argmax for sub-pixel accuracy
        keypoints = self._soft_argmax(heatmap)

        # Compute confidence scores for 3-class visibility
        flat = heatmap.view(B, K, -1)
        max_vals, _ = flat.max(dim=2)
        confidence = torch.sigmoid(max_vals)

        # Convert confidence to 3-class visibility probabilities
        # 0: not visible (conf < 0.3), 1: occluded (0.3 <= conf < 0.7), 2: visible (conf >= 0.7)
        visibilities = torch.zeros(B, K, 3, device=device)

        # Use advanced indexing to set visibility classes
        for b in range(B):
            for k in range(K):
                conf_val = confidence[b, k].item()
                if conf_val < 0.3:
                    visibilities[b, k, 0] = 1.0  # Not visible
                elif conf_val < 0.7:
                    visibilities[b, k, 1] = 1.0  # Occluded
                else:
                    visibilities[b, k, 2] = 1.0  # Visible

        return keypoints, visibilities

    def _soft_argmax(self, heatmap):
        """
        Soft-argmax for sub-pixel keypoint localization.
        Args:
            heatmap: [B, K, H, W]
        Returns:
            keypoints: [B, K, 2] normalized coordinates
        """
        B, K, H, W = heatmap.shape
        device = heatmap.device

        # Create coordinate grids
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_coords = torch.arange(H, device=device, dtype=torch.float32)

        # Normalize heatmaps with softmax
        heatmap_flat = heatmap.view(B, K, -1)
        heatmap_softmax = torch.softmax(heatmap_flat, dim=-1)
        heatmap_norm = heatmap_softmax.view(B, K, H, W)

        # Compute expected coordinates
        x_expected = torch.sum(heatmap_norm * x_coords.view(1, 1, 1, W), dim=(2, 3))
        y_expected = torch.sum(heatmap_norm * y_coords.view(1, 1, H, 1), dim=(2, 3))

        # Normalize to [0, 1]
        x_normalized = x_expected / (W - 1)
        y_normalized = y_expected / (H - 1)

        keypoints = torch.stack([x_normalized, y_normalized], dim=-1)
        return keypoints

    def _convert_keypoints_to_heatmaps(self, keypoints, visibilities, heatmap_size=(56, 56), sigma=2):
        """Convert keypoint coordinates to Gaussian heatmaps.

        Args:
            keypoints: Tensor [B, P, K, 2] of normalized coordinates
            visibilities: Tensor [B, P, K] of visibility flags
            heatmap_size: Tuple (H, W) for output heatmap size
            sigma: Gaussian sigma for heatmap generation

        Returns:
            heatmaps: Tensor [B, K, H, W]
        """
        B, P, K, _ = keypoints.shape
        H, W = heatmap_size
        device = keypoints.device

        # Initialize output heatmaps
        heatmaps = torch.zeros(B, K, H, W, device=device, dtype=torch.float32)

        # Convert normalized keypoints to pixel coordinates
        keypoints_px = keypoints.detach().clone()
        keypoints_px[..., 0] *= W
        keypoints_px[..., 1] *= H

        # Generate heatmaps using vectorized operations
        for b in range(B):
            for k in range(K):
                # Find all visible instances of this keypoint
                visible_mask = visibilities[b, :, k] > 0
                if not visible_mask.any():
                    continue

                # Get visible keypoints for this batch and keypoint
                visible_kpts = keypoints_px[b, visible_mask, k]  # [N_visible, 2]

                if len(visible_kpts) == 0:
                    continue

                # Create coordinate grids for this heatmap
                y_coords = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1)
                x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, -1)

                # Initialize heatmap for this keypoint
                max_heatmap = torch.zeros(H, W, device=device, dtype=torch.float32)

                # Generate Gaussian for each visible instance
                for kpt in visible_kpts:
                    x, y = kpt[0], kpt[1]

                    # Compute distances
                    dx = x_coords - x
                    dy = y_coords - y
                    dist_sq = dx * dx + dy * dy

                    # Generate Gaussian
                    gaussian = torch.exp(-dist_sq / (2 * sigma * sigma))

                    # Take maximum with existing heatmap
                    max_heatmap = torch.maximum(max_heatmap, gaussian)

                heatmaps[b, k] = max_heatmap

        return heatmaps

    def _create_dummy_outputs(self, batch_size, device):
        """Create dummy outputs for batches with no valid persons"""
        return {
            'keypoints': torch.zeros(batch_size, 1, self.num_keypoints, 2, device=device),
            'visibilities': torch.zeros(batch_size, 1, self.num_keypoints, device=device),
            'heatmap': torch.zeros(batch_size, 1, self.num_keypoints, 56, 56, device=device),
            'boxes': [torch.zeros(0, 4, device=device) for _ in range(batch_size)],
            'loss': torch.tensor(0.0, device=device, requires_grad=True)
        }

    def select_features(self, features_full, num_channels=64):
        batch_size = features_full.size(0)
        channel_scores = self.channel_attention(features_full)
        _, top_indices = torch.topk(
            channel_scores.view(batch_size, -1),
            k=min(num_channels, features_full.size(1)),
            dim=1
        )
        return self._gather_features(features_full, top_indices)

    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer
                   ) -> Dict[str, float]:
        """Perform single training step."""
        optimizer.zero_grad()

        outputs = self(batch)  # Changed from self(batch['image'], batch)

        # Kiểm tra xem có predictions không
        if 'loss' not in outputs:
            return {
                'loss': 0.0,
                'keypoint_loss': 0.0,
                'visibility_loss': 0.0,
                'num_detections': 0
            }

        loss = outputs['loss']
        loss.backward()
        optimizer.step()

        metrics = {
            'loss': loss.item(),
            'keypoint_loss': outputs.get('keypoint_loss', 0.0),
            'visibility_loss': outputs.get('visibility_loss', 0.0),
            'num_detections': outputs['keypoints'].size(1)
        }

        # Thêm metrics từ _calculate_metrics
        if 'keypoints' in outputs and 'keypoints' in batch:
            metrics.update(self._calculate_metrics(outputs, batch))

        return metrics

    def select_top_k_channels(features, channel_scores, k):
        """Select top k channels based on channel scores."""
        # features: [B, C, H, W], channel_scores: [B, C], k: int
        _, top_indices = torch.topk(channel_scores, k, dim=1)
        # Batch gather using torch.arange and indexing
        batch_indices = torch.arange(features.size(0))[:, None].to(features.device)
        selected_features = features[batch_indices, top_indices]
        return selected_features


    @torch.no_grad()
    def validate_step(self,
                     batch: Dict[str, torch.Tensor]
                     ) -> Dict[str, float]:
        """Perform single validation step."""
        outputs = self(batch)

        # Kiểm tra trường hợp không có detections
        if 'loss' not in outputs:
            return {
                'loss': 0.0,
                'keypoint_loss': 0.0,
                'visibility_loss': 0.0,
                'num_detections': 0,
                'avg_ADE': 0.0,
                'pck_0.002': 0.0,
                'pck_0.05': 0.0,
                'pck_0.2': 0.0
            }

        metrics = {
            'loss': outputs['loss'].item(),
            'keypoint_loss': outputs.get('keypoint_loss', 0.0),
            'visibility_loss': outputs.get('visibility_loss', 0.0),
            'num_detections': outputs['keypoints'].size(1)
        }

        if 'keypoints' in outputs and 'keypoints' in batch:
            metrics.update(self._calculate_metrics(outputs, batch))

        return metrics

    def _calculate_metrics(self,
                         outputs: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor]
                         ) -> Dict[str, float]:
        """Calculate additional metrics like PCK and ADE."""
        metrics = {}

        # Ensure matching shapes
        pred_keypoints = outputs['keypoints']
        gt_keypoints = targets['keypoints']

        # If pred_keypoints has an extra dimension, squeeze it
        if pred_keypoints.dim() == 5:  # [B, P, 1, K, 2]
            pred_keypoints = pred_keypoints.squeeze(2)  # -> [B, P, K, 2]

        # Ensure both tensors have same shape
        assert pred_keypoints.shape == gt_keypoints.shape, \
            f"Shape mismatch in metrics: pred={pred_keypoints.shape}, gt={gt_keypoints.shape}"

        # Calculate Average Distance Error (ADE)
        dist = torch.norm(
            pred_keypoints - gt_keypoints,
            dim=-1  # compute norm along last dimension (x,y coordinates)
        )
        metrics['avg_ADE'] = dist[targets['visibilities'] > 0].mean().item()

        # Calculate PCK (Percentage of Correct Keypoints)
        for threshold in self.training_config.pck_thresholds:  # Use training_config instead of config
            correct = (dist <= threshold) & (targets['visibilities'] > 0)
            metrics[f'pck_{threshold}'] = correct.float().mean().item()

        return metrics

    def _compute_loss_and_metrics(self, outputs, batch):
        """Compute loss and metrics for training"""
        heatmap_size = self.config.heatmap_head.heatmap_size

        # Reshape predicted heatmaps to [B, K, H, W]
        pred_heatmaps = outputs['heatmap']

        # Handle different heatmap shapes
        if pred_heatmaps.dim() == 5:
            # If [B, P, K, H, W], aggregate across persons (max pooling)
            pred_heatmaps = torch.max(pred_heatmaps, dim=1)[0]  # [B, K, H, W]
        elif pred_heatmaps.dim() == 4:
            # If already [B, K, H, W], use as is
            pass
        elif pred_heatmaps.dim() == 3:
            # If [B, K*H*W], reshape to [B, K, H, W]
            B = pred_heatmaps.size(0)
            pred_heatmaps = pred_heatmaps.view(B, self.num_keypoints, *heatmap_size)
        else:
            # Fallback reshape
            B = batch['image'].size(0)
            pred_heatmaps = pred_heatmaps.view(B, self.num_keypoints, *heatmap_size)

        # Use ground truth heatmaps directly from batch if available
        if 'heatmaps' in batch:
            heatmaps_gt = batch['heatmaps']
            # Handle shape mismatch: if gt has extra person dimension, aggregate
            if heatmaps_gt.dim() == 5:  # [B, P, K, H, W]
                heatmaps_gt = torch.max(heatmaps_gt, dim=1)[0]  # [B, K, H, W]
        else:
            # Use dummy heatmaps to avoid memory issues temporarily
            B = batch['image'].size(0)
            heatmaps_gt = torch.zeros(B, self.num_keypoints, *heatmap_size,
                                    device=batch['image'].device, dtype=torch.float32)

        # Prepare visibilities for loss computation
        pred_vis = outputs['visibilities']
        if pred_vis.dim() == 4:  # [B, P, K, 3]
            # Aggregate across persons (max for visibility classes)
            pred_vis = torch.max(pred_vis, dim=1)[0]  # [B, K, 3]
        elif pred_vis.dim() == 3 and pred_vis.size(-1) != 3:  # [B, P, K] old format
            # Convert to 3-class format
            pred_vis_3class = torch.zeros(pred_vis.size(0), pred_vis.size(2), 3, device=pred_vis.device)
            pred_vis_3class[:, :, 2] = torch.max(pred_vis, dim=1)[0]  # Set visible class
            pred_vis = pred_vis_3class

        gt_vis = batch['visibilities']
        if gt_vis.dim() == 3:  # [B, P, K]
            # Aggregate across persons (max for visibility)
            gt_vis = torch.max(gt_vis, dim=1)[0]  # [B, K]

        # Compute loss
        heatmap_loss, loss_dict = self.loss_fn(
            predictions={
                'heatmaps': pred_heatmaps,
                'visibilities': pred_vis,
                'keypoints': outputs['keypoints']  # Add keypoints for coordinate loss
            },
            targets={
                'heatmaps': heatmaps_gt,
                'visibility': gt_vis,
                'keypoints': batch['keypoints']  # Add target keypoints
            }
        )
        outputs['loss'] = heatmap_loss
        outputs.update(loss_dict)
        return outputs

def parse_ground_truth(file_path):
    """
    Parse ground truth data from YOLO format file.

    Args:
        file_path: Path to the ground truth file.

    Returns:
        A dictionary containing:
        - bboxes: List of tensors [B, 4] (normalized bounding boxes)
        - keypoints: Tensor [B, P, K, 2] (normalized keypoint coordinates)
        - visibilities: Tensor [B, P, K] (visibility flags)
    """
    with open(file_path, 'r') as f:
        data = f.readline().strip().split()

    # Parse bounding box and convert to list format
    bbox = torch.tensor([float(data[1]), float(data[2]),
                        float(data[3]), float(data[4])])  # [4]
    # Convert to list of tensors as expected by model
    bboxes = [bbox.unsqueeze(0)]  # List containing tensor of shape [1, 4]

    # Parse keypoints and visibilities
    keypoints = []
    visibilities = []
    num_keypoints = (len(data) - 5) // 3  # Each keypoint has x, y, visibility
    for i in range(num_keypoints):
        x = float(data[5 + i * 3])
        y = float(data[5 + i * 3 + 1])
        visibility = int(data[5 + i * 3 + 2])
        keypoints.append([x, y])
        visibilities.append(visibility)

    # Reshape to match expected formats
    keypoints = torch.tensor(keypoints).unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]
    visibilities = torch.tensor(visibilities).unsqueeze(0).unsqueeze(0)  # [1, 1, K]

    return {
        'bboxes': bboxes,  # List of tensors, each [1, 4]
        'keypoints': keypoints,  # [B=1, P=1, K, 2]
        'visibilities': visibilities  # [B=1, P=1, K]
    }

def box_center_to_corners(box):
        """Convert bounding box from center format to corner format."""
        # box: [cx, cy, w, h] normalized
        cx, cy, w, h = box.unbind()
        x1 = torch.clamp(cx - w / 2, 0, 1)
        y1 = torch.clamp(cy - h / 2, 0, 1)
        x2 = torch.clamp(cx + w / 2, 0, 1)
        y2 = torch.clamp(cy + h / 2, 0, 1)
        return torch.stack([x1, y1, x2, y2])

def pad_to_length(tensor_list, length):
    if len(tensor_list) == 0:
        # Return empty list if no tensors provided
        return []

    if len(tensor_list) >= length:
        return tensor_list[:length]

    # Create padding tensors by cloning the first tensor and zeroing it
    template = tensor_list[0]
    padding = [torch.zeros_like(template).detach() for _ in range(length - len(tensor_list))]
    return tensor_list + padding

def select_top_k_channels(features, channel_attention, k=64):
    # features: [B, C, H, W]
    batch_size, C, H, W = features.shape
    scores = channel_attention(features)
    _, topk = torch.topk(scores, min(k, C), dim=1)
    # Create batch indices for advanced indexing
    batch_idx = torch.arange(batch_size)[:, None].expand(-1, topk.size(1)).to(features.device)
    selected = features[batch_idx, topk]
    return selected

def main():
    """Test model architecture and forward pass with ground truth."""
    from dll.configs.model_config import ModelConfig, BackboneConfig, HeatmapHeadConfig
    from dll.configs.training_config import TrainingConfig, OptimizerConfig
    import torch

    # Initialize configs
    backbone_config = BackboneConfig(
        width_mult=1.0,
        in_channels=3,
        out_channels=128,
        input_size=224
    )

    heatmap_head_config = HeatmapHeadConfig(
        in_channels=64,
        hidden_channels=64,
        num_keypoints=17,
        heatmap_size=(56, 56)
    )

    model_config = ModelConfig(
        backbone=backbone_config,
        heatmap_head=heatmap_head_config,
        num_keypoints=17
    )

    training_config = TrainingConfig(
        batch_size=1,
        num_epochs=10,
        optimizer=OptimizerConfig(
            learning_rate=0.001,
            weight_decay=1e-4
        ),
        pck_thresholds=[0.05, 0.1, 0.2]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = MultiPersonKeypointModel(model_config, training_config).to(device)
    model.eval()  # Set to evaluation mode

    # Load ground truth
    ground_truth_file = "d:\\AI\\Keypoint_model\\keypoint-detection\\000000581921.txt"
    ground_truth = parse_ground_truth(ground_truth_file)

    # Create input batch
    batch = {
        'image': torch.randn(1, 3, 224, 224).to(device),  # Dummy image
        'bboxes': [bbox.to(device) for bbox in ground_truth['bboxes']],  # List of tensors
        'keypoints': ground_truth['keypoints'].to(device),
        'visibilities': ground_truth['visibilities'].to(device)
    }

    # Forward pass
    with torch.no_grad():
        outputs = model(batch)

    # Print outputs
    print("Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
