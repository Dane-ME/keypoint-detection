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
from dll.models.backbone import BACKBONE, MobileNetV3Wrapper
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
            config=training_config
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
        
        # Apply channel attention (sử dụng instance đã khởi tạo)
        channel_scores = self.channel_attention(features_full)
        channel_scores = channel_scores.view(batch_size, -1)
        
        _, top_indices = torch.topk(channel_scores, k=min(64, features_full.size(1)), dim=1)
        
        batch_indices = torch.arange(batch_size, device=features_full.device).view(-1, 1)
        
        selected_features = torch.zeros(
            batch_size, 
            min(64, features_full.size(1)), 
            features_full.size(2), 
            features_full.size(3), 
            device=features_full.device
        )
        
        # Lặp qua từng batch để chọn kênh
        for b in range(batch_size):
            for i, idx in enumerate(top_indices[b]):
                selected_features[b, i] = features_full[b, idx]
        
        # Sử dụng selected_features trực tiếp mà không cần permute
        features = selected_features

        # 1. Person Detection
        if isinstance(batch, dict) and 'bboxes' in batch:
            person_bboxes = batch['bboxes']
            # Kiểm tra và chuẩn hóa person_bboxes
            if not isinstance(person_bboxes, list):
                # Nếu là tensor [B, P, 4], chuyển thành list of tensors
                if person_bboxes.dim() == 3 and person_bboxes.size(-1) == 4:
                    person_bboxes = [boxes for boxes in person_bboxes]
                else:
                    raise ValueError(f"Invalid bboxes format: {person_bboxes.shape}")
        else:
            # Sử dụng person detector
            person_bboxes = self.person_detector(features, batch)
            # Đảm bảo person_bboxes là list
            if not isinstance(person_bboxes, list):
                person_bboxes = [person_bboxes]

        # 2. Keypoint Detection
        # Xử lý trường hợp không có người
        if not person_bboxes or all(len(boxes) == 0 for boxes in person_bboxes):
            # Trả về kết quả rỗng
            empty_output = {
                'keypoints': torch.zeros(batch_size, 0, self.num_keypoints, 2, device=x.device),
                'visibilities': torch.zeros(batch_size, 0, self.num_keypoints, device=x.device),
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
                heatmap, attention_weights = self.heatmap_head(roi_features)
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
                dummy_heat = torch.zeros(1, self.num_keypoints, 2, device=x.device)
                dummy_vis = torch.zeros(1, self.num_keypoints, device=x.device)
                batch_kpts.append(dummy_kpts)
                batch_heat.append(dummy_heat)
                batch_vis.append(dummy_vis)
            
            if len(batch_kpts) < max_persons:
                padding_needed = max_persons - len(batch_kpts)
                # Add padding with zeros
                batch_kpts.extend([torch.zeros_like(batch_kpts[0]) for _ in range(padding_needed)])
                batch_heat.extend([torch.zeros_like(batch_vis[0]) for _ in range(padding_needed)])
                batch_vis.extend([torch.zeros_like(batch_vis[0]) for _ in range(padding_needed)])
            
            pred_kpts.append(torch.stack(batch_kpts))
            pred_heat.append(torch.stack(batch_heat))
            pred_vis.append(torch.stack(batch_vis))

        outputs = {
            'heatmap': torch.stack(pred_heat).squeeze(2),  # Remove extra dimension
            'keypoints': torch.stack(pred_kpts),
            'visibilities': torch.stack(pred_vis),
            'boxes': person_bboxes
        }
        
        # Calculate loss if ground truth is provided
        if isinstance(batch, dict) and 'keypoints' in batch and 'visibilities' in batch:
            # Ensure heatmaps have correct shape before passing to loss
            heatmap_size = self.config.heatmap_head.heatmap_size  # Define heatmap_size from the model configuration
            pred_heatmaps = outputs['heatmap'].view(batch_size, -1, 224, 224)
            
            heatmap_loss, loss_dict = self.loss_fn(
                predictions={
                    'heatmaps': pred_heatmaps,  # Should be [B, K, H, W]
                    'visibilities': outputs['visibilities']  # [B, P, K]
                },
                targets={
                    'heatmaps': self._convert_keypoints_to_heatmaps(
                        keypoints=batch['keypoints'],
                        visibilities=batch['visibilities'],
                        heatmap_size=(224, 224)
                    ),
                    'visibility': batch['visibilities'].max(dim=1)[0]
                }
            )
            outputs['loss'] = heatmap_loss
            outputs.update(loss_dict)

        return outputs

    def extract_roi_features(self, features: torch.Tensor, box: torch.Tensor, 
                            output_size: Tuple[int, int] = (56, 56)) -> torch.Tensor:
        """Extract ROI features using ROI Align.
        
        Args:
            features: Feature maps from backbone [1, C, H, W]
            box: Bounding box coordinates [center_x, center_y, width, height]
            output_size: Size of ROI output features
            
        Returns:
            ROI features [1, C, output_size[0], output_size[1]]
        """
        # Ensure features is float
        features = features.float()
        
        # Ensure box is a tensor and on the same device as features
        box = torch.as_tensor(box, dtype=torch.float32, device=features.device)
        # Convert center format to corner format
        center_x, center_y, width, height = box

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        # Clamp to [0, 1] since box is in normalized coordinates
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)

        # Convert to feature map coordinates
        H, W = features.shape[2:]
        x1 = x1 * W
        y1 = y1 * H
        x2 = x2 * W
        y2 = y2 * H

        # Ensure minimum size of 1x1 to avoid zero-sized ROI
        x2 = torch.max(x2, x1 + 1)
        y2 = torch.max(y2, y1 + 1)

        # Create boxes tensor for roi_align
        boxes = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=features.device)
        
        # Apply ROI Align
        roi_feats = roi_align(
            features,
            boxes,
            output_size,
            spatial_scale=1.0
        )
        return roi_feats

    def convert_to_original_coords(self, keypoints, box):
        """Convert normalized keypoint coordinates within ROI to original image coordinates."""
        if keypoints.dim() == 3:
            keypoints = keypoints.view(-1, 2)  # Flatten to [N*17, 2]

        # Box information (normalized)
        center_x, center_y, width, height = box  # Normalized in [0,1]
        # Convert keypoints from ROI space to image space
        keypoints_x = (keypoints[:, 0]) * width + (center_x - width / 2)
        keypoints_y = (keypoints[:, 1]) * height + (center_y - height / 2)
        # Ensure keypoints remain within [0,1]
        keypoints_x = torch.clamp(keypoints_x, 0, 1)
        keypoints_y = torch.clamp(keypoints_y, 0, 1)

        original_keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)
        # Reshape back to [num_persons, 17, 2]
        original_keypoints = original_keypoints.view(-1, self.num_keypoints, 2)
        return original_keypoints

    def decode_heatmap(self, heatmap, threshold=0.1):
        """Decode heatmap to keypoint coordinates and visibility scores.
        
        Args:
            heatmap: Tensor of shape [1, K, H, W] - heatmap từ HeatmapHead
            threshold: Ngưỡng confidence để xác định keypoint visible
        
        Returns:
            keypoints: Tensor of shape [K, 2] - normalized coordinates (x, y) in [0, 1]
            visibilities: Tensor of shape [K] - visibility scores in [0, 1]
        """
        batch_size, num_keypoints, height, width = heatmap.shape
        
        # Đảm bảo chỉ xử lý một heatmap tại một thời điểm
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        # Làm phẳng heatmap để tìm max value và index
        heatmap_flat = heatmap.reshape(num_keypoints, -1)
        
        # Tìm giá trị max và index tương ứng cho mỗi keypoint
        max_values, max_indices = torch.max(heatmap_flat, dim=1)
        
        # Chuyển indices thành tọa độ x, y
        y_coords = max_indices // width
        x_coords = max_indices % width
        
        # Cải thiện độ chính xác bằng phương pháp subpixel
        # Lấy 3x3 window xung quanh điểm max
        refined_coords = []
        for k in range(num_keypoints):
            y, x = y_coords[k], x_coords[k]
            
            # Lấy window 3x3 (hoặc nhỏ hơn nếu ở biên)
            window_y_min = max(0, y - 1)
            window_y_max = min(height - 1, y + 1) + 1
            window_x_min = max(0, x - 1)
            window_x_max = min(width - 1, x + 1) + 1
            
            window = heatmap[0, k, window_y_min:window_y_max, window_x_min:window_x_max]
            
            # Tính weighted average cho subpixel accuracy
            if window.sum() > 0:
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(window_y_min, window_y_max, device=heatmap.device),
                    torch.arange(window_x_min, window_x_max, device=heatmap.device),
                    indexing='ij'
                )
                
                # Weighted average
                y_refined = (y_grid * window).sum() / window.sum()
                x_refined = (x_grid * window).sum() / window.sum()
                
                # Normalize to [0, 1]
                y_norm = y_refined / (height - 1)
                x_norm = x_refined / (width - 1)
            else:
                # Fallback to integer coordinates
                y_norm = y / (height - 1)
                x_norm = x / (width - 1)
            
            refined_coords.append(torch.tensor([x_norm, y_norm], device=heatmap.device))
        
        # Stack all keypoints
        keypoints = torch.stack(refined_coords)
        
        # Visibility scores based on heatmap max values
        visibilities = torch.sigmoid(max_values)  # Convert to [0, 1]
        
        # Apply threshold to determine visibility
        binary_visibility = (visibilities > threshold).float()
        
        return keypoints, binary_visibility

    def _heatmap_to_keypoints(self, heatmap, threshold=0.5):
        """
        Chuyển đổi heatmap thành keypoints và visibility scores.
        
        Args:
            heatmap: Tensor heatmap với shape [B, K, H, W]
            threshold: Ngưỡng để xác định visibility
        
        Returns:
            keypoints: Tensor với shape [K, 2] chứa tọa độ (x, y) đã chuẩn hóa
            visibilities: Tensor với shape [K] chứa visibility scores
        """
        # Đảm bảo heatmap có đúng số chiều
        if heatmap.dim() != 4:
            raise ValueError(f"Heatmap phải có 4 chiều [B, K, H, W], nhưng có shape {heatmap.shape}")
        
        # Lấy kích thước
        batch_size, num_keypoints, height, width = heatmap.shape
        
        # Đảm bảo batch_size = 1
        if batch_size != 1:
            raise ValueError(f"Batch size phải là 1, nhưng là {batch_size}")
        
        # Tìm vị trí có giá trị lớn nhất cho mỗi keypoint
        max_values, _ = heatmap.view(batch_size, num_keypoints, -1).max(dim=2)
        
        # Tìm vị trí có giá trị lớn nhất
        _, indices = heatmap.view(batch_size, num_keypoints, -1).max(dim=2)
        
        # Chuyển đổi indices thành tọa độ (y, x)
        y = (indices // width).float()
        x = (indices % width).float()
        
        # Tinh chỉnh tọa độ bằng cách tính weighted average trong cửa sổ 3x3
        refined_coords = []
        for k in range(num_keypoints):
            # Lấy tọa độ nguyên
            y_int = int(y[0, k].item())
            x_int = int(x[0, k].item())
            
            # Xác định cửa sổ 3x3 xung quanh điểm cực đại
            window_y_min = max(0, y_int - 1)
            window_y_max = min(height - 1, y_int + 1) + 1
            window_x_min = max(0, x_int - 1)
            window_x_max = min(width - 1, x_int + 1) + 1
            
            window = heatmap[0, k, window_y_min:window_y_max, window_x_min:window_x_max]
            
            # Tính weighted average cho subpixel accuracy
            if window.sum() > 0:
                # Kiểm tra phiên bản PyTorch để sử dụng đúng cú pháp meshgrid
                import torch
                if int(torch.__version__.split('.')[0]) < 1 or (int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) < 9):
                    # Phiên bản cũ không hỗ trợ tham số indexing
                    y_grid, x_grid = torch.meshgrid(
                        torch.arange(window_y_min, window_y_max, device=heatmap.device),
                        torch.arange(window_x_min, window_x_max, device=heatmap.device)
                    )
                else:
                    # Phiên bản mới hỗ trợ tham số indexing
                    y_grid, x_grid = torch.meshgrid(
                        torch.arange(window_y_min, window_y_max, device=heatmap.device),
                        torch.arange(window_x_min, window_x_max, device=heatmap.device),
                        indexing='ij'
                    )
                
                # Weighted average
                y_refined = (y_grid * window).sum() / window.sum()
                x_refined = (x_grid * window).sum() / window.sum()
                
                # Normalize to [0, 1]
                y_norm = y_refined / (height - 1)
                x_norm = x_refined / (width - 1)
            else:
                # Fallback to integer coordinates
                y_norm = y_int / (height - 1)
                x_norm = x_int / (width - 1)
            
            refined_coords.append(torch.tensor([x_norm, y_norm], device=heatmap.device))
        
        # Stack all keypoints
        keypoints = torch.stack(refined_coords)
        
        # Visibility scores based on heatmap max values
        visibilities = torch.sigmoid(max_values[0])  # Convert to [0, 1] and remove batch dimension
        
        # Apply threshold to determine visibility
        binary_visibility = (visibilities > threshold).float()
        
        return keypoints, binary_visibility

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
        heatmaps = torch.zeros(B, K, H, W, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # Convert normalized keypoints to pixel coordinates
        keypoints_px = torch.clone(keypoints)
        keypoints_px[..., 0] *= W
        keypoints_px[..., 1] *= H
        
        # Generate heatmaps for each batch and keypoint
        for b in range(B):
            for k in range(K):
                heatmap = torch.zeros(H, W, device=device)
                
                # Aggregate heatmaps from all visible instances of this keypoint
                for p in range(P):
                    if visibilities[b, p, k] > 0:
                        x, y = keypoints_px[b, p, k]
                        
                        # Generate Gaussian
                        dx = x_grid - x
                        dy = y_grid - y
                        dist_sq = dx * dx + dy * dy
                        exponent = dist_sq / (2 * sigma * sigma)
                        
                        # Add to heatmap with max operation
                        gaussian = torch.exp(-exponent)
                        heatmap = torch.maximum(heatmap, gaussian)
                
                heatmaps[b, k] = heatmap
        
        return heatmaps

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
