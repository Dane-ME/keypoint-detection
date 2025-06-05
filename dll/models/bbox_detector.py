"""
Complete Bounding Box Detection Model
Standalone YOLO-style object detection model for person detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math

from dll.configs.model_config import PersonDetectionConfig
from dll.models.backbone import MobileNetV3Wrapper


class BBoxDetector(nn.Module):
    """
    Complete YOLO-style bounding box detector
    Combines backbone + detection head for end-to-end person detection
    """
    
    def __init__(self, backbone_config, detection_config: PersonDetectionConfig):
        super().__init__()
        
        self.backbone = MobileNetV3Wrapper(backbone_config)
        self.detection_head = YOLODetectionHead(detection_config)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, targets=None):
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            targets: Optional ground truth for training
            
        Returns:
            List of detected bounding boxes for each batch item
        """
        # Extract features
        features = self.backbone(x)
        
        # Use the last feature map for detection
        if isinstance(features, list):
            detection_features = features[-1]
        else:
            detection_features = features
        
        # Detect objects
        detections = self.detection_head(detection_features, targets)
        
        return detections


class YOLODetectionHead(nn.Module):
    """
    YOLO-style detection head for bounding box prediction
    """
    
    def __init__(self, config: PersonDetectionConfig):
        super().__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        self.conf_threshold = config.conf_threshold
        self.nms_iou_threshold = config.nms_iou_threshold
        
        # Anchor configuration
        self.anchor_sizes = config.anchor_sizes  # [32, 64, 128]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.anchor_sizes) * len(self.aspect_ratios)
        
        # Network layers
        in_channels = config.in_channels
        
        # Feature processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
        )
        
        # Output heads
        # Each anchor predicts: [x, y, w, h, objectness, class_scores...]
        self.output_channels = self.num_anchors * (4 + 1 + self.num_classes)
        self.output_conv = nn.Conv2d(in_channels // 8, self.output_channels, 1)
        
        # Generate and register anchors
        self.register_buffer('anchors', self._generate_anchors())
    
    def _generate_anchors(self):
        """Generate anchor boxes for each grid cell"""
        anchors = []
        for size in self.anchor_sizes:
            for ratio in self.aspect_ratios:
                w = size * math.sqrt(ratio)
                h = size / math.sqrt(ratio)
                # Normalize by image size (assuming 224x224 input -> 56x56 feature map)
                w_norm = w / 224.0
                h_norm = h / 224.0
                anchors.append([w_norm, h_norm])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, features, targets=None):
        """
        Forward pass of detection head
        
        Args:
            features: Feature maps [B, C, H, W]
            targets: Optional ground truth for training
            
        Returns:
            List of detected bounding boxes for each batch item
        """
        # If training with ground truth, return GT boxes
        if self.training and targets is not None and 'bboxes' in targets:
            return targets['bboxes']
        
        # Forward through conv layers
        x = self.conv_layers(features)
        predictions = self.output_conv(x)
        
        # Reshape predictions
        B, _, H, W = predictions.shape
        predictions = predictions.view(B, self.num_anchors, -1, H, W)
        predictions = predictions.permute(0, 3, 4, 1, 2)  # [B, H, W, num_anchors, output_dim]
        
        # Decode predictions to bounding boxes
        detections = self._decode_predictions(predictions, H, W)
        
        return detections
    
    def _decode_predictions(self, predictions, grid_h, grid_w):
        """
        Decode raw predictions to bounding boxes
        
        Args:
            predictions: [B, H, W, num_anchors, (4+1+num_classes)]
            grid_h, grid_w: Grid dimensions
            
        Returns:
            List of detected boxes for each batch item
        """
        B, H, W, num_anchors, _ = predictions.shape
        device = predictions.device
        
        # Split predictions
        bbox_pred = predictions[..., :4]  # [B, H, W, num_anchors, 4]
        objectness = torch.sigmoid(predictions[..., 4])  # [B, H, W, num_anchors]
        
        if self.num_classes > 1:
            class_pred = torch.sigmoid(predictions[..., 5:])  # [B, H, W, num_anchors, num_classes]
        else:
            class_pred = torch.ones_like(objectness.unsqueeze(-1))  # [B, H, W, num_anchors, 1]
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Decode bounding box coordinates
        # Center coordinates: sigmoid + grid offset, normalized by grid size
        pred_x = (torch.sigmoid(bbox_pred[..., 0]) + grid_x.unsqueeze(-1)) / W
        pred_y = (torch.sigmoid(bbox_pred[..., 1]) + grid_y.unsqueeze(-1)) / H
        
        # Width and height: exp * anchor size
        anchor_w = self.anchors[:, 0].view(1, 1, 1, num_anchors)
        anchor_h = self.anchors[:, 1].view(1, 1, 1, num_anchors)
        
        pred_w = torch.exp(bbox_pred[..., 2]) * anchor_w
        pred_h = torch.exp(bbox_pred[..., 3]) * anchor_h
        
        # Clamp to reasonable values
        pred_w = torch.clamp(pred_w, max=1.0)
        pred_h = torch.clamp(pred_h, max=1.0)
        
        # Combine predictions
        decoded_boxes = torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1)
        
        # Process each batch item
        batch_detections = []
        for b in range(B):
            # Flatten spatial and anchor dimensions
            boxes_flat = decoded_boxes[b].reshape(-1, 4)  # [H*W*num_anchors, 4]
            obj_flat = objectness[b].reshape(-1)  # [H*W*num_anchors]

            # Compute final confidence scores
            if self.num_classes > 1:
                class_flat = class_pred[b].reshape(-1, self.num_classes)  # [H*W*num_anchors, num_classes]
                max_class_scores, _ = class_flat.max(dim=1)
                final_scores = obj_flat * max_class_scores
            else:
                final_scores = obj_flat
            
            # Filter by confidence threshold
            valid_mask = final_scores > self.conf_threshold
            if not valid_mask.any():
                batch_detections.append(torch.zeros(0, 4, device=device))
                continue
            
            valid_boxes = boxes_flat[valid_mask]
            valid_scores = final_scores[valid_mask]
            
            # Apply Non-Maximum Suppression
            if len(valid_boxes) > 0:
                keep_indices = self._nms(valid_boxes, valid_scores)
                final_boxes = valid_boxes[keep_indices]
            else:
                final_boxes = torch.zeros(0, 4, device=device)
            
            batch_detections.append(final_boxes)
        
        return batch_detections
    
    def _nms(self, boxes, scores, max_output_size=100):
        """
        Non-Maximum Suppression
        
        Args:
            boxes: [N, 4] tensor of boxes in format [cx, cy, w, h]
            scores: [N] tensor of confidence scores
            max_output_size: Maximum number of boxes to keep
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Convert to corner format for IoU computation
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0 and len(keep) < max_output_size:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0]
            keep.append(i.item())
            
            # Compute IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
            
            # Keep boxes with IoU less than threshold
            mask = iou <= self.nms_iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def main():
    """Test the BBoxDetector model"""
    from dll.configs.model_config import BackboneConfig, PersonDetectionConfig
    
    # Create configs
    backbone_config = BackboneConfig(
        width_mult=1.0,
        in_channels=3,
        out_channels=128,
        input_size=224
    )
    
    detection_config = PersonDetectionConfig(
        in_channels=128,
        num_classes=1,
        conf_threshold=0.3,
        nms_iou_threshold=0.3,
        anchor_sizes=[32, 64, 128]
    )
    
    # Create model
    model = BBoxDetector(backbone_config, detection_config)
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 RGB images
    
    print("Testing BBoxDetector model...")
    print(f"Input shape: {x.shape}")
    
    try:
        with torch.no_grad():
            detections = model(x)
        
        print(f"\nDetections for {len(detections)} batch items:")
        for i, det in enumerate(detections):
            print(f"  Batch {i}: {det.shape[0]} detections")
            if det.shape[0] > 0:
                print(f"    Sample detection (cx, cy, w, h): {det[0]}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
