import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from dll.configs.model_config import PersonDetectionConfig

class PERSON_HEAD(nn.Module):
    """Improved Head module for detecting people with YOLO-style detection"""
    def __init__(self, config: PersonDetectionConfig):
        super(PERSON_HEAD, self).__init__()

        # Extract parameters from config
        in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.conf_threshold = config.conf_threshold
        self.nms_iou_threshold = config.nms_iou_threshold
        self.anchor_sizes = config.anchor_sizes

        # Fixed feature map size for detection
        self.grid_size = (56, 56)
        self.aspect_ratios = [0.5, 1.0, 2.0]

        # Number of anchors per grid cell
        self.num_anchors = len(self.anchor_sizes) * len(self.aspect_ratios)

        # Detection head layers
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)

        # Output layers: bbox regression (4) + objectness (1) + class scores (num_classes)
        self.output_channels = self.num_anchors * (4 + 1 + self.num_classes)
        self.output_conv = nn.Conv2d(in_channels // 4, self.output_channels, kernel_size=1)

        # Batch normalization and activation
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        # Generate anchors once
        self.register_buffer('anchors', self._generate_anchors())
    
    def _generate_anchors(self):
        """Generate anchors for each anchor type (not per grid cell)"""
        anchors = []
        for size in self.anchor_sizes:
            for aspect_ratio in self.aspect_ratios:
                # Normalize anchor sizes by image size (assuming 224x224 input)
                anchor_w = (size * aspect_ratio) / 224.0
                anchor_h = (size / aspect_ratio) / 224.0
                anchors.append([anchor_w, anchor_h])
        return torch.tensor(anchors, dtype=torch.float32)
    
    def box_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes
        
        Args:
        - boxes1: (N, 4) tensor of [cx, cy, w, h]
        - boxes2: (M, 4) tensor of [cx, cy, w, h]
        
        Returns:
        - IoU matrix of shape (N, M)
        """
        # Convert center format to corner format
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        # Intersection coordinates
        inter_x1 = torch.max(b1_x1[:, None], b2_x1)
        inter_y1 = torch.max(b1_y1[:, None], b2_y1)
        inter_x2 = torch.min(b1_x2[:, None], b2_x2)
        inter_y2 = torch.min(b1_y2[:, None], b2_y2)
        
        # Intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union_area = b1_area[:, None] + b2_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-16)
        return iou
    
    def non_max_suppression(self, boxes, scores, iou_threshold=0.2, max_output_size=None):
        """
        Perform Non-Maximum Suppression
        
        Args:
        - boxes: (N, 4) tensor of boxes [cx, cy, w, h]
        - scores: (N,) tensor of confidence scores
        - iou_threshold: IoU threshold for suppression
        - max_output_size: Maximum number of boxes to keep
        
        Returns:
        - Indices of boxes to keep
        """
        # Sort boxes by scores
        _, order = scores.sort(0, descending=True)
        
        # Initialize list to keep track of kept boxes
        keep = []
        
        while order.numel() > 0:
            # If only one box left, keep it
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            # Keep the box with highest score
            i = order[0]
            keep.append(i.item())
            
            # If max output size is reached, stop
            if max_output_size and len(keep) >= max_output_size:
                break
            
            # Remove the top box
            order = order[1:]
            
            # Compute IoU of the top box with remaining boxes
            iou = self.box_iou(boxes[i].unsqueeze(0), boxes[order])
            
            # Keep boxes with low IoU
            mask = iou <= iou_threshold
            order = order[mask.squeeze()]
        
        return torch.tensor(keep)
    
    def forward(self, features, targets=None):
        """
        Forward pass of person detection head

        Args:
            features: Feature maps from backbone [B, C, H, W] or list of feature maps
            targets: Optional dict containing ground truth information

        Returns:
            List of detected bounding boxes for each batch item
        """
        # Handle both single feature map and list of feature maps
        if isinstance(features, list):
            x = features[-1]  # Use the last feature map if multiple provided
        else:
            x = features

        # If in training mode and targets provided, use ground truth boxes
        if self.training and targets is not None and 'bboxes' in targets:
            return targets['bboxes']

        # Forward through detection head
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        predictions = self.output_conv(x)

        # Reshape predictions: [B, num_anchors * (4+1+num_classes), H, W] -> [B, H, W, num_anchors, (4+1+num_classes)]
        B, _, H, W = predictions.shape
        predictions = predictions.view(B, self.num_anchors, -1, H, W).permute(0, 3, 4, 1, 2)

        # Decode predictions to get bounding boxes
        detected_boxes = self._decode_predictions(predictions)

        return detected_boxes

    def _decode_predictions(self, predictions):
        """
        Decode raw predictions to bounding boxes

        Args:
            predictions: [B, H, W, num_anchors, (4+1+num_classes)]

        Returns:
            List of detected boxes for each batch item
        """
        B, H, W, num_anchors, _ = predictions.shape
        device = predictions.device

        # Split predictions
        bbox_pred = predictions[..., :4]  # [B, H, W, num_anchors, 4]
        objectness = predictions[..., 4]  # [B, H, W, num_anchors]
        class_pred = predictions[..., 5:]  # [B, H, W, num_anchors, num_classes]

        # Apply sigmoid to objectness and class predictions
        objectness = torch.sigmoid(objectness)
        class_scores = torch.sigmoid(class_pred)

        # Create grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Expand grid for all anchors
        grid_x = grid_x.unsqueeze(-1).expand(H, W, num_anchors)  # [H, W, num_anchors]
        grid_y = grid_y.unsqueeze(-1).expand(H, W, num_anchors)  # [H, W, num_anchors]

        # Get anchor dimensions
        anchor_w = self.anchors[:, 0].view(1, 1, num_anchors)  # [1, 1, num_anchors]
        anchor_h = self.anchors[:, 1].view(1, 1, num_anchors)  # [1, 1, num_anchors]

        # Decode bounding boxes
        # Apply sigmoid to center coordinates and scale by grid
        pred_x = (torch.sigmoid(bbox_pred[..., 0]) + grid_x) / W
        pred_y = (torch.sigmoid(bbox_pred[..., 1]) + grid_y) / H

        # Apply exponential to width and height and scale by anchors
        pred_w = torch.exp(bbox_pred[..., 2]) * anchor_w
        pred_h = torch.exp(bbox_pred[..., 3]) * anchor_h

        # Combine into boxes [B, H, W, num_anchors, 4]
        decoded_boxes = torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1)

        # Process each batch item
        batch_detections = []
        for b in range(B):
            # Flatten spatial and anchor dimensions
            boxes_flat = decoded_boxes[b].reshape(-1, 4)  # [H*W*num_anchors, 4]
            obj_flat = objectness[b].reshape(-1)  # [H*W*num_anchors]

            # Get class scores (take max across classes for simplicity)
            if self.num_classes > 1:
                class_scores_flat = class_scores[b].reshape(-1, self.num_classes)  # [H*W*num_anchors, num_classes]
                max_class_scores, _ = class_scores_flat.max(dim=1)
                final_scores = obj_flat * max_class_scores
            else:
                final_scores = obj_flat

            # Filter by confidence threshold
            valid_mask = final_scores > self.conf_threshold
            if not valid_mask.any():
                # No valid detections
                batch_detections.append(torch.zeros(0, 4, device=device))
                continue

            valid_boxes = boxes_flat[valid_mask]
            valid_scores = final_scores[valid_mask]

            # Apply NMS
            if len(valid_boxes) > 0:
                keep_indices = self.non_max_suppression(
                    valid_boxes, valid_scores,
                    iou_threshold=self.nms_iou_threshold,
                    max_output_size=100  # Maximum 100 detections per image
                )
                final_boxes = valid_boxes[keep_indices]
            else:
                final_boxes = torch.zeros(0, 4, device=device)

            batch_detections.append(final_boxes)

        return batch_detections

def main():
    """Test the improved PERSON_HEAD model"""
    from dll.configs.model_config import PersonDetectionConfig

    # Create config
    config = PersonDetectionConfig(
        in_channels=128,
        num_classes=1,
        conf_threshold=0.3,
        nms_iou_threshold=0.3,
        anchor_sizes=[32, 64, 128]
    )

    # Create model
    model = PERSON_HEAD(config)
    model.eval()

    # Test input
    x = torch.randn(2, 128, 56, 56)  # Batch size 2, 128 channels, 56x56 feature map

    print("Testing PERSON_HEAD model...")
    print(f"Input shape: {x.shape}")
    print(f"Number of anchors per grid cell: {model.num_anchors}")
    print(f"Grid size: {model.grid_size}")

    try:
        with torch.no_grad():
            detections = model(x)

        print(f"\nDetections for {len(detections)} batch items:")
        for i, det in enumerate(detections):
            print(f"  Batch {i}: {det.shape[0]} detections, shape: {det.shape}")
            if det.shape[0] > 0:
                print(f"    Sample detection: {det[0]}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
