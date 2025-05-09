import torch
import torch.nn as nn
import torch.nn.functional as F

from dll.configs.model_config import PersonDetectionConfig

class PERSON_HEAD(nn.Module):
    """Head module for detecting people using fixed FPN feature map with NMS"""
    def __init__(self, config: PersonDetectionConfig):
        super(PERSON_HEAD, self).__init__()
        
        # Extract parameters from config
        in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.conf_threshold = config.conf_threshold
        self.nms_iou_threshold = config.nms_iou_threshold
        self.anchor_sizes = config.anchor_sizes
        
        # Fixed feature map size
        self.grid_size = (56, 56)
        self.aspect_ratios = [0.5, 1.0, 2.0]
        
        # Anchors for single feature map
        total_anchors = len(self.anchor_sizes) * len(self.aspect_ratios)
        
        # FPN level-specific heads
        self.box_heads = nn.ModuleList([
            nn.Conv2d(in_channels, 4 * total_anchors, kernel_size=1)
            for _ in range(4)  # Assuming 4 FPN levels
        ])
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(in_channels, self.num_classes * total_anchors, kernel_size=1)
            for _ in range(4)  # Assuming 4 FPN levels
        ])
        
        # Generate anchors once
        self.register_buffer('anchors', self._generate_anchors())
    
    def _generate_anchors(self):
        """Generate anchors for fixed (56, 56) grid"""
        h, w = self.grid_size
        anchors = []
        for i in range(h):
            for j in range(w):
                cx = (j + 0.5) / w
                cy = (i + 0.5) / h
                for size in self.anchor_sizes:
                    for aspect_ratio in self.aspect_ratios:
                        anchor_w = size * aspect_ratio
                        anchor_h = size / aspect_ratio
                        anchors.append([cx, cy, anchor_w, anchor_h])
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
            features: List of FPN feature maps
            targets: Optional dict containing ground truth information
            
        Returns:
            tuple: (person_features, person_bboxes)
        """
        # If in training mode and targets provided, use ground truth boxes
        if self.training and targets is not None and 'bboxes' in targets:
            return features[-1], targets['bboxes']  # Use the last FPN level features
        
        # Get predictions from each FPN level
        all_pred_bboxes = []
        for feat, box_head in zip(features, self.box_heads):
            pred_bboxes = box_head(feat)
            all_pred_bboxes.append(pred_bboxes)
        
        # Combine predictions from all levels
        # For simplicity, we'll just use the predictions from the last FPN level
        pred_bboxes = all_pred_bboxes[-1]
        
        return pred_bboxes
def main():
    # Test với đầu vào cụ thể
    x = torch.randn(1, 128, 28, 28)
    head_model = PERSON_HEAD(in_channels=128)
    
    try:
        output = head_model(x)
        print("Batch detections:", len(output))
        for i, det in enumerate(output):
            print(f"Batch {i}:")
            print("  Boxes shape:", det['boxes'].shape)
            print("  Scores:", det['scores'].shape)
    except Exception as e:
        print("Error occurred:", e)

if __name__ == '__main__':
    main()
