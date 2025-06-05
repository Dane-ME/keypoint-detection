"""
Bounding Box Detection Loss Functions
Implements YOLO-style loss for object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math


class BBoxDetectionLoss(nn.Module):
    """
    YOLO-style loss for bounding box detection
    Combines objectness loss, bbox regression loss, and classification loss
    """
    
    def __init__(self, 
                 num_classes: int = 1,
                 lambda_coord: float = 5.0,
                 lambda_noobj: float = 0.5,
                 lambda_obj: float = 1.0,
                 lambda_cls: float = 1.0,
                 anchor_sizes: List[int] = [32, 64, 128],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 grid_size: Tuple[int, int] = (56, 56)):
        """
        Initialize YOLO loss
        
        Args:
            num_classes: Number of object classes
            lambda_coord: Weight for coordinate loss
            lambda_noobj: Weight for no-object loss
            lambda_obj: Weight for objectness loss
            lambda_cls: Weight for classification loss
            anchor_sizes: Anchor box sizes
            aspect_ratios: Anchor box aspect ratios
            grid_size: Grid size for detection
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(anchor_sizes) * len(aspect_ratios)
        self.grid_size = grid_size
        
        # Generate anchors
        self.register_buffer('anchors', self._generate_anchors())
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
    
    def _generate_anchors(self):
        """Generate anchor boxes"""
        anchors = []
        for size in self.anchor_sizes:
            for ratio in self.aspect_ratios:
                w = size * math.sqrt(ratio)
                h = size / math.sqrt(ratio)
                # Normalize by image size (assuming 224x224 input)
                w_norm = w / 224.0
                h_norm = h / 224.0
                anchors.append([w_norm, h_norm])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO loss
        
        Args:
            predictions: Model predictions [B, H, W, num_anchors, (4+1+num_classes)]
            targets: Ground truth containing 'bboxes', 'labels' (optional)
            
        Returns:
            Dictionary containing loss components
        """
        B, H, W, num_anchors, _ = predictions.shape
        device = predictions.device
        
        # Split predictions
        bbox_pred = predictions[..., :4]  # [B, H, W, num_anchors, 4]
        obj_pred = predictions[..., 4]    # [B, H, W, num_anchors]
        
        if self.num_classes > 1:
            cls_pred = predictions[..., 5:]  # [B, H, W, num_anchors, num_classes]
        else:
            cls_pred = None
        
        # Initialize loss components
        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        # Process each batch item
        total_positive_samples = 0
        total_negative_samples = 0
        
        for b in range(B):
            # Get ground truth boxes for this batch item
            if isinstance(targets['bboxes'], list):
                gt_boxes = targets['bboxes'][b]  # [N, 4] format: [cx, cy, w, h]
            else:
                gt_boxes = targets['bboxes'][b]
            
            if len(gt_boxes) == 0:
                # No objects in this image - all predictions should be background
                noobj_mask = torch.ones(H, W, num_anchors, device=device, dtype=torch.bool)
                noobj_loss += self.bce_loss(
                    torch.sigmoid(obj_pred[b][noobj_mask]),
                    torch.zeros_like(obj_pred[b][noobj_mask])
                )
                total_negative_samples += noobj_mask.sum().item()
                continue
            
            # Create target tensors
            obj_target = torch.zeros(H, W, num_anchors, device=device)
            bbox_target = torch.zeros(H, W, num_anchors, 4, device=device)
            responsible_mask = torch.zeros(H, W, num_anchors, device=device, dtype=torch.bool)
            
            # Assign ground truth to grid cells and anchors
            for gt_box in gt_boxes:
                if torch.all(gt_box == 0):  # Skip invalid boxes
                    continue
                
                cx, cy, w, h = gt_box
                
                # Find grid cell
                grid_x = int(cx * W)
                grid_y = int(cy * H)
                
                # Clamp to valid range
                grid_x = max(0, min(W - 1, grid_x))
                grid_y = max(0, min(H - 1, grid_y))
                
                # Find best anchor based on IoU with ground truth
                gt_area = w * h
                best_anchor_idx = 0
                best_iou = 0
                
                for anchor_idx in range(num_anchors):
                    anchor_w, anchor_h = self.anchors[anchor_idx]
                    anchor_area = anchor_w * anchor_h
                    
                    # Compute IoU between GT box and anchor
                    inter_w = min(w, anchor_w)
                    inter_h = min(h, anchor_h)
                    inter_area = inter_w * inter_h
                    union_area = gt_area + anchor_area - inter_area
                    
                    iou = inter_area / (union_area + 1e-16)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor_idx = anchor_idx
                
                # Assign targets
                responsible_mask[grid_y, grid_x, best_anchor_idx] = True
                obj_target[grid_y, grid_x, best_anchor_idx] = 1.0
                
                # Encode bbox targets (relative to grid cell and anchor)
                bbox_target[grid_y, grid_x, best_anchor_idx, 0] = cx * W - grid_x  # x offset
                bbox_target[grid_y, grid_x, best_anchor_idx, 1] = cy * H - grid_y  # y offset
                bbox_target[grid_y, grid_x, best_anchor_idx, 2] = torch.log(w / self.anchors[best_anchor_idx, 0] + 1e-16)  # w
                bbox_target[grid_y, grid_x, best_anchor_idx, 3] = torch.log(h / self.anchors[best_anchor_idx, 1] + 1e-16)  # h
            
            # Compute losses for this batch item
            
            # 1. Coordinate loss (only for responsible predictions)
            if responsible_mask.any():
                coord_loss += self.lambda_coord * self.mse_loss(
                    bbox_pred[b][responsible_mask],
                    bbox_target[responsible_mask]
                )
                total_positive_samples += responsible_mask.sum().item()
            
            # 2. Objectness loss (for responsible predictions)
            if responsible_mask.any():
                obj_loss += self.lambda_obj * self.bce_loss(
                    torch.sigmoid(obj_pred[b][responsible_mask]),
                    obj_target[responsible_mask]
                )
            
            # 3. No-object loss (for non-responsible predictions)
            noobj_mask = ~responsible_mask
            if noobj_mask.any():
                noobj_loss += self.lambda_noobj * self.bce_loss(
                    torch.sigmoid(obj_pred[b][noobj_mask]),
                    torch.zeros_like(obj_pred[b][noobj_mask])
                )
                total_negative_samples += noobj_mask.sum().item()
            
            # 4. Classification loss (if multi-class)
            if self.num_classes > 1 and cls_pred is not None:
                # For simplicity, assume all objects are class 0 (person)
                if responsible_mask.any():
                    cls_target = torch.zeros(responsible_mask.sum(), self.num_classes, device=device)
                    cls_target[:, 0] = 1.0  # All objects are persons
                    
                    cls_loss += self.lambda_cls * self.bce_loss(
                        torch.sigmoid(cls_pred[b][responsible_mask]),
                        cls_target
                    )
        
        # Normalize losses
        if total_positive_samples > 0:
            coord_loss /= total_positive_samples
            obj_loss /= total_positive_samples
        
        if total_negative_samples > 0:
            noobj_loss /= total_negative_samples
        
        if total_positive_samples > 0 and self.num_classes > 1:
            cls_loss /= total_positive_samples
        
        # Total loss
        total_loss = coord_loss + obj_loss + noobj_loss + cls_loss
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'cls_loss': cls_loss,
            'num_positive': total_positive_samples,
            'num_negative': total_negative_samples
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions [N, C] or [N]
            targets: Ground truth [N, C] or [N]
            
        Returns:
            Focal loss
        """
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    """Test the BBoxDetectionLoss"""
    
    # Create loss function
    loss_fn = BBoxDetectionLoss(
        num_classes=1,
        lambda_coord=5.0,
        lambda_noobj=0.5,
        lambda_obj=1.0,
        grid_size=(56, 56)
    )
    
    # Create dummy predictions and targets
    B, H, W = 2, 56, 56
    num_anchors = 9  # 3 sizes * 3 ratios
    
    # Dummy predictions [B, H, W, num_anchors, (4+1+1)]
    predictions = torch.randn(B, H, W, num_anchors, 6)
    
    # Dummy targets
    targets = {
        'bboxes': [
            torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.2, 0.3, 0.1, 0.2]]),  # Batch 0: 2 boxes
            torch.tensor([[0.7, 0.8, 0.2, 0.3]])  # Batch 1: 1 box
        ]
    }
    
    print("Testing BBoxDetectionLoss...")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of target boxes: {[len(boxes) for boxes in targets['bboxes']]}")
    
    try:
        loss_dict = loss_fn(predictions, targets)
        
        print("\nLoss components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
