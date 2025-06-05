# Bounding Box Detection Model Guide

## Tổng quan

Tôi đã tạo một hệ thống phát hiện bounding box hoàn chỉnh cho việc detect người trong ảnh. Hệ thống bao gồm:

1. **PERSON_HEAD** - Model detection head được cải thiện
2. **BBoxDetector** - Model detection hoàn chỉnh với backbone
3. **BBoxDetectionLoss** - Loss function cho training
4. **Training & Inference Scripts** - Scripts để train và test model

## Kiến trúc Model

### 1. PERSON_HEAD (dll/models/person_head.py)
- **Chức năng**: Detection head cho việc phát hiện người
- **Kiến trúc**: YOLO-style với anchor boxes
- **Input**: Feature maps từ backbone [B, C, H, W]
- **Output**: List of bounding boxes cho mỗi batch item

**Đặc điểm chính**:
- Sử dụng 9 anchor boxes (3 sizes × 3 aspect ratios)
- Grid size: 56×56
- Có Non-Maximum Suppression (NMS)
- Confidence threshold filtering

### 2. BBoxDetector (dll/models/bbox_detector.py)
- **Chức năng**: Model detection end-to-end
- **Kiến trúc**: MobileNetV3 backbone + YOLO detection head
- **Input**: RGB images [B, 3, 224, 224]
- **Output**: Detected bounding boxes

**Đặc điểm chính**:
- Backbone: MobileNetV3 để trích xuất features
- Detection Head: YOLO-style với multiple layers
- Anchor-based detection
- Built-in NMS và confidence filtering

### 3. BBoxDetectionLoss (dll/losses/bbox_loss.py)
- **Chức năng**: Loss function cho training detection model
- **Kiến trúc**: YOLO-style loss với multiple components

**Loss components**:
- **Coordinate Loss**: MSE loss cho bbox regression
- **Objectness Loss**: BCE loss cho object presence
- **No-object Loss**: BCE loss cho background
- **Classification Loss**: BCE loss cho multi-class (optional)

## Cách sử dụng

### 1. Training Model

```bash
# Train bounding box detection model
python scripts/train_bbox.py --config configs/default_config.yaml
```

**Tham số training**:
- `--config`: Path đến config file
- `--resume`: Path đến checkpoint để resume training
- `--output_dir`: Thư mục output (override config)

### 2. Inference

```bash
# Predict single image
python scripts/predict_bbox.py \
    --config configs/default_config.yaml \
    --model outputs/bbox_detection/best_model.pth \
    --input path/to/image.jpg \
    --output results/

# Predict batch images
python scripts/predict_bbox.py \
    --config configs/default_config.yaml \
    --model outputs/bbox_detection/best_model.pth \
    --input path/to/images/ \
    --output results/
```

### 3. Test Models

```bash
# Test PERSON_HEAD
python dll/models/person_head.py

# Test BBoxDetector
python dll/models/bbox_detector.py

# Test Loss function
python dll/losses/bbox_loss.py
```

## Configuration

Model được config thông qua file YAML. Các tham số quan trọng:

```yaml
model:
  backbone:
    width_mult: 1.0
    in_channels: 3
    out_channels: 128
    input_size: 224
    
  person_head:
    in_channels: 128
    num_classes: 1
    conf_threshold: 0.3
    nms_iou_threshold: 0.3
    anchor_sizes: [32, 64, 128]

training:
  batch_size: 8
  num_epochs: 100
  optimizer:
    name: "adam"
    learning_rate: 0.001
    weight_decay: 0.0001
```

## Kết quả Output

### Training Output
- **Checkpoints**: `outputs/bbox_detection/`
- **Logs**: `outputs/bbox_detection/training.log`
- **Best model**: `outputs/bbox_detection/best_model.pth`

### Inference Output
- **Visualizations**: Images với bounding boxes được vẽ
- **Coordinates**: Bounding boxes trong format [x1, y1, x2, y2]

## Đặc điểm kỹ thuật

### Anchor Configuration
- **Sizes**: [32, 64, 128] pixels
- **Aspect ratios**: [0.5, 1.0, 2.0]
- **Total anchors**: 9 per grid cell

### Grid Configuration
- **Input size**: 224×224
- **Feature map size**: 56×56 (downsampled 4x)
- **Grid cells**: 56×56 = 3136 cells

### Detection Parameters
- **Confidence threshold**: 0.3 (configurable)
- **NMS IoU threshold**: 0.3 (configurable)
- **Max detections**: 100 per image

## Performance Tips

### Training
1. **Batch size**: Tăng batch size nếu có đủ GPU memory
2. **Learning rate**: Bắt đầu với 0.001, giảm nếu loss không giảm
3. **Data augmentation**: Enable để tăng độ robust
4. **Mixed precision**: Enable để tăng tốc training

### Inference
1. **Confidence threshold**: Tăng để giảm false positives
2. **NMS threshold**: Giảm để loại bỏ overlapping boxes
3. **Batch inference**: Process nhiều images cùng lúc

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Giảm batch size
   - Enable gradient checkpointing
   - Sử dụng mixed precision

2. **Low detection accuracy**:
   - Kiểm tra confidence threshold
   - Tăng số epochs training
   - Kiểm tra data quality

3. **Too many false positives**:
   - Tăng confidence threshold
   - Điều chỉnh NMS threshold
   - Cải thiện training data

## Mở rộng

### Thêm classes mới
1. Cập nhật `num_classes` trong config
2. Chuẩn bị training data với labels
3. Cập nhật loss function nếu cần

### Cải thiện accuracy
1. Sử dụng backbone mạnh hơn (ResNet, EfficientNet)
2. Thêm Feature Pyramid Network (FPN)
3. Sử dụng advanced augmentation
4. Implement focal loss cho class imbalance

### Tối ưu tốc độ
1. Model quantization
2. TensorRT optimization
3. ONNX export
4. Mobile deployment với TorchScript

## Kết luận

Hệ thống bounding box detection này cung cấp:
- ✅ Model detection hoàn chỉnh và hiệu quả
- ✅ Training pipeline với loss function phù hợp
- ✅ Inference scripts với visualization
- ✅ Cấu hình linh hoạt qua YAML
- ✅ Code được tổ chức tốt và dễ mở rộng

Model có thể được sử dụng độc lập hoặc tích hợp vào hệ thống keypoint detection lớn hơn.
