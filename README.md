# Multi-Person Keypoint Detection

Thư viện deep learning cho phát hiện keypoint đa người sử dụng PyTorch, với kiến trúc MobileNetV3 tùy chỉnh và hệ thống cân bằng loss function tiên tiến.

## Tổng quan dự án

Dự án này là một hệ thống AI hoàn chỉnh để phát hiện và định vị các điểm khớp (keypoints) trên cơ thể người trong ảnh. Hệ thống có thể xử lý nhiều người trong cùng một ảnh và xác định 17 điểm khớp chính trên cơ thể mỗi người.

## Tính năng chính

- **Phát hiện keypoint đa người**: Xác định 17 điểm khớp cho mỗi người trong ảnh
- **Backbone MobileNetV3 tối ưu**: Được tùy chỉnh cho thiết bị di động
- **Hệ thống loss function tiên tiến**:
  - Dynamic loss balancing - giải quyết vấn đề PCK metrics đình trệ
  - Spatial-aware coordinate loss - tăng cường trọng số lỗi pixel-space
  - Adaptive heatmap loss - trọng số theo vùng để định vị keypoint tốt hơn
- **Person detection với NMS**: Phát hiện người và loại bỏ duplicate
- **Hệ thống cấu hình linh hoạt**: Sử dụng file YAML
- **Metrics đánh giá toàn diện**: PCK, ADE và các chỉ số khác
- **Quản lý thiết bị mạnh mẽ**: Hỗ trợ CPU/GPU với error handling

## Cấu trúc dự án chi tiết

```
keypoint-detection/
├── dll/                           # Package chính
│   ├── __init__.py               # Main exports
│   ├── configs/                  # Hệ thống cấu hình
│   │   ├── base_config.py       # Base configuration classes
│   │   ├── model_config.py      # Model architecture config
│   │   ├── training_config.py   # Training hyperparameters
│   │   ├── dataset_config.py    # Dataset configuration
│   │   └── config_loader.py     # YAML config loader
│   ├── data/                     # Data processing pipeline
│   │   ├── dataloader.py        # Optimized dataset & dataloader
│   │   ├── augmentation.py      # Keypoint-aware augmentations
│   │   └── transforms.py        # Image transformations
│   ├── losses/                   # Advanced loss functions
│   │   └── keypoint_loss.py     # Dynamic loss balancing
│   ├── models/                   # Neural network architecture
│   │   ├── keypoint_model.py    # Main multi-person model
│   │   ├── backbone.py          # MobileNetV3 backbone
│   │   ├── person_head.py       # Person detection head
│   │   ├── keypoint_head.py     # Keypoint regression head
│   │   └── heatmap_head.py      # Heatmap generation head
│   ├── training/                 # Training system
│   │   └── trainer.py           # Training loop & optimization
│   ├── utils/                    # Utility functions
│   │   ├── device_manager.py    # Hardware management
│   │   ├── logger.py            # Logging system
│   │   └── metric.py            # Evaluation metrics
│   └── visualization/            # Analysis tools
│       └── backbone_vis.py      # Model visualization
├── scripts/                      # Execution scripts
│   ├── train.py                 # Training script
│   └── predict.py               # Inference script
├── configs/                      # Configuration files
│   └── default_config.yaml      # Default settings
├── outputs/                      # Training outputs
│   ├── best_model.pth           # Best model checkpoint
│   └── training.log             # Training logs
└── setup.py                     # Package installation
```

## Kiến trúc hệ thống

### 🧠 Model Architecture
- **Backbone**: MobileNetV3-Small tối ưu cho mobile
- **Person Detection**: PERSON_HEAD với NMS
- **Keypoint Detection**: Dual-head architecture (heatmap + regression)
- **Attention Mechanism**: Channel attention cho feature enhancement

### 📊 Data Pipeline
- **Dataset**: OptimizedKeypointsDataset với LRU caching
- **Augmentation**: Keypoint-aware transformations
- **Batching**: Adaptive batch sampler theo số người
- **Format**: COCO-style 17 keypoints

### 🎯 Loss Functions
- **Keypoint Loss**: MSE với spatial weighting
- **Visibility Loss**: Binary classification
- **Heatmap Loss**: Gaussian heatmap regression
- **Dynamic Balancing**: Adaptive loss weights

## 🛠️ Công cụ và Dependencies

### Core Dependencies
| Package | Phiên bản | Chức năng | Tại sao sử dụng |
|---------|-----------|-----------|-----------------|
| **torch** | Latest stable | Deep learning framework | Industry standard, GPU support, research-friendly |
| **torchvision** | Compatible | Computer vision ops | Integrated với PyTorch, optimized operations |
| **numpy** | Latest stable | Numerical computing | Fast numerical operations, scientific computing |
| **PIL (Pillow)** | Latest stable | Image processing | Standard image library, format support |
| **opencv-python** | Latest stable | Advanced CV ops | Advanced CV operations, video processing |
| **pyyaml** | Latest stable | Config parsing | Human-readable config, easy editing |
| **tqdm** | Latest stable | Progress bars | User experience, training monitoring |

### Development Tools
- **pandas + openpyxl**: Documentation export, data analysis
- **matplotlib**: Visualization và plotting
- **dataclasses**: Type-safe configuration classes
- **pathlib**: Cross-platform path handling
- **logging**: Production-ready logging system

## 📦 Installation

### 1. Clone repository
```bash
git clone https://github.com/your_username/Multi-Person-Keypoint-Detection.git
cd keypoint-detection
```

### 2. Tạo virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
# Cài đặt từ setup.py
pip install -e .

# Hoặc cài đặt manual
pip install torch torchvision numpy pillow opencv-python pyyaml tqdm pandas openpyxl
```

## 🚀 Sử dụng

### Training Model
```bash
python scripts/train.py \
    --config configs/default_config.yaml \
    --data-dir path/to/dataset \
    --output-dir path/to/output \
    [--resume path/to/checkpoint.pth]
```

### Inference
```bash
python scripts/predict.py \
    --image path/to/image.jpg \
    --checkpoint path/to/checkpoint.pth
```

### Programmatic Usage
```python
from dll import MultiPersonKeypointModel, ModelConfig, TrainingConfig
from dll.data import create_optimized_dataloader

# Load model
model = MultiPersonKeypointModel(model_config, training_config)

# Create dataloader
dataloader = create_optimized_dataloader(
    dataset_dir="path/to/data",
    batch_size=32,
    split="train"
)

# Training
from dll.training import Trainer
trainer = Trainer(model, device, train_dataloader, val_dataloader, config)
trained_model, history = trainer.train()
```

## ⚙️ Configuration System

Hệ thống cấu hình linh hoạt với YAML files:

### Cấu hình chính (default_config.yaml)
```yaml
# Device configuration
device:
  type: 'auto'              # auto, cuda, cpu
  mixed_precision: true     # AMP training
  pin_memory: true         # Faster data transfer

# Model architecture
model:
  backbone:
    width_mult: 1.5         # Model capacity
    in_channels: 1          # Grayscale input
    out_channels: 128       # Feature dimensions

  keypoint_head:
    num_keypoints: 17       # COCO format
    dropout_rate: 0.2       # Regularization

  heatmap_head:
    heatmap_size: [56, 56]  # Spatial resolution
    use_attention: true     # Channel attention

# Training hyperparameters
training:
  num_epochs: 30
  batch_size: 128
  pck_thresholds: [0.002, 0.05, 0.2]  # Evaluation metrics

  optimizer:
    name: "adam"
    learning_rate: 0.01
    weight_decay: 0.001

  lr_scheduler:
    factor: 0.1             # LR reduction factor
    patience: 3             # Epochs to wait
    min_lr: 1e-6           # Minimum LR

  loss:
    keypoint_loss_weight: 20.0    # Loss balancing
    visibility_loss_weight: 8.0
    focal_gamma: 2.5              # Focal loss params
```

### Configuration Classes
- **ModelConfig**: Architecture parameters
- **TrainingConfig**: Training hyperparameters
- **DeviceConfig**: Hardware settings
- **AugmentationConfig**: Data augmentation policies

## Dataset Format

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

- **Images:** Store image files (e.g., JPEG, PNG) in the `images/` folder.
- **Labels:** Store annotation files in the `labels/` folder (e.g., YOLO format).

## 🏗️ Model Architecture Chi tiết

### Backbone Network
- **MobileNetV3-Small**: Tối ưu cho mobile devices
- **Width Multiplier**: Điều chỉnh model capacity (0.5-2.0)
- **Channel Attention**: Focus vào features quan trọng
- **Grayscale Support**: Giảm computational cost

### Multi-Head Architecture
1. **Person Detection Head**
   - Detect person bounding boxes
   - Confidence scoring
   - Non-Maximum Suppression (NMS)

2. **Heatmap Head**
   - Generate spatial heatmaps cho mỗi keypoint
   - Gaussian-based target generation
   - Deconvolutional upsampling

3. **Keypoint Regression Head**
   - Direct coordinate prediction
   - Visibility classification
   - Fine-grained localization

### Advanced Features
- **ROI Align**: Extract person-specific features
- **Dynamic Loss Balancing**: Adaptive loss weights
- **Spatial Attention**: Focus on keypoint regions
- **Mixed Precision**: Faster training với AMP

## 📊 Evaluation Metrics

| Metric | Công thức | Threshold/Range | Ý nghĩa |
|--------|-----------|-----------------|---------|
| **PCK** | (correct_keypoints / total_keypoints) × 100 | 0.002, 0.05, 0.2 | Accuracy of keypoint localization |
| **ADE** | Mean Euclidean distance | Pixels (lower better) | Average pixel error |
| **Training Loss** | Weighted sum of components | Positive real | Overall performance |
| **Keypoint Loss** | MSE coordinates | Positive real | Coordinate accuracy |
| **Visibility Loss** | Binary cross entropy | Positive real | Visibility classification |
| **Heatmap Loss** | MSE heatmaps | Positive real | Spatial representation |

### Loss Function Components
```python
total_loss = (
    keypoint_weight * keypoint_loss +
    visibility_weight * visibility_loss +
    heatmap_weight * heatmap_loss
)
```

### Training Monitoring
- **Learning Rate**: Adaptive scheduling (0.01 → 1e-6)
- **Gradient Norm**: Clipped to max_norm=1.0
- **Early Stopping**: Patience=10 epochs
- **Checkpointing**: Best model auto-save

## � API Reference - Chi tiết Functions và Classes

### 1. **MultiPersonKeypointModel** - Model chính

#### Constructor
```python
def __init__(self, config: ModelConfig, training_config: TrainingConfig):
    """
    Khởi tạo model với cấu hình.

    Args:
        config: ModelConfig - Cấu hình kiến trúc model
        training_config: TrainingConfig - Cấu hình training
    """
```

#### Forward Pass
```python
def forward(self, batch) -> Dict[str, torch.Tensor]:
    """
    Forward pass của model.

    Args:
        batch: Dict hoặc Tensor
            - Nếu Dict: {'image': Tensor[B,C,H,W], 'bboxes': List, ...}
            - Nếu Tensor: Images [B,C,H,W]

    Returns:
        Dict chứa:
            - 'keypoints': Tensor[B,N,17,2] - Tọa độ keypoints
            - 'visibilities': Tensor[B,N,17] - Visibility scores
            - 'heatmap': Tensor[B,17,H,W] - Heatmaps
            - 'loss': Scalar - Training loss (nếu có targets)
    """
```

#### Training Step
```python
def train_step(self, batch: Dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    Thực hiện một bước training.

    Args:
        batch: Batch data với keys ['image', 'keypoints', 'visibilities']
        optimizer: PyTorch optimizer

    Returns:
        Dict metrics: {'loss', 'keypoint_loss', 'visibility_loss', 'num_detections'}
    """
```

#### Validation Step
```python
def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Thực hiện validation step.

    Args:
        batch: Validation batch data

    Returns:
        Dict metrics bao gồm PCK@multiple thresholds và ADE
    """
```

### 2. **OptimizedKeypointsDataset** - Dataset Class

#### Constructor
```python
def __init__(self,
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
             cache_size: int = 1000):
    """
    Khởi tạo optimized dataset.

    Args:
        dataset_dir: Đường dẫn đến dataset
        split: "train" hoặc "val"
        img_size: Kích thước resize image
        grayscale: Convert sang grayscale
        num_keypoints: Số keypoints (17 cho COCO)
        heatmap_size: Kích thước heatmap output
        transform: Image transformations
        augmentation: Data augmentation
        max_persons: Số người tối đa per image
        enable_caching: Bật LRU cache
        cache_size: Kích thước cache
    """
```

#### Get Item
```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    Lấy một sample từ dataset.

    Args:
        idx: Index của sample

    Returns:
        Dict chứa:
            - 'image': Tensor[C,H,W] - Processed image
            - 'keypoints': Tensor[N,17,2] - Keypoint coordinates
            - 'visibilities': Tensor[N,17] - Visibility flags
            - 'bboxes': Tensor[N,4] - Person bounding boxes
            - 'heatmaps': Tensor[17,H,W] - Target heatmaps
    """
```

### 3. **Trainer** - Training System

#### Constructor
```python
def __init__(self,
             model: nn.Module,
             device: torch.device,
             train_dataloader: DataLoader,
             val_dataloader: DataLoader,
             config: TrainingConfig,
             output_dir: Optional[Union[str, Path]] = None,
             use_amp: bool = False):
    """
    Khởi tạo trainer.

    Args:
        model: Model cần train
        device: Device (CPU/GPU)
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        config: Training configuration
        output_dir: Thư mục lưu outputs
        use_amp: Sử dụng Automatic Mixed Precision
    """
```

#### Train Method
```python
def train(self) -> Tuple[nn.Module, Dict]:
    """
    Thực hiện training loop chính.

    Returns:
        Tuple[trained_model, training_history]
            - trained_model: Model đã train
            - training_history: Dict chứa metrics theo epoch
    """
```

#### Save/Load Checkpoint
```python
def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
    """Lưu model checkpoint."""

def load_checkpoint(self, checkpoint_path: str) -> int:
    """
    Load checkpoint và resume training.

    Returns:
        start_epoch: Epoch để resume
    """
```

## �📋 Tóm tắt chức năng

### Core Features
✅ **Multi-person keypoint detection** - Phát hiện 17 keypoints/người
✅ **MobileNetV3 backbone** - Tối ưu cho mobile deployment
✅ **Advanced loss functions** - Dynamic balancing, spatial weighting
✅ **Flexible configuration** - YAML-based parameter management
✅ **Production-ready** - Robust training, checkpointing, monitoring
✅ **Hardware optimization** - CPU/GPU support, mixed precision

### Technical Highlights
🔬 **Research-grade**: State-of-the-art loss balancing techniques
⚡ **Performance**: Optimized data pipeline với caching
🛠️ **Maintainable**: Modular architecture, type-safe configs
📊 **Comprehensive**: Full metrics suite (PCK, ADE, loss components)
🎯 **Accurate**: Dual-head architecture (heatmap + regression)

## 📚 Documentation

- **📊 Excel Documentation**: `Keypoint_Detection_Documentation.xlsx` - Chi tiết đầy đủ về modules, functions, configs
- **📖 README**: File này - Hướng dẫn sử dụng và tổng quan
- **⚙️ Config Files**: `configs/default_config.yaml` - Tham số cấu hình
- **📝 Code Comments**: Inline documentation trong source code

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

[Your License Here]

## 📞 Contact & Support

- **Issues**: GitHub Issues cho bug reports
- **Discussions**: GitHub Discussions cho questions
- **Email**: [your-email@domain.com]

## 🙏 Acknowledgments

- PyTorch team cho excellent framework
- MobileNetV3 authors cho efficient architecture
- COCO dataset cho keypoint annotations
- Open source community

---

**Happy coding! 🚀**

*Dự án này được phát triển với mục đích nghiên cứu và giáo dục trong lĩnh vực Computer Vision và Human Pose Estimation.*