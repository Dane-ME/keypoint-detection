# Device Management Solution

## Vấn đề đã giải quyết

Đã thành công giải quyết lỗi **"Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"** và thiết lập hệ thống quản lý device tập trung cho toàn bộ dự án.

## Giải pháp được triển khai

### 1. DeviceManager - Quản lý device tập trung

**File:** `dll/utils/device_manager.py`

- **Singleton pattern**: Đảm bảo chỉ có một instance DeviceManager trong toàn bộ ứng dụng
- **Centralized device configuration**: Tất cả components sử dụng cùng một device
- **Automatic device detection**: Tự động chọn CUDA nếu có, ngược lại sử dụng CPU
- **Mixed precision support**: Hỗ trợ mixed precision training khi sử dụng CUDA
- **Utility functions**: Cung cấp các hàm tiện ích để tạo và di chuyển tensors

**Tính năng chính:**
```python
# Khởi tạo device manager
initialize_device_manager(device_config)

# Lấy device hiện tại
device = get_device()

# Di chuyển tensor/model lên device
tensor = to_device(tensor)
model = to_device(model)

# Di chuyển toàn bộ batch lên device
batch = move_batch_to_device(batch)

# Tạo tensor trực tiếp trên device
zeros_tensor = device_manager.zeros(2, 3, 4)
ones_tensor = device_manager.ones(2, 3, 4)
```

### 2. Config Integration - Tích hợp với hệ thống config

**Files updated:**
- `configs/default_config.yaml`
- `dll/configs/base_config.py`
- `dll/configs/training_config.py`

**Device configuration trong YAML:**
```yaml
device:
  type: 'auto'  # 'auto', 'cuda', 'cpu', hoặc specific như 'cuda:0'
  force_cpu: false
  mixed_precision: true
  pin_memory: true
```

### 3. DataLoader Updates - Cập nhật DataLoader

**File:** `dll/data/dataloader.py`

**Thay đổi chính:**
- **Device-aware collate function**: `efficient_collate_fn` sử dụng DeviceManager
- **Fallback mechanism**: Xử lý trường hợp DeviceManager chưa được khởi tạo trong worker processes
- **Pin memory management**: Tự động tắt pin_memory khi sử dụng DeviceManager để tránh xung đột
- **Consistent tensor creation**: Tất cả tensors được tạo trên cùng device

### 4. Training Pipeline Updates - Cập nhật pipeline training

**Files updated:**
- `scripts/train.py`
- `dll/training/trainer.py`
- `dll/losses/keypoint_loss.py`

**Thay đổi chính:**
- **Early initialization**: DeviceManager được khởi tạo ngay từ đầu trong train.py
- **Automatic device detection**: Trainer tự động sử dụng device từ DeviceManager
- **Mixed precision integration**: AMP scaler được cấu hình dựa trên DeviceManager
- **Loss function updates**: KeypointLoss sử dụng DeviceManager cho device consistency

### 5. Khôi phục Loss Components

**File:** `dll/losses/keypoint_loss.py`

Đã khôi phục lại `visibility_loss` và `coordinate_loss` trong KeypointLoss:
- **Visibility Loss**: Sử dụng FocalLoss cho dự đoán visibility của keypoints
- **Coordinate Loss**: Sử dụng SmoothL1Loss cho dự đoán tọa độ keypoints
- **Device consistency**: Tất cả tensors trong loss computation đều trên cùng device

## Kết quả

### ✅ Thành công
1. **Không còn lỗi device mismatch**: Training chạy hoàn toàn ổn định
2. **Device consistency**: Tất cả tensors đều trên cùng device (CUDA)
3. **Mixed precision**: Hoạt động đúng với CUDA
4. **Centralized configuration**: Chỉ cần set device trong config
5. **Loss components restored**: visibility_loss và coordinate_loss hoạt động trở lại

### 📊 Training Results
```
Train Loss: 0.4948
Val Loss: 0.0342
Average Distance Error: 0.1631
PCK@0.2: 0.4080
```

## Cách sử dụng

### 1. Cấu hình Device
Chỉnh sửa `configs/default_config.yaml`:
```yaml
device:
  type: 'auto'        # Tự động chọn CUDA nếu có
  force_cpu: false    # Bắt buộc dùng CPU
  mixed_precision: true  # Bật mixed precision
  pin_memory: true    # Bật pin memory
```

### 2. Chạy Training
```bash
python scripts/train.py --config configs/default_config.yaml --data_dir /path/to/data --output_dir outputs
```

### 3. Sử dụng DeviceManager trong code
```python
from dll.utils import initialize_device_manager, get_device, to_device

# Khởi tạo (thường được làm tự động trong train.py)
initialize_device_manager(device_config)

# Sử dụng
device = get_device()
tensor = to_device(my_tensor)
```

## Lợi ích

1. **Consistency**: Tất cả components sử dụng cùng device
2. **Simplicity**: Chỉ cần cấu hình device ở một nơi
3. **Flexibility**: Dễ dàng chuyển đổi giữa CPU/CUDA
4. **Robustness**: Xử lý fallback khi DeviceManager chưa được khởi tạo
5. **Performance**: Hỗ trợ mixed precision và pin memory optimization

## Testing

Đã tạo script test để verify functionality:
```bash
python test_device_manager.py
```

Test coverage:
- ✅ DeviceManager initialization
- ✅ Tensor creation and movement
- ✅ Batch processing
- ✅ KeypointLoss integration
- ✅ Config integration

## Troubleshooting

### Nếu gặp lỗi device mismatch:
1. Kiểm tra DeviceManager đã được khởi tạo chưa
2. Đảm bảo `num_workers=0` trong config để tránh multiprocessing issues
3. Kiểm tra tất cả tensors được tạo thông qua DeviceManager

### Nếu muốn force CPU:
```yaml
device:
  force_cpu: true
```

### Nếu muốn tắt mixed precision:
```yaml
device:
  mixed_precision: false
```

## Kết luận

Hệ thống DeviceManager đã thành công giải quyết vấn đề device mismatch và cung cấp một giải pháp quản lý device tập trung, dễ sử dụng và mạnh mẽ cho toàn bộ dự án keypoint detection.
