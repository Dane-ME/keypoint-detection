# Multi-Person Keypoint Detection

A deep learning library for multi-person keypoint detection using PyTorch, featuring a custom MobileNetV3-based architecture with an improved keypoint detection head.

## Features

- Multi-person keypoint detection with 17 keypoints per person
- Custom MobileNetV3-Small backbone optimized for mobile devices
- Improved keypoint detection head with spatial attention
- Person detection with Non-Maximum Suppression (NMS)
- Flexible training configuration system via YAML files
- Comprehensive evaluation metrics such as PCK and ADE

## Project Structure

```
dll/
├── configs/           # Configuration classes
├── data/              # Data loading and preprocessing
├── models/            # Neural network models
├── training/          # Training utilities
└── utils/             # Helper functions and visualization

scripts/
├── train.py           # Training script
└── predict.py         # Inference script
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/Multi-Person-Keypoint-Detection.git
   cd Multi-Person-Keypoint-Detection
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run the training script with the required configuration and dataset paths:

```bash
python scripts/train.py \
    --config configs/default_config.yaml \
    --data-dir path/to/dataset \
    --output-dir path/to/output \
    [--resume path/to/checkpoint.pth]
```

### Inference

To run inference on new images, use:

```bash
python scripts/predict.py --image path/to/image.jpg --checkpoint path/to/checkpoint.pth
```

## Configuration

The model and training parameters can be adjusted using YAML configuration files. Below is an example configuration:

```yaml
model:
  num_keypoints: 17
  width_mult: 1.0
  in_channels: 3

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
```

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

## Model Architecture

- **Backbone:** A custom MobileNetV3-Small model for feature extraction.
- **Person Detection:** A dedicated head that uses NMS.
- **Keypoint Detection:** An improved head with spatial attention for accurate keypoint estimation.

## Evaluation Metrics

- **PCK:** Percentage of Correct Keypoints.
- **ADE:** Average Distance Error.
- Loss components include keypoint loss, visibility loss, and heatmap loss.

## License

[Your License]

## Citation

If you use this code in your research, please cite:

```
[Your Citation]
```

Happy coding!