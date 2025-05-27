#!/usr/bin/env python3
"""
Script để tạo tài liệu Excel chi tiết cho dự án Multi-Person Keypoint Detection
"""

import pandas as pd
from pathlib import Path

def create_project_documentation():
    """Tạo file Excel với tài liệu chi tiết về dự án"""

    # Tạo writer để ghi nhiều sheet
    output_file = "Keypoint_Detection_Documentation.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: Tổng quan dự án
        overview_data = {
            'Thông tin': [
                'Tên dự án',
                'Phiên bản',
                'Ngôn ngữ lập trình',
                'Framework chính',
                'Mục đích',
                'Kiến trúc model',
                'Số keypoints',
                'Hỗ trợ đa người',
                'Thiết bị hỗ trợ',
                'Định dạng input',
                'Định dạng output'
            ],
            'Chi tiết': [
                'Multi-Person Keypoint Detection',
                '1.0.0',
                'Python 3.x',
                'PyTorch',
                'Phát hiện và định vị keypoints trên cơ thể người',
                'MobileNetV3-Small + Custom Heads',
                '17 keypoints per person',
                'Có (unlimited persons)',
                'CPU, CUDA GPU',
                'Images (JPEG, PNG)',
                'Keypoint coordinates + visibility'
            ],
            'Tại sao sử dụng': [
                'Dự án nghiên cứu pose estimation',
                'Phiên bản ổn định đầu tiên',
                'Ecosystem phong phú, dễ maintain',
                'Flexible, research-friendly, GPU support',
                'Ứng dụng trong fitness, sports analysis, AR/VR',
                'Tối ưu cho mobile, accuracy cao',
                'Chuẩn COCO format, đầy đủ cho human pose',
                'Thực tế có nhiều người trong ảnh',
                'Linh hoạt deployment',
                'Chuẩn computer vision',
                'Dễ integrate với downstream tasks'
            ]
        }

        df_overview = pd.DataFrame(overview_data)
        df_overview.to_excel(writer, sheet_name='Tổng quan', index=False)

        # Sheet 2: Cấu trúc module
        module_data = {
            'Module/Package': [
                'dll/',
                'dll/configs/',
                'dll/data/',
                'dll/losses/',
                'dll/models/',
                'dll/training/',
                'dll/utils/',
                'dll/visualization/',
                'scripts/',
                'configs/'
            ],
            'Chức năng chính': [
                'Package chính chứa toàn bộ library',
                'Quản lý cấu hình hệ thống',
                'Xử lý dữ liệu và data loading',
                'Các hàm loss function tiên tiến',
                'Kiến trúc neural network models',
                'Training loop và optimization',
                'Các utility functions',
                'Visualization và debugging tools',
                'Scripts chạy training/inference',
                'File cấu hình YAML'
            ],
            'Files quan trọng': [
                '__init__.py, main exports',
                'model_config.py, training_config.py',
                'dataloader.py, augmentation.py',
                'keypoint_loss.py',
                'keypoint_model.py, backbone.py',
                'trainer.py',
                'device_manager.py, logger.py',
                'backbone_vis.py',
                'train.py, predict.py',
                'default_config.yaml'
            ],
            'Tại sao cần thiết': [
                'Tổ chức code modular, dễ maintain',
                'Centralized config management, flexibility',
                'Efficient data pipeline, augmentation',
                'Advanced loss balancing, better convergence',
                'Modular architecture, easy to extend',
                'Robust training process, monitoring',
                'Code reusability, debugging support',
                'Model analysis, debugging',
                'Easy to use interface',
                'Reproducible experiments'
            ]
        }

        df_modules = pd.DataFrame(module_data)
        df_modules.to_excel(writer, sheet_name='Cấu trúc Module', index=False)

        # Sheet 3: Models và Architecture
        models_data = {
            'Component': [
                'MultiPersonKeypointModel',
                'MobileNetV3Wrapper (Backbone)',
                'PERSON_HEAD',
                'HeatmapHead',
                'KEYPOINT_HEAD',
                'ChannelAttention',
                'KeypointLoss',
                'ROI Align',
                'NMS (Non-Maximum Suppression)'
            ],
            'Chức năng': [
                'Model chính tích hợp tất cả components',
                'Feature extraction từ images',
                'Phát hiện bounding boxes của người',
                'Tạo heatmaps cho keypoints',
                'Regression keypoint coordinates',
                'Attention mechanism cho features',
                'Loss function với dynamic balancing',
                'Extract features từ person regions',
                'Loại bỏ duplicate detections'
            ],
            'Input/Output': [
                'Images → Keypoints + Visibility',
                'Images → Feature maps',
                'Features → Person bboxes + confidence',
                'Features → Heatmaps [B,K,H,W]',
                'Features → Coordinates + Visibility',
                'Features → Weighted features',
                'Predictions + Targets → Loss values',
                'Features + ROIs → ROI features',
                'Bboxes + scores → Filtered bboxes'
            ],
            'Tại sao sử dụng': [
                'End-to-end training, modular design',
                'Efficient, mobile-optimized, proven architecture',
                'Multi-person support, accurate detection',
                'Spatial keypoint representation, differentiable',
                'Direct coordinate prediction, fast inference',
                'Focus on important features, better accuracy',
                'Address training instability, better convergence',
                'Handle variable person sizes, spatial alignment',
                'Remove redundant detections, clean output'
            ]
        }

        df_models = pd.DataFrame(models_data)
        df_models.to_excel(writer, sheet_name='Models & Architecture', index=False)

        # Sheet 4: Data Processing
        data_data = {
            'Component': [
                'OptimizedKeypointsDataset',
                'KeypointAugmentation',
                'LRUCache',
                'AdaptiveBatchSampler',
                'ImageTransforms',
                'AnnotationData',
                'Heatmap Generation',
                'Data Collation'
            ],
            'Chức năng': [
                'Dataset class với caching và optimization',
                'Data augmentation cho keypoints',
                'Cache images để tăng tốc loading',
                'Dynamic batching theo số người',
                'Resize, normalize, tensor conversion',
                'Structured annotation storage',
                'Convert keypoints thành heatmaps',
                'Batch multiple samples together'
            ],
            'Tại sao quan trọng': [
                'Fast data loading, memory efficient',
                'Improve generalization, prevent overfitting',
                'Reduce I/O overhead, faster training',
                'Memory optimization, handle variable persons',
                'Standardize input format, model compatibility',
                'Type safety, easy manipulation',
                'Spatial representation, differentiable training',
                'Efficient batch processing'
            ],
            'Công nghệ sử dụng': [
                'PyTorch Dataset, PIL, caching',
                'Geometric transformations, probability',
                'OrderedDict, LRU algorithm',
                'Custom PyTorch Sampler',
                'torchvision.transforms, PIL',
                'Python dataclasses',
                'Gaussian kernels, spatial indexing',
                'PyTorch DataLoader collate_fn'
            ]
        }

        df_data = pd.DataFrame(data_data)
        df_data.to_excel(writer, sheet_name='Data Processing', index=False)

        # Sheet 5: Training System
        training_data = {
            'Component': [
                'Trainer',
                'TrainingHistory',
                'LR Scheduler',
                'Optimizer (Adam/SGD)',
                'Mixed Precision (AMP)',
                'Gradient Clipping',
                'Early Stopping',
                'Checkpoint System',
                'Metrics Tracking'
            ],
            'Chức năng': [
                'Main training loop và coordination',
                'Track training progress và metrics',
                'Adaptive learning rate adjustment',
                'Parameter optimization',
                'Faster training với reduced memory',
                'Prevent gradient explosion',
                'Stop training khi không improve',
                'Save/load model states',
                'Monitor PCK, ADE, loss values'
            ],
            'Cấu hình': [
                'num_epochs, batch_size, validation_interval',
                'Automatic tracking all metrics',
                'ReduceLROnPlateau, factor=0.1, patience=3',
                'lr=0.01, weight_decay=0.001',
                'Enabled on CUDA, GradScaler',
                'max_norm=1.0',
                'patience=10 epochs',
                'best_model.pth, regular intervals',
                'PCK@[0.002,0.05,0.2], ADE'
            ],
            'Tại sao cần thiết': [
                'Organized training process, reproducibility',
                'Monitor progress, debugging, analysis',
                'Prevent overfitting, optimize convergence',
                'Efficient parameter updates',
                'Faster training, larger batch sizes',
                'Training stability, prevent NaN',
                'Save time, prevent overfitting',
                'Resume training, model deployment',
                'Evaluate model performance objectively'
            ]
        }

        df_training = pd.DataFrame(training_data)
        df_training.to_excel(writer, sheet_name='Training System', index=False)

        # Sheet 6: Dependencies và Tools
        deps_data = {
            'Package/Tool': [
                'torch',
                'torchvision',
                'numpy',
                'PIL (Pillow)',
                'opencv-python',
                'pyyaml',
                'tqdm',
                'pandas',
                'openpyxl',
                'matplotlib',
                'dataclasses',
                'pathlib',
                'logging'
            ],
            'Phiên bản': [
                'Latest stable',
                'Compatible với torch',
                'Latest stable',
                'Latest stable',
                'Latest stable',
                'Latest stable',
                'Latest stable',
                'Latest stable',
                'Latest stable',
                'Latest stable',
                'Python 3.7+',
                'Python 3.4+',
                'Built-in'
            ],
            'Chức năng': [
                'Deep learning framework chính',
                'Computer vision operations',
                'Numerical computing, arrays',
                'Image loading và processing',
                'Advanced image processing',
                'YAML config file parsing',
                'Progress bars cho training',
                'Data manipulation, Excel export',
                'Excel file creation',
                'Visualization và plotting',
                'Structured configuration classes',
                'File path operations',
                'Logging và debugging'
            ],
            'Tại sao sử dụng': [
                'Industry standard, GPU support, research friendly',
                'Integrated với PyTorch, optimized operations',
                'Fast numerical operations, scientific computing',
                'Standard image library, format support',
                'Advanced CV operations, video processing',
                'Human-readable config, easy editing',
                'User experience, training monitoring',
                'Data analysis, documentation export',
                'Professional documentation format',
                'Model analysis, result visualization',
                'Type safety, IDE support, validation',
                'Cross-platform path handling',
                'Debugging, monitoring, production logging'
            ]
        }

        df_deps = pd.DataFrame(deps_data)
        df_deps.to_excel(writer, sheet_name='Dependencies & Tools', index=False)

        # Sheet 7: Configuration System
        config_data = {
            'Config Section': [
                'paths',
                'device',
                'model.backbone',
                'model.person_head',
                'model.keypoint_head',
                'model.heatmap_head',
                'training',
                'training.optimizer',
                'training.lr_scheduler',
                'training.augmentation',
                'training.loss'
            ],
            'Tham số chính': [
                'data_dir, output_dir, default_config',
                'type, force_cpu, mixed_precision, pin_memory',
                'width_mult, in_channels, out_channels, input_size',
                'conf_threshold, nms_iou_threshold, anchor_sizes',
                'num_keypoints, height, width, dropout_rate',
                'hidden_channels, heatmap_size, use_attention',
                'num_epochs, batch_size, num_workers, pck_thresholds',
                'name, learning_rate, weight_decay, momentum',
                'factor, patience, min_lr, mode, threshold',
                'enabled, prob, flip, rotate, scale',
                'keypoint_loss_weight, visibility_loss_weight, focal_gamma'
            ],
            'Mục đích': [
                'Định nghĩa đường dẫn files và directories',
                'Cấu hình hardware và performance',
                'Kiến trúc backbone network',
                'Person detection parameters',
                'Keypoint regression settings',
                'Heatmap generation configuration',
                'Training hyperparameters',
                'Optimization algorithm settings',
                'Learning rate scheduling',
                'Data augmentation policies',
                'Loss function balancing'
            ],
            'Tại sao cần thiết': [
                'Flexibility, environment adaptation',
                'Hardware optimization, compatibility',
                'Model capacity control, mobile optimization',
                'Detection accuracy, speed tradeoff',
                'Keypoint precision, overfitting control',
                'Spatial representation quality',
                'Training efficiency, convergence control',
                'Optimization effectiveness',
                'Adaptive learning, convergence',
                'Generalization, robustness',
                'Training stability, component balancing'
            ]
        }

        df_config = pd.DataFrame(config_data)
        df_config.to_excel(writer, sheet_name='Configuration System', index=False)

        # Sheet 8: Metrics và Evaluation
        metrics_data = {
            'Metric': [
                'PCK (Percentage of Correct Keypoints)',
                'ADE (Average Distance Error)',
                'Training Loss',
                'Keypoint Loss',
                'Visibility Loss',
                'Heatmap Loss',
                'Coordinate Loss',
                'Learning Rate',
                'Gradient Norm'
            ],
            'Công thức/Cách tính': [
                '(correct_keypoints / total_keypoints) * 100',
                'Mean Euclidean distance between pred và gt',
                'Weighted sum of all loss components',
                'MSE between predicted và ground truth coordinates',
                'Binary cross entropy for keypoint visibility',
                'MSE between predicted và target heatmaps',
                'Spatial-aware coordinate regression loss',
                'Current optimizer learning rate',
                'L2 norm of model gradients'
            ],
            'Threshold/Range': [
                'Thresholds: 0.002, 0.05, 0.2 (normalized)',
                'Pixels (lower is better)',
                'Positive real number (lower is better)',
                'Positive real number (lower is better)',
                'Positive real number (lower is better)',
                'Positive real number (lower is better)',
                'Positive real number (lower is better)',
                '0.01 → 1e-6 (adaptive)',
                'Clipped to max_norm=1.0'
            ],
            'Ý nghĩa': [
                'Accuracy of keypoint localization',
                'Average pixel error in keypoint prediction',
                'Overall model performance',
                'Keypoint coordinate prediction accuracy',
                'Keypoint visibility classification accuracy',
                'Spatial heatmap representation quality',
                'Coordinate regression with spatial weighting',
                'Training speed control',
                'Training stability indicator'
            ]
        }

        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_excel(writer, sheet_name='Metrics & Evaluation', index=False)

    print(f"✅ Đã tạo file tài liệu: {output_file}")
    return output_file

if __name__ == "__main__":
    create_project_documentation()
