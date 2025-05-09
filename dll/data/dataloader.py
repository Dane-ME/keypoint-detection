import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.optim as optim
import torch
import numpy as np
from dll.data.transforms import ITransform
from dll.data.augmentation import KeypointAugmentation
from dll.models.heatmap_head import generate_target_heatmap

class KeypointsDataset(Dataset):
    def __init__(self, dataset_dir, split='train', img_size=512, grayscale=False):
        self.img_dir = os.path.join(dataset_dir, split, 'images')
        self.label_dir = os.path.join(dataset_dir, split, 'labels')
        self.img_size = img_size
        self.grayscale = grayscale
        self.num_keypoints = 17
        
        all_img_files = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.img_files = []
        self.label_files = []
        
        for img_path in all_img_files:
            base_name = Path(img_path).stem
            label_path = os.path.join(self.label_dir, f"{base_name}.txt")
            if os.path.exists(label_path):
                self.img_files.append(img_path)
                self.label_files.append(label_path)
            else:
                print(f"Warning: No label file found for {img_path}")
        
        print(f"Dataset loaded: {len(self.img_files)} images with valid labels")
        
        self.transform = ITransform(
            img_size=self.img_size, 
            clip_limit=1.5, 
            tile_size=(8, 8)
        )
        
        self.augmentation = KeypointAugmentation() if split == 'train' else None
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        if idx >= len(self.img_files) or idx >= len(self.label_files):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.img_files)} images and {len(self.label_files)} labels.")
        
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # Load image as grayscale if specified
        if self.grayscale is True:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=2)  # Add channel dimension
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_width, orig_height = image.shape[1], image.shape[0]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        all_keypoints = []
        all_visibilities = []
        all_classes = []
        all_bboxes = []
        
        for line in lines:
            data = line.strip().split()
            
            # Kiểm tra dữ liệu có đủ không
            if len(data) < 5:  # Ít nhất phải có class_id + bbox info (4 giá trị)
                print(f"Warning: Line does not have enough data: {len(data)} values")
                continue
            
            # Lấy class (0 = person)
            class_id = int(data[0])
            
            # Thông tin bbox: width, height, center_x, center_y
            
            bbox_center_x = float(data[1])
            bbox_center_y = float(data[2])
            bbox_width = float(data[3])
            bbox_height = float(data[4])
            
            # Tạo mảng cho keypoint và visibility
            keypoints = []
            visibilities = []
            
            # Xử lý các keypoint, mỗi keypoint có 3 giá trị (x, y, visible)
            for i in range(5, len(data) - 2, 3):
                if i + 2 < len(data):
                    x = float(data[i])
                    y = float(data[i + 1])
                    v = int(float(data[i + 2]))  # Trạng thái visible (0, 1, hoặc 2)
                    
                    # Đảm bảo các giá trị nằm trong khoảng [0, 1]
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    
                    keypoints.append([x, y])
                    visibilities.append(v)
            
            # Đảm bảo đủ 17 keypoint
            while len(keypoints) < self.num_keypoints:
                keypoints.append([0, 0])
                visibilities.append(0)
            
            # Cắt bớt nếu có nhiều hơn 17 keypoint
            keypoints = keypoints[:self.num_keypoints]
            visibilities = visibilities[:self.num_keypoints]
            
            all_keypoints.append(keypoints)
            all_visibilities.append(visibilities)
            all_classes.append(class_id)
            all_bboxes.append([bbox_center_x, bbox_center_y, bbox_width, bbox_height])
        
        # Trường hợp không có person nào trong ảnh
        if len(all_keypoints) == 0:
            print(f"Warning: No valid keypoints found in {label_path}")
            # Tạo một dummy person với tất cả keypoint không hiển thị
            dummy_keypoints = [[0, 0] for _ in range(self.num_keypoints)]
            dummy_visibilities = [0 for _ in range(self.num_keypoints)]
            dummy_bbox = [0, 0, 0, 0]  # center_x, center_y, width, height
            all_keypoints.append(dummy_keypoints)
            all_visibilities.append(dummy_visibilities)
            all_classes.append(0)  # Class 0 = person
            all_bboxes.append(dummy_bbox)
        
        # Chuyển đổi sang tensor
        keypoints_tensor = torch.tensor(all_keypoints, dtype=torch.float32)
        visibilities_tensor = torch.tensor(all_visibilities, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.long)
        bboxes_tensor = torch.tensor(all_bboxes, dtype=torch.float32)
        
        # Convert numpy array to PIL Image before transforms
        if self.grayscale:
            pil_image = Image.fromarray(image.squeeze(), mode='L')
        else:
            pil_image = Image.fromarray(image, mode='RGB')
        
        # Apply augmentation if in training mode
        if self.augmentation is not None:
            pil_image, keypoints_tensor, visibilities_tensor, bboxes_tensor = self.augmentation(
                pil_image, keypoints_tensor, visibilities_tensor, bboxes_tensor)
        
        # Apply existing transforms
        image_return = self.transform.transform(pil_image)
        
        # Generate heatmaps from keypoints
        gt_heatmaps = generate_target_heatmap(
            keypoints=keypoints_tensor.unsqueeze(0),  # Add batch dimension [1, num_persons, 17, 2]
            heatmap_size=(56, 56),  # Adjust size according to your model's output
            sigma=2.0
        )

        # Remove batch dimension as it will be added by DataLoader
        gt_heatmaps = gt_heatmaps.squeeze(0)  # [num_persons, 17, 56, 56]

        return {
            'image': image_return,
            'heatmaps': gt_heatmaps,  # Shape: [num_persons, 17, 56, 56]
            'visibilities': visibilities_tensor,  # Shape: [num_persons, 17]
            'bboxes': bboxes_tensor,  # Shape: [num_persons, 4]
            'num_persons': len(all_keypoints),
            'img_path': img_path,
            'orig_size': (orig_width, orig_height)
        }

def custom_collate_fn(batch):
    """Custom collate function to handle variable number of people per image"""
    batch_images = []
    batch_heatmaps = []
    batch_visibilities = []
    batch_bboxes = []
    batch_num_persons = []
    
    # Find max number of persons in batch
    max_persons = max(sample['heatmaps'].shape[0] for sample in batch)
    
    for sample in batch:
        batch_images.append(sample['image'])
        
        curr_persons = sample['heatmaps'].shape[0]
        num_keypoints = sample['heatmaps'].shape[1]
        heatmap_size = sample['heatmaps'].shape[2:]  # (56, 56)
        
        if curr_persons < max_persons:
            # Pad heatmaps
            padding_heatmaps = torch.zeros(
                (max_persons - curr_persons, num_keypoints, *heatmap_size),
                dtype=sample['heatmaps'].dtype
            )
            heatmaps_padded = torch.cat([sample['heatmaps'], padding_heatmaps], dim=0)
            
            # Pad visibilities
            padding_visibilities = torch.zeros(
                (max_persons - curr_persons, num_keypoints),
                dtype=sample['visibilities'].dtype
            )
            visibilities_padded = torch.cat([sample['visibilities'], padding_visibilities], dim=0)
            
            # Pad bboxes
            padding_bboxes = torch.zeros(
                (max_persons - curr_persons, 4),
                dtype=sample['bboxes'].dtype
            )
            bboxes_padded = torch.cat([sample['bboxes'], padding_bboxes], dim=0)
        else:
            heatmaps_padded = sample['heatmaps']
            visibilities_padded = sample['visibilities']
            bboxes_padded = sample['bboxes']
        
        batch_heatmaps.append(heatmaps_padded)
        batch_visibilities.append(visibilities_padded)
        batch_bboxes.append(bboxes_padded)
        batch_num_persons.append(curr_persons)
    
    # Stack all tensors
    batch_images = torch.stack(batch_images)
    batch_heatmaps = torch.stack(batch_heatmaps)
    batch_visibilities = torch.stack(batch_visibilities)
    batch_bboxes = torch.stack(batch_bboxes)
    batch_num_persons = torch.tensor(batch_num_persons)
    
    return {
        'image': batch_images,
        'heatmaps': batch_heatmaps,  # Shape: [batch_size, max_persons, 17, 56, 56]
        'visibilities': batch_visibilities,  # Shape: [batch_size, max_persons, 17]
        'bboxes': batch_bboxes,  # Shape: [batch_size, max_persons, 4]
        'num_persons': batch_num_persons  # Shape: [batch_size]
    }

def create_dataloader(dataset_dir: str, 
                     batch_size: int = 32,
                     num_workers: int = 4,
                     split: str = 'train',
                     img_size: int = 224,
                     grayscale: bool = True):
    """Create dataloader for training or validation"""
    
    dataset = KeypointsDataset(
        dataset_dir=dataset_dir,
        split=split,
        img_size=img_size,
        grayscale=grayscale
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    return dataloader

def visualize_keypoints_multi_person(image, keypoints_list, visibilities_list, save_path=None):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import hsv_to_rgb
    
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        image = image.permute(1, 2, 0).cpu().numpy()
        # Reverse normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1) * 255
        image = image.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    # Define colors for different visibility states
    color_map = {
        1: (0, 255, 0),    # Green for visible
        2: (255, 0, 0),    # Blue for occluded
        0: (128, 128, 128) # Gray for not visible
    }
    
    def get_coordinate(coord):
        """Helper function to safely extract coordinate value"""
        if isinstance(coord, torch.Tensor):
            return coord.item() if coord.numel() == 1 else coord[0].item()
        elif isinstance(coord, np.ndarray):
            return coord.item() if coord.size == 1 else coord[0]
        return float(coord)
    
    def get_visibility_state(vis):
        """Helper function to convert visibility value to 0, 1, or 2"""
        if isinstance(vis, (np.ndarray, torch.Tensor)):
            if vis.size > 1:
                v = np.argmax(vis)
            else:
                v = int(vis.item() if hasattr(vis, 'item') else vis)
        else:
            v = int(vis)
        
        # Ensure visibility is 0, 1, or 2
        return max(0, min(2, v))
    
    # Draw keypoints for each person
    num_persons = len(keypoints_list)
    for person_idx in range(num_persons):
        keypoints = keypoints_list[person_idx]
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        visibilities = visibilities_list[person_idx]
        if isinstance(visibilities, torch.Tensor):
            visibilities = visibilities.cpu().numpy()
        
        # Create unique color for each person
        hue = person_idx / max(1, num_persons - 1)
        person_color_hsv = np.array([hue, 0.8, 0.8])
        person_color_rgb = hsv_to_rgb(person_color_hsv)
        person_color_bgr = (
            int(person_color_rgb[2] * 255),
            int(person_color_rgb[1] * 255),
            int(person_color_rgb[0] * 255)
        )
        
        # Draw keypoints
        for i in range(len(keypoints)):
            # Get visibility score and ensure it's 0, 1, or 2
            v = get_visibility_state(visibilities[i])
            
            # Skip if not visible
            if v == 0:
                continue
            
            # Get coordinates
            try:
                x = get_coordinate(keypoints[i][0])
                y = get_coordinate(keypoints[i][1])
            except (IndexError, AttributeError, ValueError):
                continue
            
            # Convert coordinates from [0,1] to pixels
            x_px = int(x * w)
            y_px = int(y * h)
            
            # Ensure coordinates are within image bounds
            x_px = max(0, min(w-1, x_px))
            y_px = max(0, min(h-1, y_px))
            
            # Draw point
            cv2.circle(img_bgr, (x_px, y_px), 5, color_map[v], -1)
            
            # Add keypoint label
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            if i < len(keypoint_names):
                cv2.putText(
                    img_bgr, 
                    f"{i+1}:{keypoint_names[i]}", 
                    (x_px + 5, y_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    color_map[v], 
                    1
                )
    
    # Save or display the image
    if save_path:
        cv2.imwrite(save_path, img_bgr)
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def visualize_batch(batch, max_images=4, save_dir=None):
    """
    Hiển thị một số ảnh từ batch với keypoints được vẽ
    
    Args:
        batch: Batch từ dataloader
        max_images: Số lượng ảnh tối đa để hiển thị
        save_dir: Thư mục để lưu ảnh, nếu None thì hiển thị
    """
    images = batch['image']
    all_keypoints = batch['keypoints']
    all_visibilities = batch['visibilities']
    
    batch_size = len(images)
    num_to_show = min(batch_size, max_images)
    
    for i in range(num_to_show):
        image = images[i]
        keypoints = all_keypoints[i]
        visibilities = all_visibilities[i]
        
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"vis_image_{i}.png")
        
        visualize_keypoints_multi_person(
            image, 
            keypoints, 
            visibilities, 
            save_path=save_path
        )

def main():
    """Test dataloader functionality"""
    import matplotlib.pyplot as plt
    from pathlib import Path
    import time
    
    # Test configuration
    dataset_dir = "D:/AI/MobileNET/_project/minidatasets"
    batch_size = 2  # Reduced batch size for testing
    img_size = 512
    num_workers = 0
    
    print("\n=== Testing Dataloader Configuration ===")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Number of workers: {num_workers}")
    
    # Create dataloaders once and reuse
    dataloaders = {}
    datasets_info = {}
    
    for split in ['train', 'val']:
        print(f"\n=== Initializing {split} dataloader ===")
        
        try:
            dataloader = create_dataloader(
                dataset_dir=dataset_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                img_size=img_size
            )
            
            dataloaders[split] = dataloader
            datasets_info[split] = {
                'num_samples': len(dataloader.dataset),
                'num_batches': len(dataloader)
            }
            
            print(f"Successfully created {split} dataloader:")
            print(f"- Number of samples: {datasets_info[split]['num_samples']}")
            print(f"- Number of batches: {datasets_info[split]['num_batches']}")
            
        except Exception as e:
            print(f"Failed to create {split} dataloader: {str(e)}")
            continue
    
    print("\n=== Testing Batch Loading ===")
    
    for split, dataloader in dataloaders.items():
        print(f"\nTesting {split} dataloader:")
        
        if datasets_info[split]['num_batches'] == 0:
            print(f"No batches available in {split} dataset")
            continue
        
        try:
            # Test first batch
            start_time = time.time()
            batch = next(iter(dataloader))
            load_time = time.time() - start_time
            
            print(f"\nFirst batch loaded in {load_time:.2f} seconds")
            print("Batch contents:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"- {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"- {key}: {type(value)}")
            
            # Validate batch contents
            assert 'image' in batch, "Batch missing 'image' key"
            assert 'keypoints' in batch, "Batch missing 'keypoints' key"
            assert 'visibilities' in batch, "Batch missing 'visibilities' key"
            assert 'bboxes' in batch, "Batch missing 'bboxes' key"
            
            # Test batch shapes
            assert batch['image'].dim() == 4, f"Expected 4D tensor for images, got {batch['image'].dim()}D"
            assert batch['keypoints'].dim() == 4, f"Expected 4D tensor for keypoints, got {batch['keypoints'].dim()}D"
            assert batch['visibilities'].dim() == 3, f"Expected 3D tensor for visibilities, got {batch['visibilities'].dim()}D"
            
            # Visualize batch
            print("\nVisualizing batch...")
            save_dir = Path("debug_output") / split
            save_dir.mkdir(parents=True, exist_ok=True)
            
            visualize_batch(
                batch,
                max_images=min(2, len(batch['image'])),
                save_dir=save_dir
            )
            print(f"Visualizations saved to {save_dir}")
            
            # Test full iteration
            print("\nTesting full dataset iteration...")
            start_time = time.time()
            total_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                total_batches += 1
                if batch_idx % 2 == 0:  # Print progress every 2 batches
                    print(f"Successfully loaded batch {batch_idx}/{datasets_info[split]['num_batches']}")
            
            iteration_time = time.time() - start_time
            print(f"\nFull iteration completed:")
            print(f"- Total batches processed: {total_batches}")
            print(f"- Total time: {iteration_time:.2f} seconds")
            print(f"- Average time per batch: {iteration_time/total_batches:.2f} seconds")
            
        except Exception as e:
            print(f"Error during {split} dataset testing:")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n=== Dataloader Testing Completed ===")

if __name__ == '__main__':
    main()
