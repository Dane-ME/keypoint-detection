"""Image transformation utilities."""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import distance_transform_edt

class ITransform:
    """A custom image transformation pipeline with resizing, grayscale conversion, CLAHE, and normalization."""

    def __init__(self, img_size=224, clip_limit=1.5, tile_size=(8, 8), grayscale=True):
        """Initialize transform pipeline.

        Args:
            img_size (int): Target size for resizing the image
            clip_limit (float): Threshold for contrast limiting in CLAHE
            tile_size (tuple): Grid size for CLAHE
            grayscale (bool): Whether to apply grayscale conversion
        """
        self.img_size = img_size
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.grayscale = grayscale

        if grayscale:
            # Grayscale pipeline
            self.transform = transforms.Compose([
                transforms.Lambda(self.to_grayscale_clahe),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize cho 1 kênh
            ])
        else:
            # RGB pipeline
            self.transform = transforms.Compose([
                transforms.Lambda(self.to_rgb_clahe),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])

    def to_grayscale_clahe(self, img):
        """Convert image to grayscale, apply CLAHE and combine with edges."""
        # Chuyển sang numpy array và grayscale
        img = np.array(img)
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Áp dụng CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)
        clahe_img = clahe.apply(gray)

        # Xử lý nhiễu và tạo biên
        denoised = cv2.GaussianBlur(clahe_img, (5, 5), 1.5)
        denoised = cv2.medianBlur(denoised, 5)
        edges = cv2.Canny(denoised, 100, 200)

        # Thực hiện morphology để làm mịn biên
        kernel = np.ones((3, 3), np.uint8)
        dilated1 = cv2.dilate(edges, kernel, iterations=1)
        eroded1 = cv2.erode(dilated1, kernel, iterations=1)
        dilated2 = cv2.dilate(eroded1, kernel, iterations=1)
        dilated_edges = cv2.erode(dilated2, kernel, iterations=1)

        dilated_edges = cv2.GaussianBlur(dilated_edges, (3, 3), 0)
        # Chuẩn hóa edges về dải [0, 255]
        normalized_edges = (dilated_edges.astype(float) / dilated_edges.max() * 255).astype(np.uint8) if dilated_edges.max() > 0 else dilated_edges

        # Kết hợp CLAHE và edges với trọng số bằng nhau
        combined = cv2.addWeighted(clahe_img, 0.7, normalized_edges, 0.3, 0)

        #Làm mịn kết quả cuối cùng
        #combined = cv2.GaussianBlur(combined, (3, 3), 0)

        return Image.fromarray(combined.astype(np.uint8), mode='L')

    def to_rgb_clahe(self, img):
        """Apply CLAHE to RGB image while preserving color channels."""
        # Convert to numpy array
        img = np.array(img)

        # Ensure RGB format
        if len(img.shape) == 2:
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Apply CLAHE to each channel separately
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)

        # Split channels
        r, g, b = cv2.split(img)

        # Apply CLAHE to each channel
        r_clahe = clahe.apply(r)
        g_clahe = clahe.apply(g)
        b_clahe = clahe.apply(b)

        # Merge channels back
        img_clahe = cv2.merge([r_clahe, g_clahe, b_clahe])

        # Optional: Apply slight denoising
        img_clahe = cv2.GaussianBlur(img_clahe, (3, 3), 0.5)

        return Image.fromarray(img_clahe.astype(np.uint8), mode='RGB')

    def __call__(self, img):
        """Apply the full transformation pipeline to the input image."""
        return self.transform(img)

# Ví dụ sử dụng
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = "D:\\AI\\Keypoint_model\\minidatasets\\train\\images\\000000581860.jpg"
    try:
        # Đọc ảnh
        img = Image.open(img_path)
        custom_transform = ITransform(img_size=224, clip_limit=2.0, tile_size=(5, 5))

        # Lấy ảnh gốc
        img_array = np.array(img)

        # Chuyển sang grayscale để so sánh
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Gọi trực tiếp hàm to_grayscale_clahe
        processed_img = custom_transform.to_grayscale_clahe(img)
        processed_array = np.array(processed_img)

        # Hiển thị kết quả
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(img_array)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(processed_array, cmap='gray')
        plt.title('After to_grayscale_clahe')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # In thông tin về ảnh
        print("\nThông tin ảnh sau to_grayscale_clahe:")
        print(f"Shape: {processed_array.shape}")
        print(f"Value range: [{processed_array.min()}, {processed_array.max()}]")
        print(f"Mean: {processed_array.mean():.2f}")
        print(f"Std: {processed_array.std():.2f}")

    except FileNotFoundError:
        print(f"Không tìm thấy file: {img_path}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
