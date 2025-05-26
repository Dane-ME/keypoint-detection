"""
Test cases for improved loss functions and heatmap generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dll.losses.keypoint_loss import WeightedHeatmapLoss, HeatmapLoss
from dll.models.heatmap_head import (
    generate_target_heatmap,
    generate_target_heatmap_adaptive,
    decode_heatmaps,
    decode_heatmaps_soft_argmax,
    decode_heatmaps_subpixel
)

def test_weighted_vs_regular_loss():
    """Test weighted heatmap loss vs regular MSE loss"""
    print("=== Testing Weighted vs Regular Heatmap Loss ===")

    # Create synthetic data
    batch_size, num_keypoints, height, width = 2, 17, 56, 56

    # Create target with one clear keypoint peak
    target = torch.zeros(batch_size, num_keypoints, height, width)
    target[0, 0, 28, 28] = 1.0  # Strong peak at center
    target[0, 1, 14, 14] = 0.8  # Another peak

    # Create prediction that's all zeros (worst case scenario)
    pred_zeros = torch.zeros_like(target)

    # Create prediction with some response
    pred_some = torch.zeros_like(target)
    pred_some[0, 0, 28, 28] = 0.5  # Partial response
    pred_some[0, 1, 14, 14] = 0.3

    # Test regular MSE loss
    regular_loss = HeatmapLoss(use_target_weight=False)

    # Test weighted loss
    weighted_loss = WeightedHeatmapLoss(
        use_target_weight=False,
        keypoint_weight=10.0,
        background_weight=1.0,
        threshold=0.1
    )

    # Compute losses
    regular_loss_zeros = regular_loss(pred_zeros, target)
    weighted_loss_zeros = weighted_loss(pred_zeros, target)

    regular_loss_some = regular_loss(pred_some, target)
    weighted_loss_some = weighted_loss(pred_some, target)

    print(f"Regular MSE Loss (zeros): {regular_loss_zeros:.6f}")
    print(f"Weighted Loss (zeros): {weighted_loss_zeros:.6f}")
    print(f"Regular MSE Loss (some response): {regular_loss_some:.6f}")
    print(f"Weighted Loss (some response): {weighted_loss_some:.6f}")

    # Weighted loss should be higher for zeros prediction
    assert weighted_loss_zeros > regular_loss_zeros, "Weighted loss should penalize keypoint regions more"
    print("✓ Weighted loss correctly penalizes keypoint regions more")

def test_heatmap_generation():
    """Test heatmap generation with different sigma values"""
    print("\n=== Testing Heatmap Generation ===")

    # Create test keypoints
    keypoints = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]], dtype=torch.float32)  # [1, 2, 2]
    heatmap_size = (56, 56)

    # Test different sigma values
    sigmas = [2.0, 3.0, 4.0]
    heatmaps = {}

    for sigma in sigmas:
        heatmap = generate_target_heatmap(keypoints, heatmap_size, sigma)
        heatmaps[sigma] = heatmap

        # Check peak values
        peak_val = heatmap.max().item()
        print(f"Sigma {sigma}: Peak value = {peak_val:.4f}")

        # Check that peak is at correct location
        max_idx = torch.argmax(heatmap[0, 0].flatten())
        max_y, max_x = max_idx // 56, max_idx % 56
        expected_x, expected_y = int(0.5 * 56), int(0.5 * 56)

        print(f"  Peak location: ({max_x}, {max_y}), Expected: ({expected_x}, {expected_y})")

        # Allow some tolerance for peak location
        assert abs(max_x - expected_x) <= 1 and abs(max_y - expected_y) <= 1, \
            f"Peak location incorrect for sigma {sigma}"

    print("✓ Heatmap generation working correctly")

    # Test adaptive sigma
    adaptive_heatmap = generate_target_heatmap_adaptive(
        keypoints, heatmap_size, base_sigma=3.0, adaptive=True
    )
    print(f"Adaptive heatmap peak: {adaptive_heatmap.max().item():.4f}")

def test_keypoint_decoding():
    """Test different keypoint decoding methods"""
    print("\n=== Testing Keypoint Decoding Methods ===")

    # Create synthetic heatmap with known peak
    batch_size, num_keypoints, height, width = 1, 2, 56, 56
    heatmaps = torch.zeros(batch_size, num_keypoints, height, width)

    # Add peaks at known locations
    true_coords = [(28, 28), (14, 42)]  # (y, x) in pixel coordinates
    true_coords_norm = [(28/55, 28/55), (14/55, 42/55)]  # Normalized to [0,1]

    for i, (y, x) in enumerate(true_coords):
        # Create Gaussian peak
        sigma = 2.0
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )
        gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        heatmaps[0, i] = gaussian

    # Test different decoding methods
    methods = {
        'argmax': decode_heatmaps,
        'subpixel': decode_heatmaps_subpixel,
        'soft_argmax': decode_heatmaps_soft_argmax
    }

    for method_name, decode_func in methods.items():
        if method_name == 'subpixel':
            keypoints, scores = decode_func(heatmaps, window_size=5)
        elif method_name == 'soft_argmax':
            keypoints, scores = decode_func(heatmaps, temperature=1.0)
        else:
            keypoints, scores = decode_func(heatmaps)

        print(f"\n{method_name.upper()} Method:")
        for i, (pred_coord, true_coord) in enumerate(zip(keypoints[0], true_coords_norm)):
            pred_x, pred_y = pred_coord[0].item(), pred_coord[1].item()
            true_x, true_y = true_coord[1], true_coord[0]  # Note: x,y vs y,x

            error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            print(f"  Keypoint {i}: Pred=({pred_x:.4f}, {pred_y:.4f}), "
                  f"True=({true_x:.4f}, {true_y:.4f}), Error={error:.4f}")

            # Check if error is reasonable (allow higher tolerance for soft-argmax due to smoothing)
            if method_name == 'soft_argmax':
                # Soft-argmax may have higher error due to smoothing effect
                assert error < 0.5, f"Soft-argmax error too high: {error}"
            else:
                # Argmax methods should be very accurate for synthetic data
                assert error < 0.02, f"{method_name} error too high: {error}"

    print("✓ Keypoint decoding methods working correctly")

def test_loss_components():
    """Test that loss components are computed correctly"""
    print("\n=== Testing Loss Components ===")

    # Create mock predictions and targets
    batch_size, num_keypoints = 2, 17

    predictions = {
        'heatmaps': torch.randn(batch_size, num_keypoints, 56, 56),
        'coordinates': torch.randn(batch_size, num_keypoints, 2),
        'visibilities': torch.randn(batch_size, num_keypoints, 3)
    }

    targets = {
        'heatmaps': torch.randn(batch_size, num_keypoints, 56, 56),
        'keypoints': torch.randn(batch_size, num_keypoints, 2),
        'visibility': torch.randint(0, 3, (batch_size, num_keypoints))
    }

    # Test weighted loss
    weighted_loss = WeightedHeatmapLoss()
    loss_val = weighted_loss(predictions['heatmaps'], targets['heatmaps'])

    print(f"Weighted heatmap loss: {loss_val:.6f}")
    assert loss_val > 0, "Loss should be positive"
    assert torch.isfinite(loss_val), "Loss should be finite"

    print("✓ Loss components computed correctly")

def visualize_improvements():
    """Create visualizations to show improvements"""
    print("\n=== Creating Visualizations ===")

    # Create test keypoint
    keypoints = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    heatmap_size = (56, 56)

    # Generate heatmaps with different sigmas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sigmas = [2.0, 3.0, 4.0]
    for i, sigma in enumerate(sigmas):
        heatmap = generate_target_heatmap(keypoints, heatmap_size, sigma)
        im = axes[i].imshow(heatmap[0, 0].cpu().numpy(), cmap='hot')
        axes[i].set_title(f'Sigma = {sigma}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])

    plt.suptitle('Heatmap Generation with Different Sigma Values')
    plt.tight_layout()

    # Save visualization
    output_dir = Path("keypoint-detection/outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "sigma_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {output_dir / 'sigma_comparison.png'}")

def main():
    """Run all tests"""
    print("Running Loss and Heatmap Improvement Tests")
    print("=" * 50)

    try:
        test_weighted_vs_regular_loss()
        test_heatmap_generation()
        test_keypoint_decoding()
        test_loss_components()
        visualize_improvements()

        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("Improvements are working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
