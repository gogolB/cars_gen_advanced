# src/augment_scratch.py
# From-scratch implementation of a basic AugmentPipe for StyleGAN2.

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

class AugmentPipe(torch.nn.Module):
    """
    Applies a series of differentiable augmentations.
    This is a simplified, from-scratch version inspired by StyleGAN2-ADA.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Extract probabilities and ranges for each augmentation, providing defaults.
        # Geometric augmentations
        self.p_xflip = self.cfg.get('p_xflip', 0.0)
        self.p_rotate90 = self.cfg.get('p_rotate90', 0.0)
        
        # Color augmentations
        self.p_brightness = self.cfg.get('p_brightness', 0.0)
        self.brightness_range = self.cfg.get('brightness_range', [-0.2, 0.2])
        
        self.p_contrast = self.cfg.get('p_contrast', 0.0)
        self.contrast_range = self.cfg.get('contrast_range', [0.8, 1.2])

        print("AugmentPipe (from-scratch) initialized with:")
        print(f"  p_xflip: {self.p_xflip}")
        print(f"  p_rotate90: {self.p_rotate90}")
        print(f"  p_brightness: {self.p_brightness}, range: {self.brightness_range}")
        print(f"  p_contrast: {self.p_contrast}, range: {self.contrast_range}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply a sequence of augmentations to a batch of images.
        The order is fixed: flip -> rotate -> brightness -> contrast.

        Args:
            images (torch.Tensor): Input images, shape (N, C, H, W). Assumed to be in [-1, 1] range.
        
        Returns:
            torch.Tensor: Augmented images.
        """
        # Work on a clone to avoid modifying the original tensor.
        x = images.clone()

        # 1. Horizontal Flip (xflip)
        if self.p_xflip > 0 and torch.rand(1).item() < self.p_xflip:
            x = torch.flip(x, dims=[-1])

        # 2. 90-degree Rotations (rotate90)
        if self.p_rotate90 > 0 and torch.rand(1).item() < self.p_rotate90:
            # Randomly choose k from 1, 2, 3 for 90, 180, 270 deg rotations.
            k = torch.randint(1, 4, (1,)).item() 
            x = torch.rot90(x, k=k, dims=[-2, -1])

        # 3. Brightness Adjustment (additive)
        if self.p_brightness > 0 and torch.rand(1).item() < self.p_brightness:
            brightness_min, brightness_max = self.brightness_range
            # Create shifts of shape (N, 1, 1, 1) for broadcasting.
            shifts = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) * (brightness_max - brightness_min) + brightness_min
            x = x + shifts

        # 4. Contrast Adjustment (multiplicative)
        if self.p_contrast > 0 and torch.rand(1).item() < self.p_contrast:
            contrast_min, contrast_max = self.contrast_range
            # Create factors of shape (N, 1, 1, 1).
            factors = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) * (contrast_max - contrast_min) + contrast_min
            x = x * factors
        
        # Clamp output to the expected [-1, 1] range for StyleGAN.
        x = torch.clamp(x, -1.0, 1.0)

        return x

if __name__ == '__main__':
    print("\nRunning basic tests for from-scratch AugmentPipe...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_cfg = OmegaConf.create({
        "p_xflip": 0.5, "p_rotate90": 0.5,
        "p_brightness": 0.5, "brightness_range": [-0.3, 0.3],
        "p_contrast": 0.5, "contrast_range": [0.7, 1.3]
    })

    pipe = AugmentPipe(dummy_cfg).to(device)
    dummy_images = torch.randn(4, 1, 32, 32, device=device)
    augmented_images = pipe(dummy_images)

    assert augmented_images.shape == dummy_images.shape, "Output shape mismatch."
    print("AugmentPipe basic instantiation and forward pass test completed.")
