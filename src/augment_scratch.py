# src/augment_scratch.py
# From-scratch implementation of a basic AugmentPipe for StyleGAN2,
# refactored to support Adaptive Discriminator Augmentation (ADA).

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

class AugmentPipe(torch.nn.Module):
    """
    Applies a series of differentiable augmentations controlled by a master probability `p`.
    This version is designed to be used with the ADA feedback loop.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # The 'recipe' of augmentations to apply if the pipeline is activated.
        # These are the probabilities for each op, given that p > 0.
        self.p_xflip = self.cfg.get('p_xflip', 0.0)
        self.p_rotate90 = self.cfg.get('p_rotate90', 0.0)
        self.p_brightness = self.cfg.get('p_brightness', 0.0)
        self.p_contrast = self.cfg.get('p_contrast', 0.0)
        
        # Parameter ranges for color augmentations
        self.brightness_range = self.cfg.get('brightness_range', [-0.2, 0.2])
        self.contrast_range = self.cfg.get('contrast_range', [0.8, 1.2])

        print("AugmentPipe (from-scratch, ADA-ready) initialized with recipe:")
        print(f"  p_xflip: {self.p_xflip}")
        print(f"  p_rotate90: {self.p_rotate90}")
        print(f"  p_brightness: {self.p_brightness}, range: {self.brightness_range}")
        print(f"  p_contrast: {self.p_contrast}, range: {self.contrast_range}")

    def forward(self, images: torch.Tensor, p: float = 0.0) -> torch.Tensor:
        """
        Apply a sequence of augmentations to a batch of images.

        Args:
            images (torch.Tensor): Input images, shape (N, C, H, W). Assumed to be in [-1, 1] range.
            p (float): The master probability of applying the augmentation pipeline to an image.
        
        Returns:
            torch.Tensor: Augmented images.
        """
        if p == 0.0:
            return images # No need to clone if no ops are applied.

        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        # Generate a mask to determine which images in the batch get augmented.
        # Shape: [N]
        pipeline_active = torch.rand(batch_size, device=device) < p

        # If no images are selected for augmentation in this batch, return early.
        if not pipeline_active.any():
            return images

        # Work on a clone to avoid modifying the original tensor in-place.
        x = images.clone()
        
        # Select the subset of images to be augmented.
        images_to_augment = x[pipeline_active]
        
        # --- Apply the augmentation recipe to the selected subset ---

        # 1. Horizontal Flip (xflip)
        if self.p_xflip > 0:
            # Decide which of the selected images get this specific augmentation.
            op_active = torch.rand(images_to_augment.shape[0], device=device) < self.p_xflip
            if op_active.any():
                images_to_augment[op_active] = torch.flip(images_to_augment[op_active], dims=[-1])

        # 2. 90-degree Rotations (rotate90)
        if self.p_rotate90 > 0:
            op_active = torch.rand(images_to_augment.shape[0], device=device) < self.p_rotate90
            indices_to_rotate = torch.where(op_active)[0]
            # This op is applied with a loop as k is random for each image.
            for i in indices_to_rotate:
                k = torch.randint(1, 4, (1,)).item()
                images_to_augment[i] = torch.rot90(images_to_augment[i], k, dims=[-2, -1])

        # 3. Brightness Adjustment (additive)
        if self.p_brightness > 0:
            op_active = torch.rand(images_to_augment.shape[0], device=device) < self.p_brightness
            if op_active.any():
                brightness_min, brightness_max = self.brightness_range
                # Generate shifts only for the images where this op is active.
                shifts = torch.rand(op_active.sum(), 1, 1, 1, device=device, dtype=dtype) * (brightness_max - brightness_min) + brightness_min
                images_to_augment[op_active] += shifts

        # 4. Contrast Adjustment (multiplicative)
        if self.p_contrast > 0:
            op_active = torch.rand(images_to_augment.shape[0], device=device) < self.p_contrast
            if op_active.any():
                contrast_min, contrast_max = self.contrast_range
                # Generate factors only for the images where this op is active.
                factors = torch.rand(op_active.sum(), 1, 1, 1, device=device, dtype=dtype) * (contrast_max - contrast_min) + contrast_min
                images_to_augment[op_active] *= factors
        
        # Place the augmented images back into the original tensor.
        x[pipeline_active] = images_to_augment
        
        # Clamp output to the expected [-1, 1] range for StyleGAN.
        x = torch.clamp(x, -1.0, 1.0)

        return x

if __name__ == '__main__':
    print("\nRunning basic tests for ADA-ready from-scratch AugmentPipe...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_cfg = OmegaConf.create({
        "p_xflip": 0.5, "p_rotate90": 0.5,
        "p_brightness": 0.5, "brightness_range": [-0.3, 0.3],
        "p_contrast": 0.5, "contrast_range": [0.7, 1.3]
    })

    pipe = AugmentPipe(dummy_cfg).to(device)
    dummy_images = torch.randn(16, 1, 32, 32, device=device)
    
    # Test with p=0.0 (should do nothing)
    augmented_images_p0 = pipe(dummy_images, p=0.0)
    assert torch.equal(dummy_images, augmented_images_p0), "p=0.0 should not alter the images."
    print("Test with p=0.0 passed.")

    # Test with p=1.0 (should alter the images)
    augmented_images_p1 = pipe(dummy_images, p=1.0)
    assert not torch.equal(dummy_images, augmented_images_p1), "p=1.0 should alter the images."
    assert augmented_images_p1.shape == dummy_images.shape, "Output shape mismatch for p=1.0."
    print("Test with p=1.0 passed.")

    # Test with p=0.5 (should alter some images)
    augmented_images_p05 = pipe(dummy_images, p=0.5)
    assert not torch.equal(dummy_images, augmented_images_p05), "p=0.5 should alter the images."
    assert augmented_images_p05.shape == dummy_images.shape, "Output shape mismatch for p=0.5."
    print("Test with p=0.5 passed.")
    print("\nAugmentPipe basic ADA-ready tests completed.")