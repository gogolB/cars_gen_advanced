# tests/test_augment_scratch.py

import torch
import pytest
from omegaconf import OmegaConf

# This try-except block helps ensure the test file can be run from the project root
# where `pytest` is typically invoked.
try:
    from src.augment_scratch import AugmentPipe
except ImportError:
    import sys
    import os
    # Add the project root directory (which contains 'src') to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.augment_scratch import AugmentPipe

# --- Test Fixtures ---

@pytest.fixture
def device():
    """Provides the appropriate torch device (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_images(device):
    """Creates a batch of dummy images with a distinct pattern for testing."""
    batch_size, img_channels, img_size = 4, 1, 32
    images = torch.zeros(batch_size, img_channels, img_size, img_size, device=device, dtype=torch.float32)
    # Create a simple quadrant pattern for easy visual/numerical verification
    images[:, :, :img_size//2, :img_size//2] = 0.25  # Top-left
    images[:, :, :img_size//2, img_size//2:] = 0.50  # Top-right
    images[:, :, img_size//2:, :img_size//2] = 0.75  # Bottom-left
    images[:, :, img_size//2:, img_size//2:] = 1.0   # Bottom-right
    # Scale to a typical [-1, 1] range
    return (images * 2) - 1

# --- Tests ---

def test_augment_pipe_initialization(device):
    """Tests that the AugmentPipe initializes correctly from a config."""
    cfg = OmegaConf.create({
        "p_xflip": 0.5, "p_rotate90": 0.3,
        "p_brightness": 0.8, "brightness_range": [-0.1, 0.1],
        "p_contrast": 0.7, "contrast_range": [0.9, 1.1]
    })
    pipe = AugmentPipe(cfg).to(device)
    assert pipe.p_xflip == 0.5
    assert pipe.p_rotate90 == 0.3
    assert pipe.p_brightness == 0.8
    assert pipe.brightness_range == [-0.1, 0.1]
    assert pipe.p_contrast == 0.7
    assert pipe.contrast_range == [0.9, 1.1]
    print("\nPASSED: test_augment_pipe_initialization")

def test_augment_pipe_no_op(dummy_images, device):
    """Tests that no changes occur if all probabilities are zero."""
    cfg = OmegaConf.create({ "p_xflip": 0.0, "p_rotate90": 0.0, "p_brightness": 0.0, "p_contrast": 0.0 })
    pipe = AugmentPipe(cfg).to(device)
    augmented = pipe(dummy_images)
    assert torch.equal(dummy_images, augmented), "Output should be identical if all probabilities are 0."
    print("PASSED: test_augment_pipe_no_op")

def test_augment_pipe_xflip(dummy_images, device):
    """Tests horizontal flip with probability 1.0."""
    cfg = OmegaConf.create({ "p_xflip": 1.0 })
    pipe = AugmentPipe(cfg).to(device)
    augmented = pipe(dummy_images)
    expected_flipped = torch.flip(dummy_images, dims=[-1])
    assert torch.equal(augmented, expected_flipped), "Output should be horizontally flipped."
    print("PASSED: test_augment_pipe_xflip")

def test_augment_pipe_rotate90(dummy_images, device):
    """Tests that 90-degree rotation changes the image."""
    cfg = OmegaConf.create({ "p_rotate90": 1.0 })
    pipe = AugmentPipe(cfg).to(device)
    augmented = pipe(dummy_images)
    assert augmented.shape == dummy_images.shape
    # With p=1.0, it must be rotated (k=1,2, or 3) and thus different from the original.
    assert not torch.equal(augmented, dummy_images), "Output should be rotated and thus different from the original."
    print("PASSED: test_augment_pipe_rotate90")

def test_augment_pipe_brightness(dummy_images, device):
    """Tests brightness adjustment with a fixed deterministic shift."""
    fixed_shift = 0.25
    cfg = OmegaConf.create({ 
        "p_brightness": 1.0, "brightness_range": [fixed_shift, fixed_shift]
    })
    pipe = AugmentPipe(cfg).to(device)
    augmented = pipe(dummy_images)
    # The pipe clamps the output, so we must clamp the expected output too.
    expected_brightened = torch.clamp(dummy_images + fixed_shift, -1.0, 1.0)
    assert torch.allclose(augmented, expected_brightened), "Output should be brightened by a fixed amount."
    print("PASSED: test_augment_pipe_brightness")

def test_augment_pipe_contrast(dummy_images, device):
    """Tests contrast adjustment with a fixed deterministic factor."""
    fixed_factor = 1.5
    cfg = OmegaConf.create({ 
        "p_contrast": 1.0, "contrast_range": [fixed_factor, fixed_factor]
    })
    pipe = AugmentPipe(cfg).to(device)
    augmented = pipe(dummy_images)
    # The pipe clamps the output.
    expected_contrasted = torch.clamp(dummy_images * fixed_factor, -1.0, 1.0)
    assert torch.allclose(augmented, expected_contrasted), "Output should have contrast adjusted by a fixed factor."
    print("PASSED: test_augment_pipe_contrast")

def test_output_properties(dummy_images, device):
    """Ensures output properties match input properties after stochastic augmentations."""
    cfg = OmegaConf.create({ "p_xflip": 0.5, "p_rotate90": 0.5, "p_brightness": 0.5, "p_contrast": 0.5 })
    pipe = AugmentPipe(cfg).to(device)
    augmented = pipe(dummy_images)
    assert augmented.shape == dummy_images.shape, "Output shape should match input shape."
    assert augmented.dtype == dummy_images.dtype, "Output dtype should match input dtype."
    assert augmented.device == dummy_images.device, "Output device should match input device."
    print("PASSED: test_output_properties")
