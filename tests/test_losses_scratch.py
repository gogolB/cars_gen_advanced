# tests/test_losses_scratch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np 

# Assuming src is in PYTHONPATH or tests are run from project root
try:
    from src.losses_scratch import (
        compute_d_loss_logistic,
        compute_g_loss_nonsaturating,
        compute_r1_penalty,
        calculate_path_lengths # Import the new function
    )
    from src.models.stylegan2_networks_scratch import SynthesisNetwork # Import for PLR test
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.losses_scratch import (
        compute_d_loss_logistic,
        compute_g_loss_nonsaturating,
        compute_r1_penalty,
        calculate_path_lengths
    )
    from src.models.stylegan2_networks_scratch import SynthesisNetwork

# --- Dummy Discriminator Class (can be defined locally in tests) ---
class DummyD(nn.Module):
    def __init__(self, in_channels=1, resolution=32):
        super().__init__()
        self.resolution = resolution 
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.fc_in_features = 8 * resolution * resolution
        self.fc = nn.Linear(self.fc_in_features, 1)
    
    def forward(self, x):
        if x.shape[2] != self.resolution or x.shape[3] != self.resolution:
            pass
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = x.view(x.size(0), -1) 
        if x.shape[1] != self.fc_in_features:
            raise ValueError(f"Shape mismatch for FC layer in DummyD. Expected {self.fc_in_features} features, got {x.shape[1]}. Input image resolution might be different from DummyD initialized resolution.")
        return self.fc(x)

# --- Test Fixtures ---
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tests for Discriminator Logistic Loss ---
def test_d_loss_logistic_output_type_and_shape(device):
    d_real_logits = torch.randn(4, 1, device=device)
    d_fake_logits = torch.randn(4, 1, device=device)
    loss = compute_d_loss_logistic(d_real_logits, d_fake_logits)
    assert isinstance(loss, torch.Tensor), "D loss should be a Tensor"
    assert loss.ndim == 0, "D loss should be a scalar"
    assert not torch.isnan(loss).any(), "D loss should not be NaN"
    assert not torch.isinf(loss).any(), "D loss should not be Inf"

def test_d_loss_logistic_values(device):
    d_real_logits1 = torch.tensor([10.0, 10.0], device=device, dtype=torch.float32) 
    d_fake_logits1 = torch.tensor([-10.0, -10.0], device=device, dtype=torch.float32)
    loss1 = compute_d_loss_logistic(d_real_logits1, d_fake_logits1)
    assert loss1.item() < 0.01, f"Expected low D loss for perfect D, got {loss1.item()}"

    d_real_logits2 = torch.tensor([-10.0, -10.0], device=device, dtype=torch.float32)
    d_fake_logits2 = torch.tensor([10.0, 10.0], device=device, dtype=torch.float32)
    loss2 = compute_d_loss_logistic(d_real_logits2, d_fake_logits2)
    assert loss2.item() > 19.0, f"Expected high D loss for poor D, got {loss2.item()}"

    d_real_logits3 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    d_fake_logits3 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    loss3 = compute_d_loss_logistic(d_real_logits3, d_fake_logits3)
    expected_loss3_val = 2 * np.log(2.0)
    assert torch.isclose(loss3, torch.tensor(expected_loss3_val, device=device, dtype=loss3.dtype)), \
        f"Expected D loss ~{expected_loss3_val:.3f} for zero logits, got {loss3.item()}"


# --- Tests for Generator Non-Saturating Logistic Loss ---
def test_g_loss_nonsaturating_output_type_and_shape(device):
    d_fake_logits = torch.randn(4, 1, device=device)
    loss = compute_g_loss_nonsaturating(d_fake_logits)
    assert isinstance(loss, torch.Tensor), "G loss should be a Tensor"
    assert loss.ndim == 0, "G loss should be a scalar"
    assert not torch.isnan(loss).any(), "G loss should not be NaN"
    assert not torch.isinf(loss).any(), "G loss should not be Inf"

def test_g_loss_nonsaturating_values(device):
    d_fake_logits1 = torch.tensor([10.0, 10.0], device=device, dtype=torch.float32)
    loss1 = compute_g_loss_nonsaturating(d_fake_logits1)
    assert loss1.item() < 0.01, f"Expected low G loss when G fools D, got {loss1.item()}"

    d_fake_logits2 = torch.tensor([-10.0, -10.0], device=device, dtype=torch.float32)
    loss2 = compute_g_loss_nonsaturating(d_fake_logits2)
    assert loss2.item() > 9.0, f"Expected high G loss when G fails, got {loss2.item()}"

    d_fake_logits3 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    loss3 = compute_g_loss_nonsaturating(d_fake_logits3)
    expected_loss3_val = np.log(2.0)
    assert torch.isclose(loss3, torch.tensor(expected_loss3_val, device=device, dtype=loss3.dtype)), \
        f"Expected G loss ~{expected_loss3_val:.3f} for zero fake logits, got {loss3.item()}"


# --- Tests for R1 Gradient Penalty ---
def test_r1_penalty_output_type_and_shape(device):
    batch_size = 2
    img_channels = 1
    img_resolution = 16 
    
    test_D = DummyD(in_channels=img_channels, resolution=img_resolution).to(device)
    test_D.train() 

    real_images = torch.randn(batch_size, img_channels, img_resolution, img_resolution, device=device, requires_grad=True)
    d_real_logits_for_r1 = test_D(real_images)
    
    penalty = compute_r1_penalty(real_images, d_real_logits_for_r1, device=device)
    
    assert isinstance(penalty, torch.Tensor), "R1 penalty should be a Tensor"
    assert penalty.ndim == 0, "R1 penalty should be a scalar"
    assert not torch.isnan(penalty).any(), "R1 penalty should not be NaN"
    assert not torch.isinf(penalty).any(), "R1 penalty should not be Inf"
    assert penalty.item() >= 0, "R1 penalty should be non-negative"

def test_r1_penalty_requires_grad_on_input(device):
    batch_size = 2
    img_channels = 1
    img_resolution = 16
    
    test_D = DummyD(in_channels=img_channels, resolution=img_resolution).to(device)
    test_D.train()

    # Case 1: requires_grad = True (should work)
    real_images_grad = torch.randn(batch_size, img_channels, img_resolution, img_resolution, device=device, requires_grad=True)
    d_real_logits_for_r1_grad = test_D(real_images_grad)
    penalty_grad = compute_r1_penalty(real_images_grad, d_real_logits_for_r1_grad, device=device)
    assert penalty_grad is not None

    # Case 2: requires_grad = False 
    real_images_no_grad = torch.randn(batch_size, img_channels, img_resolution, img_resolution, device=device, requires_grad=False)
    # The compute_r1_penalty function itself will raise a ValueError.
    with pytest.raises(ValueError, match="Input 'real_images_for_r1' to compute_r1_penalty must have requires_grad=True."):
         _ = compute_r1_penalty(real_images_no_grad, None, device=device) # Pass None for logits as they won't be used


# --- New Test for Path Length Regularization ---
def test_calculate_path_lengths(device):
    batch_size = 2
    w_dim = 64
    img_resolution = 16
    img_channels = 1

    # Create a dummy synthesis network
    G_synth_test = SynthesisNetwork(
        w_dim=w_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        channel_base=512, # Use smaller values for test
        channel_max=64
    ).to(device)
    
    # Create dummy `ws` input that requires gradients
    ws_test = torch.randn(batch_size, G_synth_test.num_ws, w_dim, device=device, requires_grad=True)
    
    # Calculate path lengths
    path_lengths = calculate_path_lengths(ws_test, G_synth_test)
    
    # Assertions
    assert isinstance(path_lengths, torch.Tensor), "Path lengths should be a Tensor"
    assert path_lengths.shape == (batch_size,), f"Path lengths should have shape [{batch_size}], got {path_lengths.shape}"
    assert not torch.isnan(path_lengths).any(), "Path lengths should not be NaN"
    assert not torch.isinf(path_lengths).any(), "Path lengths should not be Inf"
    assert (path_lengths >= 0).all(), "Path lengths should be non-negative"

    # Test that it fails if ws does not require grad
    ws_no_grad = torch.randn(batch_size, G_synth_test.num_ws, w_dim, device=device, requires_grad=False)
    with pytest.raises(ValueError, match="Input 'ws' to calculate_path_lengths must have requires_grad=True."):
        _ = calculate_path_lengths(ws_no_grad, G_synth_test)

