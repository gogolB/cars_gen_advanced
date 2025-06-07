# src/losses_scratch.py
# From-scratch implementations of StyleGAN2 loss components.

import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_d_loss_logistic(d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the logistic loss for the discriminator.
    Aims to make D output 1 for reals and 0 for fakes.
    This can be numerically stabilized using softplus.

    Args:
        d_real_logits (torch.Tensor): Discriminator's output logits for real images.
        d_fake_logits (torch.Tensor): Discriminator's output logits for fake images.

    Returns:
        torch.Tensor: The mean discriminator logistic loss.
    """
    loss_real = F.softplus(-d_real_logits)
    loss_fake = F.softplus(d_fake_logits)
    d_loss = (loss_real + loss_fake).mean()
    return d_loss

def compute_g_loss_nonsaturating(d_fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the non-saturating logistic loss for the generator.
    Aims to make D output 1 for fakes (i.e., fool D).
    This can be numerically stabilized using softplus.

    Args:
        d_fake_logits (torch.Tensor): Discriminator's output logits for fake images.

    Returns:
        torch.Tensor: The mean generator non-saturating logistic loss.
    """
    g_loss = F.softplus(-d_fake_logits).mean()
    return g_loss

def compute_r1_penalty(real_images_for_r1: torch.Tensor, 
                       d_real_logits_for_r1: torch.Tensor, 
                       device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Computes the R1 gradient penalty for the discriminator.
    Penalizes the L2 norm of gradients of D's output w.r.t. real (potentially augmented) inputs.

    Args:
        real_images_for_r1 (torch.Tensor): Real images that were fed to D. MUST have requires_grad=True.
        d_real_logits_for_r1 (torch.Tensor): Discriminator's output logits for real_images_for_r1.
        device (torch.device): The device to perform calculations on.

    Returns:
        torch.Tensor: The R1 penalty term (raw, mean of sum of squared gradients per sample).
    """
    if not real_images_for_r1.requires_grad:
        raise ValueError("Input 'real_images_for_r1' to compute_r1_penalty must have requires_grad=True.")

    grad_real = torch.autograd.grad(
        outputs=d_real_logits_for_r1.sum(), 
        inputs=real_images_for_r1, 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True   
    )[0] 
    
    grad_penalty_per_sample = grad_real.square().sum(dim=[1,2,3]) 
    grad_penalty = grad_penalty_per_sample.mean() 
    
    return grad_penalty

def calculate_path_lengths(ws: torch.Tensor, 
                           G_synthesis: torch.nn.Module) -> torch.Tensor:
    """
    Calculates the path length for Path Length Regularization (PLR).
    This computes || J_w^T * y || where J_w is the Jacobian d(G(w))/dw
    and y is a random normalized vector.

    Args:
        ws (torch.Tensor): Style vectors `w` of shape [N, num_ws, w_dim].
        G_synthesis (torch.nn.Module): The synthesis network of the generator.

    Returns:
        torch.Tensor: A tensor of path lengths for each sample in the batch, shape [N].
    """
    # Generate random images `y` from a normal distribution.
    # These serve as the random projection vectors.
    num_pixels = G_synthesis.img_resolution * G_synthesis.img_resolution
    y = torch.randn(
        ws.shape[0], # batch_size
        G_synthesis.img_channels, 
        G_synthesis.img_resolution, 
        G_synthesis.img_resolution,
        device=ws.device
    )
    
    # Normalize `y` to make the penalty resolution-independent.
    # The normalization factor is sqrt(num_pixels).
    y_normalized = y / (num_pixels ** 0.5)

    # Generate images from w codes.
    # The synthesis network needs requires_grad=True for its inputs (ws) for this to work.
    # The caller (LightningModule) should handle this.
    if not ws.requires_grad:
        raise ValueError("Input 'ws' to calculate_path_lengths must have requires_grad=True.")
        
    fake_images = G_synthesis(ws)
    
    # Compute the Jacobian-vector product: J_w^T * y
    # This is done efficiently by computing the gradient of (output * y).sum() w.r.t. ws.
    grad_ws = torch.autograd.grad(
        outputs=(fake_images * y_normalized).sum(),
        inputs=ws,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0] # Shape: [N, num_ws, w_dim]

    # Calculate the L2 norm of the resulting gradients for each sample.
    # This norm is the path length.
    path_lengths = (grad_ws.square().sum(dim=[1, 2]) + 1e-8).sqrt() # Sum over num_ws and w_dim

    return path_lengths


if __name__ == '__main__':
    # --- Basic Tests for Loss Functions ---
    print("Running basic tests for loss functions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ... (existing tests for d_loss, g_loss, r1_penalty) ...
    # Test D_loss_logistic
    d_real_logits_test = torch.tensor([2.0, -1.0], device=device) 
    d_fake_logits_test = torch.tensor([-3.0, 1.5], device=device) 
    d_loss = compute_d_loss_logistic(d_real_logits_test, d_fake_logits_test)
    print(f"D_loss: {d_loss.item():.4f}")

    # Test G_loss_nonsaturating
    g_fake_logits_test = torch.tensor([-3.0, 2.5], device=device) 
    g_loss = compute_g_loss_nonsaturating(g_fake_logits_test)
    print(f"G_loss: {g_loss.item():.4f}")

    # Test R1_penalty
    class DummyD(nn.Module):
        def __init__(self, res): 
            super().__init__(); 
            self.fc = nn.Linear(res*res, 1)
        def forward(self, x): return self.fc(x.view(x.shape[0], -1))

    dummy_D = DummyD(32).to(device)
    test_real_images = torch.randn(2, 1, 32, 32, device=device, requires_grad=True)
    test_d_real_logits_for_r1 = dummy_D(test_real_images)
    r1_penalty_val = compute_r1_penalty(test_real_images, test_d_real_logits_for_r1, device=device)
    print(f"R1 penalty value: {r1_penalty_val.item():.4f}")

    # --- Test for Path Length Regularization ---
    print("\nTesting Path Length calculation...")
    
    # Import G_synthesis from our scratch networks for the test
    from src.models.stylegan2_networks_scratch import SynthesisNetwork

    BATCH_SIZE = 2
    W_DIM = 64
    IMG_RESOLUTION = 32
    IMG_CHANNELS = 1

    # Create a dummy synthesis network
    G_synth_test = SynthesisNetwork(
        w_dim=W_DIM,
        img_resolution=IMG_RESOLUTION,
        img_channels=IMG_CHANNELS,
        channel_base=512, # Use smaller values for test
        channel_max=64
    ).to(device)
    
    # Create dummy `ws` input that requires gradients
    ws_test = torch.randn(BATCH_SIZE, G_synth_test.num_ws, W_DIM, device=device, requires_grad=True)
    
    path_lengths = calculate_path_lengths(ws_test, G_synth_test)
    
    print(f"Calculated path lengths shape: {path_lengths.shape}")
    print(f"Path lengths: {path_lengths.cpu().detach().numpy()}")
    
    assert path_lengths.shape == (BATCH_SIZE,), "Path lengths should have shape [N]"
    assert not torch.isnan(path_lengths).any(), "Path lengths should not be NaN"
    assert not torch.isinf(path_lengths).any(), "Path lengths should not be Inf"
    
    print("Path Length calculation test completed.")

