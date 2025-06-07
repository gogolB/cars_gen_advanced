# src/losses_scratch.py
import torch
import torch.nn.functional as F

def compute_d_loss_logistic(real_logits, fake_logits):
    """Logistic loss for the discriminator."""
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss

def compute_g_loss_nonsaturating(fake_logits):
    """Non-saturating logistic loss for the generator."""
    return F.softplus(-fake_logits).mean()

def compute_r1_penalty(real_images, real_logits):
    """R1 gradient penalty for the discriminator."""
    # Note: device argument is removed as it can be inferred from the tensors.
    grads = torch.autograd.grad(
        outputs=real_logits.sum(), 
        inputs=real_images, 
        create_graph=True, 
        only_inputs=True
    )[0]
    r1_penalty = grads.square().sum(dim=[1, 2, 3])
    return r1_penalty.mean()

def calculate_path_lengths(ws, synthesis_net):
    """
    Calculates the path length penalty for the generator.
    This encourages a smoother latent space.
    """
    # Get random perceptual path length variations
    pl_noise = torch.randn_like(ws) / (ws.shape[2] ** 0.5)
    
    # Get the generator's output for the original and noised latents
    # CORRECTED: Removed the incorrect tuple unpacking `_,`. The synthesis network
    # returns a single tensor.
    fake_images_original = synthesis_net(ws)
    fake_images_noised = synthesis_net(ws + pl_noise)

    # Calculate the perceptual distance (L2 norm of the difference in images)
    # The sum is over the C, H, W dimensions, leaving a per-image distance.
    # CORRECTED: Removed the redundant and numerically unstable division by batch size.
    # The .mean() operation on the penalty is handled later in the training step.
    distances = (fake_images_original - fake_images_noised).square().sum(dim=[1, 2, 3])
    
    return distances

