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
    """Fixed R1 gradient penalty calculation"""
    # Compute gradients
    grads = torch.autograd.grad(
        outputs=real_logits.sum(), 
        inputs=real_images, 
        create_graph=True, 
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # FIXED: Proper gradient penalty calculation
    # Sum over spatial dimensions, then take mean over batch
    grad_penalty = grads.pow(2).sum(dim=[1, 2, 3]).mean()
    
    return grad_penalty


def calculate_path_lengths(ws, synthesis_net):
    """Fixed path length regularization"""
    batch_size = ws.shape[0]
    
    # FIXED: Use proper noise scaling
    # Generate small perturbations in W space
    noise_std = ws.std() * 0.1  # 10% of W's standard deviation
    pl_noise = torch.randn_like(ws) * noise_std
    
    # Compute images for original and perturbed latents
    images_orig = synthesis_net(ws)
    images_pert = synthesis_net(ws + pl_noise)
    
    # FIXED: Compute perceptual path length properly
    # Calculate pixel-space distance
    pixel_dist = (images_orig - images_pert).pow(2).sum(dim=[1, 2, 3])
    
    # Normalize by noise magnitude
    path_lengths = pixel_dist / (pl_noise.pow(2).sum(dim=[1, 2]) + 1e-8)
    
    return path_lengths.sqrt()

