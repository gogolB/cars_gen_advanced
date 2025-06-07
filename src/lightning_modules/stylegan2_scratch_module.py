# src/lightning_modules/stylegan2_scratch_module.py

import os
import sys 
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torchvision

# Import from-scratch components
from src.models.stylegan2_networks_scratch import Generator, Discriminator
from src.losses_scratch import (
    compute_d_loss_logistic,
    compute_g_loss_nonsaturating,
    compute_r1_penalty,
    calculate_path_lengths
)
from src.augment_scratch import AugmentPipe


class StyleGAN2ScratchLightningModule(pl.LightningModule):
    def __init__(self,
                 model_cfg: DictConfig,    
                 training_cfg: DictConfig, 
                 data_cfg: DictConfig      
                ):
        super().__init__()
        self.save_hyperparameters(logger=True) 

        # --- Network Initialization ---
        print("Initializing From-Scratch Generator...")
        self.G = Generator(**self.hparams.model_cfg.generator_kwargs)
        
        print("Initializing From-Scratch Discriminator...")
        self.D = Discriminator(**self.hparams.model_cfg.discriminator_kwargs)

        print("Initializing G_ema...")
        self.G_ema = copy.deepcopy(self.G).eval()
        for param in self.G_ema.parameters():
            param.requires_grad = False
        
        # --- EMA Parameters ---
        self.ema_kimg = self.hparams.training_cfg.get('ema_kimg', 10.0) 
        self.ema_rampup_ratio = self.hparams.training_cfg.get('ema_rampup_ratio', 0.05) 
        self.ema_beta = self.hparams.training_cfg.get('ema_beta', 0.999) 
        self.ema_nimg = self.ema_kimg * 1000 
        self.ema_rampup = None 

        # --- Loss and Regularization Parameters ---
        self.r1_gamma = self.hparams.training_cfg.get('r1_gamma', 10.0)
        self.d_reg_interval = self.hparams.training_cfg.get('d_reg_interval', 16)
        
        self.pl_weight = self.hparams.training_cfg.get('pl_weight', 0.0) 
        self.g_reg_interval = self.hparams.training_cfg.get('g_reg_interval', 4)
        self.pl_decay = self.hparams.training_cfg.get('pl_decay', 0.01)
        if self.pl_weight > 0: 
            self.register_buffer('pl_mean', torch.zeros([]))

        # --- AugmentPipe Initialization ---
        if self.hparams.model_cfg.get('augment_pipe_kwargs'):
            print("Initializing AugmentPipe (from-scratch, ADA-ready)...")
            # CORRECTED: Pass the config object directly, without unpacking using **
            self.augment_pipe = AugmentPipe(self.hparams.model_cfg.augment_pipe_kwargs)
        else:
            self.augment_pipe = None
            print("AugmentPipe not configured, augmentations will be skipped.")
            
        # --- Adaptive Discriminator Augmentation (ADA) State ---
        self.ada_enabled = self.augment_pipe is not None and self.hparams.training_cfg.get('ada_kwargs') is not None
        if self.ada_enabled:
            print("ADA is enabled. Initializing ADA state...")
            ada_kwargs = self.hparams.training_cfg.ada_kwargs
            self.ada_target = ada_kwargs.get('target_rt', 0.6)
            self.ada_interval_kimg = ada_kwargs.get('interval_kimg', 4)
            self.ada_speed_kimg = ada_kwargs.get('speed_kimg', 500)
            
            self.register_buffer('ada_p', torch.zeros([]))
            self.register_buffer('ada_stats', torch.zeros([]))
            
            self.ada_interval = self.ada_interval_kimg * 1000
            self.batch_size = self.hparams.data_cfg.batch_size 
            self.ada_adjust_speed = self.batch_size * self.ada_interval / (self.ada_speed_kimg * 1000)
            print(f"  ADA Target r_t: {self.ada_target}")
            print(f"  ADA Update Interval: {self.ada_interval_kimg} kimg ({self.ada_interval} images)")
            print(f"  ADA Adjustment Speed (kimg): {self.ada_speed_kimg}")

        # --- Training State Tracking ---
        self.cur_nimg = 0
        self.last_ada_update_nimg = 0
        self.automatic_optimization = False 

        print("StyleGAN2ScratchLightningModule initialized.")

    def configure_optimizers(self):
        g_params = list(self.G.parameters()) 
        d_params = list(self.D.parameters())

        g_lr = self.hparams.training_cfg.g_lr
        d_lr = self.hparams.training_cfg.d_lr
        adam_betas = tuple(self.hparams.training_cfg.adam_betas) 
        adam_eps = self.hparams.training_cfg.adam_eps

        opt_g = torch.optim.Adam(g_params, lr=g_lr, betas=adam_betas, eps=adam_eps)
        opt_d = torch.optim.Adam(d_params, lr=d_lr, betas=adam_betas, eps=adam_eps)
        
        print("Optimizers configured.")
        return opt_g, opt_d

    def on_train_start(self):
        # EMA ramp-up calculation
        if self.ema_rampup_ratio is not None and self.ema_rampup_ratio > 0 and \
           hasattr(self.hparams.training_cfg, 'total_kimg') and self.hparams.training_cfg.total_kimg > 0:
            total_nimg = self.hparams.training_cfg.total_kimg * 1000
            self.ema_nimg = total_nimg * self.ema_rampup_ratio
        else: 
            self.ema_nimg = self.ema_kimg * 1000
        
        # Move buffers to correct device
        if self.pl_weight > 0: self.pl_mean = self.pl_mean.to(self.device)
        if self.ada_enabled:
            self.ada_p = self.ada_p.to(self.device)
            self.ada_stats = self.ada_stats.to(self.device)
        if self.augment_pipe is not None:
            self.augment_pipe = self.augment_pipe.to(self.device)

    def forward(self, z, c=None, truncation_psi=0.7, noise_mode='random', return_ws=False):
        # Use G_ema for inference
        return self.G_ema(z, c=c, truncation_psi=truncation_psi, noise_mode=noise_mode, return_ws=return_ws)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers() 
        
        real_images, _ = batch 
        real_images = real_images.to(self.device).to(torch.float32)

        # Determine current augmentation probability 'p'
        current_p = self.ada_p.item() if self.ada_enabled else 0.0

        # --- Train Discriminator ---
        self.G.requires_grad_(False)
        self.D.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)

        # Generate fake images for D
        z_for_d = torch.randn(real_images.shape[0], self.G.z_dim, device=self.device)
        with torch.no_grad(): 
            fake_images_d = self.G(z_for_d) 
        
        # Apply augmentations (controlled by current_p) to both real and fake images
        real_images_for_d = self.augment_pipe(real_images, p=current_p) if self.augment_pipe else real_images
        fake_images_for_d = self.augment_pipe(fake_images_d.detach(), p=current_p) if self.augment_pipe else fake_images_d.detach()
        
        d_real_logits = self.D(real_images_for_d) 
        d_fake_logits = self.D(fake_images_for_d) 

        # --- ADA: Update overfitting metric (r_t) ---
        if self.ada_enabled:
            signs = d_real_logits.sign().detach()
            self.ada_stats.copy_(signs.mean().lerp(self.ada_stats, 0.999))
            
        d_loss = compute_d_loss_logistic(d_real_logits, d_fake_logits)
        self.log('d_loss/main', d_loss, on_step=True, prog_bar=True)

        # R1 Regularization (applied to augmented real images)
        if self.r1_gamma > 0 and (self.global_step % self.d_reg_interval == 0):
            real_images_for_r1 = real_images_for_d.detach().requires_grad_(True) 
            d_real_logits_for_r1 = self.D(real_images_for_r1)
            r1_penalty = compute_r1_penalty(real_images_for_r1, d_real_logits_for_r1) * (self.r1_gamma / 2) * self.d_reg_interval
            if not torch.isnan(r1_penalty).any():
                d_loss += r1_penalty
                self.log('d_loss/r1_penalty', r1_penalty, on_step=True)
        
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Train Generator ---
        self.G.requires_grad_(True)
        self.D.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)

        # Generate fake images for G
        z_for_g = torch.randn(real_images.shape[0], self.G.z_dim, device=self.device)
        fake_images_g, ws_for_g_loss = self.G(z_for_g, return_ws=True) 
        
        # Apply augmentations (controlled by current_p) to fake images
        fake_images_g_for_d = self.augment_pipe(fake_images_g, p=current_p) if self.augment_pipe else fake_images_g
        
        d_fake_logits_g = self.D(fake_images_g_for_d) 
        g_loss = compute_g_loss_nonsaturating(d_fake_logits_g)
        self.log('g_loss/main', g_loss, on_step=True, prog_bar=True)

        # Path Length Regularization (PLR)
        if self.pl_weight > 0 and (self.global_step % self.g_reg_interval == 0):
            pl_batch_size = max(1, real_images.shape[0] // 2)
            pl_z = torch.randn(pl_batch_size, self.G.z_dim, device=self.device)
            ws_single = self.G.mapping(pl_z)
            ws_for_plr = ws_single.unsqueeze(1).repeat(1, self.G.num_ws, 1)
            path_lengths = calculate_path_lengths(ws_for_plr, self.G.synthesis)
            pl_penalty = (path_lengths - self.pl_mean).square()
            self.pl_mean.copy_(path_lengths.mean().detach().lerp(self.pl_mean, self.pl_decay))
            pl_loss = pl_penalty.mean() * self.pl_weight * self.g_reg_interval
            if not torch.isnan(pl_loss).any():
                g_loss += pl_loss
                self.log('g_loss/pl_penalty', pl_loss, on_step=True)
                self.log('g_loss/pl_mean', self.pl_mean, on_step=True)

        self.manual_backward(g_loss)
        opt_g.step()

        # --- Update Training State (cur_nimg, EMA) ---
        world_size = self.trainer.world_size if self.trainer else 1
        self.cur_nimg += self.batch_size * world_size

        if self.ema_nimg != float('inf'):
            ema_rampup = self.ema_nimg / (self.batch_size * world_size)
            beta = 0.5 ** (1.0 / max(ema_rampup, 1e-8))
        else:
            beta = self.ema_beta
        
        for p_ema, p_main in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.copy_(p_main.detach().lerp(p_ema, beta))
        for b_ema, b_main in zip(self.G_ema.buffers(), self.G.buffers()):
            b_ema.copy_(b_main.detach())
        
        self.log('progress/kimg', self.cur_nimg / 1000.0, on_step=True, rank_zero_only=True)

    def on_train_batch_end(self, outputs, batch: any, batch_idx: int) -> None:
        # --- ADA Controller Logic ---
        if self.ada_enabled and self.cur_nimg >= self.last_ada_update_nimg + self.ada_interval:
            self.last_ada_update_nimg = self.cur_nimg
            
            adjustment = torch.sign(self.ada_stats - self.ada_target) * self.ada_adjust_speed
            self.ada_p.copy_((self.ada_p + adjustment).clamp(0.0, 1.0))
            
            self.log('ada/p', self.ada_p, on_step=True, rank_zero_only=True)
            self.log('ada/rt_stat', self.ada_stats, on_step=True, rank_zero_only=True)
            if self.trainer.is_global_zero:
                print(f"\nADA update at kimg {self.cur_nimg / 1000:.2f}: r_t={self.ada_stats.item():.4f}, p={self.ada_p.item():.4f}")
        
        # Snapshot generation logic
        if self.trainer.is_global_zero:
            cfg = self.hparams.training_cfg
            world_size = self.trainer.world_size if self.trainer else 1
            effective_batch_size = self.batch_size * world_size
            if effective_batch_size > 0:
                steps_per_kimg = 1000 / effective_batch_size
                snapshot_interval = int(steps_per_kimg * cfg.kimg_per_tick * cfg.snapshot_ticks)
                if snapshot_interval > 0 and (self.global_step + 1) % snapshot_interval == 0:
                    self.generate_and_log_samples()

    @torch.no_grad()
    def generate_and_log_samples(self):
        print(f"\nGenerating image snapshot at step {self.global_step} (kimg {self.cur_nimg // 1000})...")
        z = torch.randn(self.hparams.training_cfg.nimg_snapshot, self.G.z_dim, device=self.device)
        samples = self.G_ema(z).cpu()
        samples = (samples + 1) / 2.0 # Denormalize from [-1, 1] to [0, 1]
        grid = torchvision.utils.make_grid(samples, nrow=int(self.hparams.training_cfg.nimg_snapshot**0.5))
        
        if self.logger and self.logger.experiment:
             self.logger.experiment.add_image("Generated Samples", grid, self.global_step)
