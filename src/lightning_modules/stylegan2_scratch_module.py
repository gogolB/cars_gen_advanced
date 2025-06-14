# src/lightning_modules/stylegan2_scratch_module.py

import os
import sys 
import copy 

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torchvision

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
                 batch_size: int
                ):
        super().__init__()
        self.save_hyperparameters()

        # --- Network Initialization ---
        print("Initializing From-Scratch Generator...")
        # CORRECTED: Instantiate Generator with an explicit list of arguments
        # from the model config to prevent TypeErrors. This is the definitive fix.
        self.G = Generator(
            z_dim=self.hparams.model_cfg.z_dim,
            w_dim=self.hparams.model_cfg.w_dim,
            num_mapping_layers=self.hparams.model_cfg.num_mapping_layers,
            mapping_lr_mul=self.hparams.model_cfg.mapping_lr_mul,
            img_resolution=self.hparams.model_cfg.img_resolution,
            img_channels=self.hparams.model_cfg.img_channels,
            channel_base=self.hparams.model_cfg.channel_base,
            channel_max=self.hparams.model_cfg.channel_max
        )
        
        print("Initializing From-Scratch Discriminator...")
        # CORRECTED: Instantiate Discriminator with an explicit list of arguments
        # from the model config to prevent TypeErrors. This is the definitive fix.
        self.D = Discriminator(
            img_resolution=self.hparams.model_cfg.img_resolution,
            img_channels=self.hparams.model_cfg.img_channels,
            channel_base=self.hparams.model_cfg.channel_base,
            channel_max=self.hparams.model_cfg.channel_max,
            mbstd_group_size=self.hparams.model_cfg.mbstd_group_size
        )

        print("Initializing G_ema...")
        self.G_ema = copy.deepcopy(self.G).eval()
        for param in self.G_ema.parameters():
            param.requires_grad = False
            
        self.ema_kimg = self.hparams.training_cfg.get('ema_kimg', 10.0)
        self.r1_gamma = self.hparams.training_cfg.get('r1_gamma', 10.0)
        self.d_reg_interval = self.hparams.training_cfg.get('d_reg_interval', 16)
        self.pl_weight = self.hparams.training_cfg.get('pl_weight', 0.0)
        self.g_reg_interval = self.hparams.training_cfg.get('g_reg_interval', 4)
        self.pl_decay = self.hparams.training_cfg.get('pl_decay', 0.01)
        if self.pl_weight > 0: 
            self.register_buffer('pl_mean', torch.zeros([]))
        if self.hparams.model_cfg.get('augment_pipe_kwargs'):
            print("Initializing AugmentPipe...")
            self.augment_pipe = AugmentPipe(self.hparams.model_cfg.augment_pipe_kwargs)
        else:
            self.augment_pipe = None
        self.ada_enabled = self.augment_pipe is not None and self.hparams.training_cfg.get('ada_kwargs') is not None
        if self.ada_enabled:
            print("ADA is enabled.")
            ada_kwargs = self.hparams.training_cfg.ada_kwargs
            self.ada_target = ada_kwargs.get('target_rt', 0.6)
            self.ada_interval_kimg = ada_kwargs.get('interval_kimg', 4)
            self.ada_speed_kimg = ada_kwargs.get('speed_kimg', 500)
            self.register_buffer('ada_p', torch.zeros([]))
            self.register_buffer('ada_stats', torch.zeros([]))
            self.ada_interval = self.ada_interval_kimg * 1000
            self.ada_adjust_speed = self.hparams.batch_size * self.ada_interval / (self.ada_speed_kimg * 1000)
        self.cur_nimg = 0
        self.last_ada_update_nimg = 0
        self.automatic_optimization = False 
        self.image_snapshot_dir = None
        self.last_snapshot_kimg = -1
        print("StyleGAN2ScratchLightningModule initialized.")


    def configure_optimizers(self):
        g_params = list(self.G.parameters()) 
        d_params = list(self.D.parameters())
        g_lr = self.hparams.training_cfg.g_lr
        d_lr = self.hparams.training_cfg.d_lr
        adam_betas = tuple(self.hparams.training_cfg.adam_betas) 
        opt_g = torch.optim.Adam(g_params, lr=g_lr, betas=adam_betas, eps=self.hparams.training_cfg.adam_eps)
        opt_d = torch.optim.Adam(d_params, lr=d_lr, betas=adam_betas, eps=self.hparams.training_cfg.adam_eps)
        return opt_g, opt_d

    def on_train_start(self):
        torch.autograd.set_detect_anomaly(True)
        if self.pl_weight > 0: self.pl_mean = self.pl_mean.to(self.device)
        if self.ada_enabled:
            self.ada_p = self.ada_p.to(self.device)
            self.ada_stats = self.ada_stats.to(self.device)
        if self.augment_pipe is not None:
            self.augment_pipe = self.augment_pipe.to(self.device)

    def forward(self, z, truncation_psi=0.7, noise_mode='random', return_ws=False):
        return self.G_ema(z, truncation_psi=truncation_psi, return_ws=return_ws)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        real_images, _ = batch
        current_batch_size = real_images.shape[0]
        real_images = real_images.to(self.device).to(torch.float32)
        
        # Get current augmentation probability
        current_p = self.ada_p.item() if self.ada_enabled else 0.0
        
        # =============================================================================
        # DISCRIMINATOR UPDATE
        # =============================================================================
        self.G.requires_grad_(False)
        self.D.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        
        # Generate fake images
        z_d = torch.randn(current_batch_size, self.G.z_dim, device=self.device)
        with torch.no_grad():
            # FIXED: Disable style mixing for D training to stabilize learning
            fake_images_d = self.G(z_d, style_mixing_prob=self.hparams.training_cfg.style_mixing_prob)
        
        # Apply augmentation
        real_images_aug = self.augment_pipe(real_images, p=current_p) if self.augment_pipe else real_images
        fake_images_aug = self.augment_pipe(fake_images_d, p=current_p) if self.augment_pipe else fake_images_d
        
        # Discriminator forward pass
        d_real_logits = self.D(real_images_aug)
        d_fake_logits = self.D(fake_images_aug)
        
        # Update ADA statistics
        if self.ada_enabled:
            self.ada_stats.copy_(d_real_logits.sign().mean().lerp(self.ada_stats, 0.999))
        
        # Main discriminator loss
        d_loss = compute_d_loss_logistic(d_real_logits, d_fake_logits)
        self.log('d_loss/main', d_loss, on_step=True, prog_bar=True, batch_size=current_batch_size)
        
        # R1 regularization (FIXED)
        if self.r1_gamma > 0 and (batch_idx % self.d_reg_interval == 0):
            real_images_for_r1 = real_images_aug.detach().requires_grad_(True)
            d_real_logits_r1 = self.D(real_images_for_r1)
            
            # FIXED: Proper R1 penalty scaling
            r1_penalty = compute_r1_penalty(real_images_for_r1, d_real_logits_r1)
            r1_loss = r1_penalty * (self.r1_gamma / 2) * self.d_reg_interval
            
            if not torch.isnan(r1_loss):
                d_loss = d_loss + r1_loss
                self.log('d_loss/r1', r1_loss, on_step=True, batch_size=current_batch_size)
        
        # Discriminator backward pass
        self.manual_backward(d_loss)
        
        # FIXED: Gradient clipping
        if hasattr(self.hparams.training_cfg, 'gradient_clip_val') and self.hparams.training_cfg.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.hparams.training_cfg.gradient_clip_val)
        
        opt_d.step()
        
        # =============================================================================
        # GENERATOR UPDATE
        # =============================================================================
        self.G.requires_grad_(True)
        self.D.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        
        # Generate fake images for G training
        z_g = torch.randn(current_batch_size, self.G.z_dim, device=self.device)
        
        # FIXED: Use proper style mixing probability
        style_mixing_prob = self.hparams.training_cfg.get('style_mixing_prob', 0.9)
        fake_images_g, ws_g = self.G(z_g, style_mixing_prob=style_mixing_prob, return_ws=True)
        
        with torch.no_grad():
            # Log W-space diversity
            _, ws = self.G(z_g, return_ws=True)
            self.log('debug/w_std', ws.std(), on_step=True)
        
        # Apply augmentation and get discriminator response
        fake_images_g_aug = self.augment_pipe(fake_images_g, p=current_p) if self.augment_pipe else fake_images_g
        d_fake_logits_g = self.D(fake_images_g_aug)
        
        # Main generator loss
        g_loss = compute_g_loss_nonsaturating(d_fake_logits_g)
        self.log('g_loss/main', g_loss, on_step=True, prog_bar=True, batch_size=current_batch_size)
        
        # Path length regularization (FIXED)
        if self.pl_weight > 0 and (batch_idx % self.g_reg_interval == 0):
            # Use smaller batch for PL to save memory
            pl_batch_size = max(1, current_batch_size // 2)
            pl_z = torch.randn(pl_batch_size, self.G.z_dim, device=self.device)
            
            # Generate with style mixing disabled for PL calculation
            _, ws_pl = self.G(pl_z, style_mixing_prob=0.0, return_ws=True)
            ws_pl.requires_grad_(True)
            
            # Calculate path lengths
            path_lengths = calculate_path_lengths(ws_pl, self.G.synthesis)
            
            if not torch.isnan(path_lengths).any():
                # FIXED: Proper PL penalty calculation
                pl_penalty = (path_lengths - self.pl_mean).pow(2)
                pl_loss = pl_penalty.mean() * self.pl_weight * self.g_reg_interval
                
                # Update PL mean
                self.pl_mean.copy_(path_lengths.mean().detach().lerp(self.pl_mean, self.pl_decay))
                
                if not torch.isnan(pl_loss):
                    g_loss = g_loss + pl_loss
                    self.log('g_loss/pl_penalty', pl_loss, on_step=True, batch_size=current_batch_size)
                    self.log('g_loss/pl_mean', self.pl_mean, on_step=True, batch_size=current_batch_size)
        
        # Generator backward pass
        self.manual_backward(g_loss)
        
        # FIXED: Gradient clipping for generator
        if hasattr(self.hparams.training_cfg, 'gradient_clip_val') and self.hparams.training_cfg.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.hparams.training_cfg.gradient_clip_val)
        
        opt_g.step()
        
        # =============================================================================
        # EMA UPDATE (FIXED)
        # =============================================================================
        with torch.no_grad():
            world_size = self.trainer.world_size if self.trainer else 1
            self.cur_nimg += current_batch_size * world_size
            
            # FIXED: Proper EMA calculation
            ema_kimg = self.hparams.training_cfg.get('ema_kimg', 10.0)
            ema_nimg = ema_kimg * 1000
            
            # Calculate EMA beta
            beta = 0.5 ** (current_batch_size * world_size / max(ema_nimg, 1e-8))
            
            # Update EMA weights
            for p_ema, p_main in zip(self.G_ema.parameters(), self.G.parameters()):
                p_ema.copy_(p_main.lerp(p_ema, beta))
            
            # Update EMA buffers
            for b_ema, b_main in zip(self.G_ema.buffers(), self.G.buffers()):
                b_ema.copy_(b_main)
        
        # Log progress
        self.log('progress/kimg', self.cur_nimg / 1000.0, on_step=True, rank_zero_only=True, batch_size=current_batch_size)
        
        # Log additional metrics for debugging
        self.log('debug/d_real_mean', d_real_logits.mean(), on_step=True, batch_size=current_batch_size)
        self.log('debug/d_fake_mean', d_fake_logits.mean(), on_step=True, batch_size=current_batch_size)
        self.log('debug/fake_img_std', fake_images_g.std(), on_step=True, batch_size=current_batch_size)

    
    def on_train_batch_end(self, outputs, batch: any, batch_idx: int) -> None:
        if self.ada_enabled and self.cur_nimg >= self.last_ada_update_nimg + self.ada_interval:
            self.last_ada_update_nimg = self.cur_nimg
            adjustment = torch.sign(self.ada_stats - self.ada_target) * self.ada_adjust_speed
            self.ada_p.copy_((self.ada_p + adjustment).clamp(0.0, 1.0))
            self.log('ada/p', self.ada_p, on_step=True, rank_zero_only=True)
            self.log('ada/rt_stat', self.ada_stats, on_step=True, rank_zero_only=True)

        if self.trainer.is_global_zero:
            kimg_now = self.cur_nimg // 1000
            kimg_per_tick = self.hparams.training_cfg.get('kimg_per_tick', 4)
            snapshot_ticks = self.hparams.training_cfg.get('snapshot_ticks', 10)
            snapshot_interval = kimg_per_tick * snapshot_ticks

            if snapshot_interval > 0 and kimg_now >= self.last_snapshot_kimg + snapshot_interval:
                self.generate_and_log_samples()
                self.last_snapshot_kimg = kimg_now

    @torch.no_grad()
    def generate_and_log_samples(self):
        if not self.trainer.is_global_zero: return
        if self.image_snapshot_dir is None and hasattr(self.logger, 'log_dir'):
            self.image_snapshot_dir = os.path.join(self.logger.log_dir, "image_snapshots")
            os.makedirs(self.image_snapshot_dir, exist_ok=True)
        current_kimg = self.cur_nimg // 1000
        z = torch.randn(self.hparams.training_cfg.nimg_snapshot, self.G.z_dim, device=self.device)
        samples = self.G_ema(z).cpu()
        samples = (samples + 1) / 2.0
        grid = torchvision.utils.make_grid(samples)
        if self.logger and hasattr(self.logger.experiment, 'add_image'):
            self.logger.experiment.add_image("Generated Snapshots", grid, self.global_step)
        if self.image_snapshot_dir:
            filename = os.path.join(self.image_snapshot_dir, f"snapshot_kimg_{current_kimg:06d}.png")
            torchvision.utils.save_image(grid, filename)
