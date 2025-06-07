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

from src.models.stylegan2_networks_scratch import Generator, Discriminator
from src.losses_scratch import (
    compute_d_loss_logistic,
    compute_g_loss_nonsaturating,
    compute_r1_penalty,
    calculate_path_lengths
)
# Import our from-scratch AugmentPipe
from src.augment_scratch import AugmentPipe


class StyleGAN2ScratchLightningModule(pl.LightningModule):
    def __init__(self,
                 model_cfg: DictConfig,    
                 training_cfg: DictConfig, 
                 data_cfg: DictConfig      
                ):
        super().__init__()
        self.save_hyperparameters(logger=True) 

        print("Initializing From-Scratch Generator...")
        self.G = Generator(**self.hparams.model_cfg.generator_kwargs)
        
        print("Initializing From-Scratch Discriminator...")
        self.D = Discriminator(**self.hparams.model_cfg.discriminator_kwargs)

        print("Initializing G_ema...")
        self.G_ema = copy.deepcopy(self.G).eval()
        for param in self.G_ema.parameters():
            param.requires_grad = False
        
        # EMA parameters
        self.ema_kimg = self.hparams.training_cfg.get('ema_kimg', 10.0) 
        self.ema_rampup_ratio = self.hparams.training_cfg.get('ema_rampup_ratio', 0.05) 
        self.ema_beta = self.hparams.training_cfg.get('ema_beta', 0.999) 
        self.ema_nimg = self.ema_kimg * 1000 
        self.ema_rampup = None 

        # Loss related parameters
        self.r1_gamma = self.hparams.training_cfg.get('r1_gamma', 10.0)
        self.d_reg_interval = self.hparams.training_cfg.get('d_reg_interval', 16)
        
        self.pl_weight = self.hparams.training_cfg.get('pl_weight', 0.0) 
        self.g_reg_interval = self.hparams.training_cfg.get('g_reg_interval', 4)
        self.pl_decay = self.hparams.training_cfg.get('pl_decay', 0.01)
        if self.pl_weight > 0: 
            self.register_buffer('pl_mean', torch.zeros([]))

        # --- AugmentPipe Initialization ---
        if self.hparams.model_cfg.get('augment_pipe_kwargs') and \
           self.hparams.model_cfg.augment_pipe_kwargs.get('enabled', False):
            print("Initializing AugmentPipe from scratch...")
            self.augment_pipe = AugmentPipe(self.hparams.model_cfg.augment_pipe_kwargs)
        else:
            self.augment_pipe = None
            print("AugmentPipe not configured or not enabled, augmentations will be skipped.")
        # --- End AugmentPipe Initialization ---

        # Tracking
        self.cur_nimg = 0
        self.batch_size = self.hparams.data_cfg.batch_size 
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
        if self.ema_rampup_ratio is not None and self.ema_rampup_ratio > 0 and \
           hasattr(self.hparams.training_cfg, 'total_kimg') and self.hparams.training_cfg.total_kimg > 0:
            total_nimg = self.hparams.training_cfg.total_kimg * 1000
            self.ema_nimg = total_nimg * self.ema_rampup_ratio
            print(f"EMA ramp-up nimg set to {self.ema_nimg} ({self.ema_rampup_ratio*100}% of total_kimg)")
        elif self.ema_kimg > 0 :
            self.ema_nimg = self.ema_kimg * 1000
            print(f"EMA nimg set to {self.ema_nimg} (from ema_kimg)")
        else: 
            self.ema_nimg = float('inf') 
            print(f"EMA using fixed beta: {self.ema_beta} (ema_kimg or rampup_ratio not configured for rampup)")

        if self.pl_weight > 0 and hasattr(self, 'pl_mean'):
             self.pl_mean = self.pl_mean.to(self.device)
        
        # Move augment_pipe to the correct device if it exists
        if self.augment_pipe is not None:
            self.augment_pipe = self.augment_pipe.to(self.device)

    def forward(self, z, c=None, truncation_psi=0.7, noise_mode='random', return_ws=False):
        if self.G_ema:
            return self.G_ema.forward(z, c=c, truncation_psi=truncation_psi, noise_mode=noise_mode, return_ws=return_ws)
        else: 
            return self.G.forward(z, c=c, truncation_psi=truncation_psi, noise_mode=noise_mode, return_ws=return_ws)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers() 
        
        real_images, real_labels = batch 
        real_images = real_images.to(self.device).to(torch.float32)

        # --- Train Discriminator ---
        for p in self.G.parameters(): p.requires_grad = False
        for p in self.D.parameters(): p.requires_grad = True
        opt_d.zero_grad()

        z_for_d = torch.randn(real_images.shape[0], self.G.z_dim, device=self.device)
        with torch.no_grad(): 
            fake_images_d = self.G(z_for_d) 
        
        # Apply augmentations to both real and fake images before D
        real_images_for_d = self.augment_pipe(real_images) if self.augment_pipe else real_images
        fake_images_for_d_detached = fake_images_d.detach() 
        fake_images_for_d = self.augment_pipe(fake_images_for_d_detached) if self.augment_pipe else fake_images_for_d_detached
        
        d_real_logits = self.D(real_images_for_d) 
        d_fake_logits = self.D(fake_images_for_d) 

        d_loss = compute_d_loss_logistic(d_real_logits, d_fake_logits)
        
        if self.trainer and self.trainer.logger:
            self.log('d_loss/main', d_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.r1_gamma > 0 and self.d_reg_interval > 0 and (batch_idx % self.d_reg_interval == 0):
            # For R1, operate on the same augmented real images D saw for its main loss
            real_images_for_r1_input = real_images_for_d.detach().requires_grad_(True) 
            d_real_logits_for_r1 = self.D(real_images_for_r1_input)
            
            r1_penalty_raw = compute_r1_penalty(real_images_for_r1_input, d_real_logits_for_r1, device=self.device)
            r1_penalty = r1_penalty_raw * (self.r1_gamma / 2.0) * self.d_reg_interval
            
            if not torch.isnan(r1_penalty).any() and not torch.isinf(r1_penalty).any():
                d_loss = d_loss + r1_penalty
                if self.trainer and self.trainer.logger:
                    self.log('d_loss/r1_penalty', r1_penalty, on_step=True, on_epoch=False, logger=True)
            else:
                current_global_step = self.global_step if hasattr(self, 'global_step') else batch_idx
                print(f"Warning: R1 penalty is NaN/Inf at step {current_global_step}. Skipping R1 for this D step.")
        
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Train Generator ---
        for p in self.G.parameters(): p.requires_grad = True
        for p in self.D.parameters(): p.requires_grad = False
        opt_g.zero_grad()

        z_for_g = torch.randn(real_images.shape[0], self.G.z_dim, device=self.device)
        fake_images_g, ws_for_g_loss = self.G(z_for_g, return_ws=True) 
        
        # Apply augmentations to fake images before G loss calculation
        fake_images_g_for_d = self.augment_pipe(fake_images_g) if self.augment_pipe else fake_images_g
        
        d_fake_logits_g = self.D(fake_images_g_for_d) 
        g_loss = compute_g_loss_nonsaturating(d_fake_logits_g)
        
        if self.trainer and self.trainer.logger:
            self.log('g_loss/main', g_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # --- Path Length Regularization (PLR) Integration ---
        if self.pl_weight > 0 and self.g_reg_interval > 0 and (batch_idx % self.g_reg_interval == 0):
            pl_batch_size = max(1, real_images.shape[0] // 2)
            pl_z = torch.randn(pl_batch_size, self.G.z_dim, device=self.device)
            
            # Get the single mapped ws, which is what we need to calculate grads w.r.t.
            ws_single_for_plr = self.G.mapping(pl_z)
            # Tile it to the shape expected by the synthesis network
            ws_for_plr = ws_single_for_plr.unsqueeze(1).repeat(1, self.G.num_ws, 1)
            ws_for_plr.requires_grad_(True)

            path_lengths = calculate_path_lengths(ws_for_plr, self.G.synthesis)
            
            pl_penalty = (path_lengths - self.pl_mean).square().mean()
            
            self.pl_mean.copy_(path_lengths.mean().detach().lerp(self.pl_mean, self.pl_decay))
            
            pl_loss = pl_penalty * self.pl_weight * self.g_reg_interval
            
            if not torch.isnan(pl_loss).any() and not torch.isinf(pl_loss).any():
                g_loss = g_loss + pl_loss
                if self.trainer and self.trainer.logger:
                    self.log('g_loss/pl_penalty', pl_loss, on_step=True, on_epoch=False, logger=True)
                    self.log('g_loss/pl_mean', self.pl_mean, on_step=True, on_epoch=False, logger=True)
            else:
                current_global_step = self.global_step if hasattr(self, 'global_step') else batch_idx
                print(f"Warning: PL penalty is NaN/Inf at step {current_global_step}. Skipping PL for this G step.")
        # --- End PLR Integration ---

        self.manual_backward(g_loss)
        opt_g.step()

        # --- EMA Update for G ---
        world_size = self.trainer.world_size if self.trainer and hasattr(self.trainer, 'world_size') else 1
        effective_batch_size = self.batch_size * world_size 

        if self.ema_nimg != float('inf'): 
            if self.ema_rampup is None: 
                 self.ema_rampup = self.ema_nimg / effective_batch_size if effective_batch_size > 0 else float('inf')

            if self.ema_rampup > 0: 
                beta = 0.5 ** (effective_batch_size / max(self.ema_rampup, 1e-8))
            else: 
                beta = self.ema_beta
        else: 
            beta = self.ema_beta
        
        for p_ema, p_main in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.copy_(p_main.detach().lerp(p_ema, beta))
        for b_ema, b_main in zip(self.G_ema.buffers(), self.G.buffers()):
            b_ema.copy_(b_main.detach())
            
        self.cur_nimg += effective_batch_size
        current_kimg = self.cur_nimg / 1000.0
        if self.trainer and self.trainer.logger:
            self.log('progress/kimg', current_kimg, on_step=True, on_epoch=False, logger=True, rank_zero_only=True)

    def on_train_batch_end(self, outputs, batch: any, batch_idx: int) -> None:
        if hasattr(self, 'trainer') and self.trainer.is_global_zero: 
            cfg_training = self.hparams.training_cfg
            
            images_per_kimg = 1000
            world_size = self.trainer.world_size if hasattr(self.trainer, 'world_size') else 1
            accumulate_grad_batches = self.trainer.accumulate_grad_batches if hasattr(self.trainer, 'accumulate_grad_batches') else 1
            
            effective_batch_size = self.batch_size * world_size * accumulate_grad_batches
            if effective_batch_size == 0: effective_batch_size = self.batch_size 

            steps_per_kimg_unit = images_per_kimg / effective_batch_size if effective_batch_size > 0 else float('inf')
            
            if steps_per_kimg_unit == float('inf'): return 

            steps_per_log_tick = int(steps_per_kimg_unit * cfg_training.kimg_per_tick)
            if steps_per_log_tick == 0: steps_per_log_tick = 1 

            steps_per_snapshot_tick = int(steps_per_log_tick * cfg_training.snapshot_ticks)
            if steps_per_snapshot_tick == 0: steps_per_snapshot_tick = steps_per_log_tick 

            current_global_step = self.global_step if hasattr(self, 'global_step') else 0
            if current_global_step > 0 and \
               steps_per_snapshot_tick > 0 and \
               (current_global_step + 1) % steps_per_snapshot_tick == 0: 
                self.generate_and_log_samples(tag_prefix="train_snapshot")


    @torch.no_grad()
    def generate_and_log_samples(self, tag_prefix="train_snapshot"):
        if not hasattr(self, 'trainer') or not self.trainer.is_global_zero:
            return

        current_global_step = self.global_step if hasattr(self, 'global_step') else 0

        print(f"\nGenerating image snapshot at step {current_global_step} (kimg {self.cur_nimg // 1000})...")
        z_sample = torch.randn(
            self.hparams.training_cfg.nimg_snapshot, 
            self.G.z_dim, 
            device=self.device
        )
        fake_samples = self.G_ema(z_sample) 
        fake_samples = fake_samples.cpu() 

        if self.hparams.data_cfg.output_range == [-1.0, 1.0]: 
            fake_samples = (fake_samples + 1) / 2.0
        fake_samples = torch.clamp(fake_samples, 0.0, 1.0)
            
        grid = torchvision.utils.make_grid(
            fake_samples, 
            nrow=int(self.hparams.training_cfg.nimg_snapshot**0.5),
            normalize=False 
        )
            
        if hasattr(self, 'logger') and self.logger and hasattr(self.logger, 'experiment') and \
           hasattr(self.logger.experiment, 'add_image'):
             self.logger.experiment.add_image(f"{tag_prefix}/generated_samples", grid, current_global_step)
        
        output_dir_base = "." 
        image_snapshot_dir = os.path.join(output_dir_base, "image_snapshots")
        os.makedirs(image_snapshot_dir, exist_ok=True)
            
        kimg_val = self.cur_nimg // 1000
        filename = os.path.join(image_snapshot_dir, f"{tag_prefix}_step_{current_global_step:07d}_kimg_{kimg_val:04d}.png")
        torchvision.utils.save_image(grid, filename)
        print(f"Saved snapshot to {filename}")


if __name__ == '__main__':
    # This block can be used for direct script testing.
    # It will need to be updated to test the AugmentPipe integration as well.
    pass
