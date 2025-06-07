# src/lightning_modules/stylegan2_ada_module.py

import os
import sys
import copy 
import inspect 
import contextlib 

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict 
import torchvision 

# --- Local Helper Functions ---
def local_requires_grad(model, flag=True):
    """Sets requires_grad attribute for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag

@contextlib.contextmanager
def local_ddp_sync(module, sync):
    """Context manager to enable/disable DDP synchronization."""
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        if hasattr(module, 'no_sync') and callable(module.no_sync):
             with module.no_sync():
                yield
        else:
            yield 
# --- End Local Helper Functions ---


# --- Helper to add StyleGAN2-ADA repo to path ---
def add_stylegan_path(stylegan_repo_path_from_config: str):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    path_if_absolute = stylegan_repo_path_from_config
    path_relative_to_project = os.path.join(project_root, stylegan_repo_path_from_config)
    actual_path_to_add = None
    
    if os.path.isabs(stylegan_repo_path_from_config) and os.path.isdir(stylegan_repo_path_from_config):
        actual_path_to_add = stylegan_repo_path_from_config
    elif os.path.isdir(path_relative_to_project):
        actual_path_to_add = os.path.abspath(path_relative_to_project) 
    else:
        raise FileNotFoundError(
            f"StyleGAN2-ADA PyTorch directory not found. Config path: '{stylegan_repo_path_from_config}'.\n"
            f"  - Checked as absolute: '{path_if_absolute}' (if it was an absolute path in config)\n"
            f"  - Checked as relative to project root ('{project_root}'): '{path_relative_to_project}'\n"
            "Please ensure 'stylegan2_ada_pytorch_path' in your model config is correct."
        )

    if actual_path_to_add not in sys.path:
        sys.path.insert(0, actual_path_to_add) 
        print(f"Attempted to add to sys.path: {actual_path_to_add}")


class StyleGAN2ADALightningModule(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig, data_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(logger=False) 

        add_stylegan_path(self.hparams.model_cfg.stylegan2_ada_pytorch_path)
        # print(f"Current sys.path after attempting to add StyleGAN2-ADA path: {sys.path[:5]}...") 

        try:
            from torch_utils import training_stats 
            from training.loss import StyleGAN2Loss 
            from training.networks import Generator, Discriminator 
            from dnnlib import EasyDict 
            try:
                from torch_utils.misc import ddp_sync as vendor_ddp_sync
                self.ddp_sync_context = vendor_ddp_sync
                print("Using ddp_sync from StyleGAN2-ADA vendor code.")
            except (ImportError, AttributeError) as e_ddp:
                print(f"Warning: Could not import ddp_sync from StyleGAN2-ADA vendor code (Error: {e_ddp}). Using local fallback.")
                self.ddp_sync_context = local_ddp_sync
            
            from torch_utils.ops import conv2d_gradfix 
            if self.hparams.training_cfg.accelerator == "cpu":
                if hasattr(conv2d_gradfix, 'enabled'): 
                    conv2d_gradfix.enabled = False
                    print("Disabled conv2d_gradfix for CPU execution.")
                else:
                    print("Warning: conv2d_gradfix module does not have 'enabled' attribute. Cannot disable for CPU.")
            else:
                if hasattr(conv2d_gradfix, 'enabled'):
                    conv2d_gradfix.enabled = True 
            from training.augment import AugmentPipe 
        except ImportError as e:
            print(f"Failed to import from StyleGAN2-ADA PyTorch codebase. Error: {e}")
            print(f"Ensure 'stylegan2_ada_pytorch_path' in model_cfg ('{self.hparams.model_cfg.stylegan2_ada_pytorch_path}') is correct and the repository is intact.")
            raise
        
        common_kwargs_dict = {
            "c_dim": 0, 
            "img_resolution": self.hparams.model_cfg.resolution,
            "img_channels": self.hparams.model_cfg.img_channels
        }
        
        g_mapping_kwargs = EasyDict(OmegaConf.to_container(self.hparams.model_cfg.mapping_kwargs, resolve=True))
        g_synthesis_kwargs = EasyDict(OmegaConf.to_container(self.hparams.model_cfg.synthesis_kwargs, resolve=True))
        
        self.G = Generator(
            z_dim=self.hparams.model_cfg.z_dim,
            w_dim=self.hparams.model_cfg.w_dim,
            mapping_kwargs=g_mapping_kwargs,
            synthesis_kwargs=g_synthesis_kwargs,
            **common_kwargs_dict 
        )

        d_kwargs_dict = OmegaConf.to_container(self.hparams.model_cfg.discriminator_kwargs, resolve=True)
        if d_kwargs_dict is None: d_kwargs_dict = {} 
        epilogue_kwargs_config = d_kwargs_dict.pop('epilogue_kwargs', {}) 
        if epilogue_kwargs_config is None: epilogue_kwargs_config = {}
        if 'mbstd_group_size' not in epilogue_kwargs_config:
            epilogue_kwargs_config['mbstd_group_size'] = 4 
        final_epilogue_kwargs = EasyDict(epilogue_kwargs_config)
        final_block_kwargs = EasyDict(d_kwargs_dict.pop('block_kwargs', {}))
        final_mapping_kwargs_d = EasyDict(d_kwargs_dict.pop('mapping_kwargs', {}))

        self.D = Discriminator(
            block_kwargs=final_block_kwargs,
            mapping_kwargs=final_mapping_kwargs_d,
            epilogue_kwargs=final_epilogue_kwargs, 
            **d_kwargs_dict, 
            **common_kwargs_dict 
        )

        self.G_ema = copy.deepcopy(self.G).eval()
        # Ensure G_ema parameters do not require gradients
        for param in self.G_ema.parameters():
            param.requires_grad = False
            
        self.ema_rampup = None 
        self.ema_kimg = self.hparams.training_cfg.get('ema_kimg', 10)
        
        augment_kwargs_dict = OmegaConf.to_container(self.hparams.model_cfg.augment_kwargs, resolve=True)
        self.augment_pipe = AugmentPipe(**augment_kwargs_dict).train().requires_grad_(False)
        
        self.ada_stats = None
        if self.hparams.training_cfg.get('ada_target', None) is not None:
            self.ada_stats = training_stats.Collector(regex='Loss/signs/real')
            print("Initialized ADA stats collector.")
        else:
            print("ADA target not set, ADA stats collector not initialized.")

        self.ada_interval = self.hparams.training_cfg.ada_interval 
        self.ada_kimg = self.hparams.training_cfg.ada_kimg 
        
        loss_init_args = {
            "device": self.device, 
            "G_mapping": self.G.mapping,
            "G_synthesis": self.G.synthesis,
            "D": self.D,
            "augment_pipe": self.augment_pipe,
            "r1_gamma": self.hparams.training_cfg.r1_gamma,
            "pl_weight": self.hparams.training_cfg.pl_weight,
            "pl_decay": self.hparams.training_cfg.pl_decay,
            "style_mixing_prob": self.hparams.model_cfg.loss_kwargs.get('style_mixing_prob', 0.9),
            "pl_batch_shrink": self.hparams.model_cfg.loss_kwargs.get('pl_batch_shrink', 2)
        }
        print("Instantiating StyleGAN2Loss...")
        self.loss_module = StyleGAN2Loss(**loss_init_args)


        self.cur_nimg = 0 
        self.batch_size = self.hparams.data_cfg.batch_size 
        self.automatic_optimization = False 

    def on_train_start(self):
        if self.augment_pipe is not None:
            self.augment_pipe.p.copy_(torch.as_tensor(0.0).to(self.device))
        if hasattr(self.loss_module, 'device'): 
            self.loss_module.device = self.device
        if hasattr(self.loss_module, 'pl_mean') and isinstance(self.loss_module.pl_mean, torch.Tensor):
             self.loss_module.pl_mean = self.loss_module.pl_mean.to(self.device)
        self.augment_pipe = self.augment_pipe.to(self.device)


    def forward(self, z): 
        return self.G_ema(z, None, truncation_psi=0.7, noise_mode='const') 

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        real_img, real_c = batch 
        real_img = real_img.to(torch.float32) 
        real_c = real_c.to(self.device) 

        if real_img.ndim == 3: 
            real_img = real_img.unsqueeze(1)
        elif real_img.ndim == 4 and real_img.shape[1] != self.hparams.model_cfg.img_channels:
             if real_img.shape[3] == self.hparams.model_cfg.img_channels: 
                real_img = real_img.permute(0, 3, 1, 2)
             else:
                raise ValueError(f"Unexpected image shape: {real_img.shape}")
        
        local_requires_grad(self.G, False) 
        local_requires_grad(self.D, True)  
        opt_d.zero_grad(set_to_none=True)
        
        z_d_phase = torch.randn([real_img.shape[0], self.G.z_dim], device=self.device)
        self.loss_module.accumulate_gradients(phase='Dmain', real_img=real_img, real_c=real_c, gen_z=z_d_phase, gen_c=None, sync=False, gain=1.0)
        
        if self.hparams.training_cfg.d_reg_interval > 0 and \
           (self.global_step % self.hparams.training_cfg.d_reg_interval == 0):
            self.loss_module.accumulate_gradients(phase='Dreg', real_img=real_img, real_c=real_c, gen_z=z_d_phase, gen_c=None, sync=False, gain=float(self.hparams.training_cfg.d_reg_interval))
        
        opt_d.step()

        if hasattr(self.loss_module, 'd_loss'):
            self.log('d_loss', self.loss_module.d_loss, prog_bar=True, logger=True, sync_dist=True)
        if hasattr(self.loss_module, 'r1_penalty') and self.loss_module.r1_penalty is not None:
             if self.hparams.training_cfg.d_reg_interval > 0 and \
                (self.global_step % self.hparams.training_cfg.d_reg_interval == 0):
                self.log('d_r1_penalty_attr', self.loss_module.r1_penalty, logger=True, sync_dist=True)

        if self.ada_stats is not None: # Check if ada_stats was initialized
            if hasattr(self.loss_module, 'd_real_logits') and self.loss_module.d_real_logits is not None:
                self.ada_stats.update_with_logits(self.loss_module.d_real_logits.detach(), real_img.shape[0])
            else:
                print("Warning: self.loss_module.d_real_logits not found for ADA update. ADA stats might be incorrect.")
        
        local_requires_grad(self.G, True)
        local_requires_grad(self.D, False)
        opt_g.zero_grad(set_to_none=True)

        z_g_phase = torch.randn([real_img.shape[0], self.G.z_dim], device=self.device)
        self.loss_module.accumulate_gradients(phase='Gmain', real_img=None, real_c=None, gen_z=z_g_phase, gen_c=None, sync=False, gain=1.0)
        
        if self.hparams.training_cfg.pl_weight > 0 and self.hparams.training_cfg.g_reg_interval > 0 and \
           (self.global_step % self.hparams.training_cfg.g_reg_interval == 0):
            self.loss_module.accumulate_gradients(phase='Greg', real_img=None, real_c=None, gen_z=z_g_phase, gen_c=None, sync=False, gain=float(self.hparams.training_cfg.g_reg_interval))

        opt_g.step()

        if hasattr(self.loss_module, 'g_loss'):
            self.log('g_loss', self.loss_module.g_loss, prog_bar=True, logger=True, sync_dist=True)
        if hasattr(self.loss_module, 'pl_penalty') and self.loss_module.pl_penalty is not None: 
            if self.hparams.training_cfg.pl_weight > 0 and self.hparams.training_cfg.g_reg_interval > 0 and \
               (self.global_step % self.hparams.training_cfg.g_reg_interval == 0):
                self.log('g_pl_penalty_attr', self.loss_module.pl_penalty, logger=True, sync_dist=True)
        if hasattr(self.loss_module, 'pl_mean') and isinstance(self.loss_module.pl_mean, torch.Tensor):
             self.log('pl_mean', self.loss_module.pl_mean, logger=True, sync_dist=True)

        world_size = self.trainer.world_size if self.trainer and hasattr(self.trainer, 'world_size') else 1
        if self.ema_rampup is None and self.hparams.training_cfg.total_kimg > 0:
            # Ensure ema_kimg is positive for rampup calculation
            current_ema_kimg = self.ema_kimg if self.ema_kimg > 0 else 10.0 # Use default if invalid
            self.ema_rampup = current_ema_kimg * 1000 / (self.batch_size * world_size) 

        if self.ema_rampup is not None and self.ema_rampup > 0: 
            ema_beta = 0.5 ** (self.batch_size * world_size / max(self.ema_rampup, 1e-8))
        else: 
            ema_beta = 0.999 
        
        for p_ema, p_main in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.copy_(p_main.detach().lerp(p_ema, ema_beta)) 
        for b_ema, b_main in zip(self.G_ema.buffers(), self.G.buffers()):
            b_ema.copy_(b_main.detach()) 
            
        if self.ada_stats is not None and (self.global_step + 1) % self.hparams.training_cfg.ada_interval == 0: 
            self.ada_stats.update() # Process collected stats from StyleGAN2Loss's internal training_stats.report
            # self.ada_stats.broadcast() # Collector does not have broadcast
            
            sign_estimate = self.ada_stats.get_value('Loss/signs/real') 
            if sign_estimate is None: 
                sign_estimate = self.hparams.training_cfg.ada_target 
                print("Warning: 'Loss/signs/real' not found in ada_stats. Using ada_target for sign_estimate.")

            if self.hparams.training_cfg.ada_kimg > 0: 
                if not isinstance(sign_estimate, torch.Tensor):
                    sign_estimate_tensor = torch.tensor(sign_estimate, device=self.device, dtype=torch.float32)
                else:
                    sign_estimate_tensor = sign_estimate.to(self.device)

                adjust = torch.sign(sign_estimate_tensor - self.hparams.training_cfg.ada_target) * \
                         (self.batch_size * self.hparams.training_cfg.ada_interval) / \
                         (self.hparams.training_cfg.ada_kimg * 1000)
                if hasattr(self.augment_pipe, 'p'): 
                    self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(torch.zeros_like(self.augment_pipe.p)))
                    self.log('ada_p', self.augment_pipe.p.item(), logger=True, sync_dist=True)
                else:
                    print("Warning: self.augment_pipe does not have attribute 'p'. Cannot update ADA p.")
            
            self.log('ada_sign_estimate', sign_estimate, logger=True, sync_dist=True)
            # No explicit clear needed for Collector with regex usually.
            # However, if it accumulates indefinitely, it might need clearing or windowing.
            # The official training_loop does not explicitly clear ada_stats (Collector) in the main loop.

        self.cur_nimg += self.batch_size * world_size 
        current_kimg = self.cur_nimg / 1000.0
        self.log('progress_kimg', current_kimg, logger=True, sync_dist=True, rank_zero_only=True)
        
    def configure_optimizers(self):
        g_params = self.G.parameters()
        d_params = self.D.parameters()

        opt_g = torch.optim.Adam(g_params, lr=self.hparams.training_cfg.g_lr, 
                                 betas=tuple(self.hparams.training_cfg.adam_betas), 
                                 eps=self.hparams.training_cfg.adam_eps)
        opt_d = torch.optim.Adam(d_params, lr=self.hparams.training_cfg.d_lr,
                                 betas=tuple(self.hparams.training_cfg.adam_betas), 
                                 eps=self.hparams.training_cfg.adam_eps)
        return opt_g, opt_d 

    @torch.no_grad()
    def generate_and_log_samples(self, tag_prefix="train"):
        if self.trainer.is_global_zero: 
            z_sample = torch.randn([self.hparams.training_cfg.nimg_snapshot, self.G.z_dim], device=self.device)
            fake_samples = self.G_ema(z_sample, c=None, truncation_psi=0.7, noise_mode='const').cpu()
            
            if self.hparams.model_cfg.img_channels == 1: 
                fake_samples = (fake_samples + 1) / 2 
            
            grid = torchvision.utils.make_grid(fake_samples, nrow=int(self.hparams.training_cfg.nimg_snapshot**0.5))
            
            if self.logger and hasattr(self.logger.experiment, 'add_image'):
                 self.logger.experiment.add_image(f"{tag_prefix}/generated_samples", grid, self.global_step)
            
            output_dir_base = "." 
            image_snapshot_dir = os.path.join(output_dir_base, "image_snapshots")
            os.makedirs(image_snapshot_dir, exist_ok=True)
            
            kimg_val = self.cur_nimg // 1000
            filename = os.path.join(image_snapshot_dir, f"{tag_prefix}_step_{self.global_step:07d}_kimg_{kimg_val:04d}.png")
            torchvision.utils.save_image(grid, filename)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.is_global_zero:
            images_per_kimg = 1000
            world_size = self.trainer.world_size if self.trainer and hasattr(self.trainer, 'world_size') else 1
            accumulate_grad_batches = self.trainer.accumulate_grad_batches if self.trainer and hasattr(self.trainer, 'accumulate_grad_batches') else 1

            effective_batch_size = self.batch_size * world_size * accumulate_grad_batches
            if effective_batch_size == 0: effective_batch_size = self.batch_size 

            steps_per_kimg_unit = images_per_kimg / effective_batch_size if effective_batch_size > 0 else float('inf')
            
            if steps_per_kimg_unit == float('inf'): return 

            steps_per_log_tick = int(steps_per_kimg_unit * self.hparams.training_cfg.kimg_per_tick)
            if steps_per_log_tick == 0: steps_per_log_tick = 1 

            steps_per_snapshot_tick = int(steps_per_log_tick * self.hparams.training_cfg.snapshot_ticks)
            if steps_per_snapshot_tick == 0: steps_per_snapshot_tick = steps_per_log_tick 

            if self.global_step > 0 and steps_per_snapshot_tick > 0 and \
               (self.global_step + 1) % steps_per_snapshot_tick == 0: 
                self.generate_and_log_samples(tag_prefix="train_snapshot")

