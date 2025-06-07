# tests/test_lightning_module_scratch.py

import torch
import pytest
from omegaconf import OmegaConf, DictConfig
import numpy as np 

try:
    from src.lightning_modules.stylegan2_scratch_module import StyleGAN2ScratchLightningModule
    from src.models.stylegan2_networks_scratch import Generator, Discriminator 
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.lightning_modules.stylegan2_scratch_module import StyleGAN2ScratchLightningModule
    from src.models.stylegan2_networks_scratch import Generator, Discriminator


# --- Test Fixtures ---
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_model_cfg() -> DictConfig:
    return OmegaConf.create({
        "generator_kwargs": {
            "z_dim": 64, "w_dim": 64, "num_mapping_layers": 2,
            "img_resolution": 32, "img_channels": 1,
            "channel_base": 1024, "channel_max": 64 
        },
        "discriminator_kwargs": {
            "img_resolution": 32, "img_channels": 1,
            "channel_base": 1024, "channel_max": 64, 
            "mbstd_group_size": 2 
        }
    })

@pytest.fixture(params=[0.0, 2.0]) # Test with PLR disabled and enabled
def pl_weight_param(request):
    return request.param

@pytest.fixture
def dummy_training_cfg(pl_weight_param) -> DictConfig:
    return OmegaConf.create({
        "g_lr": 0.002, "d_lr": 0.002,
        "adam_betas": [0.0, 0.99], "adam_eps": 1e-8,
        "r1_gamma": 1.0, "d_reg_interval": 1, 
        "pl_weight": pl_weight_param, # Use the parametrized pl_weight
        "g_reg_interval": 1, "pl_decay": 0.01,
        "ema_kimg": 0.1, "ema_rampup_ratio": None, "ema_beta": 0.999,
        "ada_target": None, "ada_interval": 4, "ada_kimg":100,
        "nimg_snapshot": 4, "snapshot_ticks": 1, 
        "kimg_per_tick": 1, "total_kimg": 0.01 
    })

@pytest.fixture
def dummy_data_cfg() -> DictConfig:
    return OmegaConf.create({
        "batch_size": 2,
        "num_channels": 1, 
        "image_size": [32,32], 
        "output_range": [-1.0, 1.0] 
    })

@pytest.fixture
def lightning_module(dummy_model_cfg, dummy_training_cfg, dummy_data_cfg, device):
    module = StyleGAN2ScratchLightningModule(
        model_cfg=dummy_model_cfg,
        training_cfg=dummy_training_cfg,
        data_cfg=dummy_data_cfg
    )
    return module.to(device)

# --- Mocking Classes for Trainer ---
class MockStrategy:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        # Add the _lightning_optimizers attribute that the LightningModule.optimizers() property expects
        self._lightning_optimizers = optimizers

    def backward(self, tensor, model, *args, **kwargs):
        if tensor.requires_grad:
            tensor.backward(retain_graph=kwargs.get('retain_graph', False))

class MockTrainer:
    def __init__(self, configured_optimizers_tuple):
        self.global_step = 0
        self.world_size = 1
        self.accumulate_grad_batches = 1
        self.is_global_zero = True
        self.logger = None  
        self.strategy = MockStrategy(optimizers=list(configured_optimizers_tuple))


# --- Tests ---

def test_lightning_module_initialization(lightning_module, dummy_model_cfg, dummy_training_cfg):
    assert isinstance(lightning_module, StyleGAN2ScratchLightningModule)
    assert isinstance(lightning_module.G, Generator)
    assert isinstance(lightning_module.D, Discriminator)
    assert lightning_module.pl_weight == dummy_training_cfg.pl_weight
    # Check if pl_mean buffer is created only when pl_weight > 0
    if dummy_training_cfg.pl_weight > 0:
        assert 'pl_mean' in lightning_module._buffers
    else:
        assert 'pl_mean' not in lightning_module._buffers
    print("LightningModule initialization test passed.")

def test_lightning_module_configure_optimizers(lightning_module):
    optimizers = lightning_module.configure_optimizers()
    assert isinstance(optimizers, tuple) and len(optimizers) == 2
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(optimizers[1], torch.optim.Adam)
    print("LightningModule configure_optimizers test passed.")

def test_lightning_module_forward_pass(lightning_module, dummy_data_cfg, dummy_model_cfg, device):
    batch_size = dummy_data_cfg.batch_size
    z_dim = dummy_model_cfg.generator_kwargs.z_dim
    img_channels = dummy_model_cfg.generator_kwargs.img_channels
    img_resolution = dummy_model_cfg.generator_kwargs.img_resolution

    test_z = torch.randn(batch_size, z_dim, device=device)
    with torch.no_grad():
        output_img = lightning_module(test_z) 
    
    assert output_img.shape == (batch_size, img_channels, img_resolution, img_resolution)
    print("LightningModule forward pass test passed.")

def test_lightning_module_training_step(lightning_module, dummy_data_cfg, dummy_model_cfg, device):
    batch_size = dummy_data_cfg.batch_size
    img_channels = dummy_model_cfg.generator_kwargs.img_channels
    img_resolution = dummy_model_cfg.generator_kwargs.img_resolution

    # Setup mock trainer and assign to module
    opt_g, opt_d = lightning_module.configure_optimizers()
    mock_trainer_instance = MockTrainer(configured_optimizers_tuple=(opt_g, opt_d))
    lightning_module.trainer = mock_trainer_instance

    lightning_module.on_train_start()
    lightning_module = lightning_module.to(device)

    dummy_images = torch.randn(
        batch_size, img_channels, img_resolution, img_resolution, device=device
    )
    dummy_labels = torch.randint(0, 1, (batch_size,), device=device) 
    dummy_batch = (dummy_images, dummy_labels)

    try:
        lightning_module.training_step(dummy_batch, batch_idx=0)
        pl_status = "Enabled" if lightning_module.pl_weight > 0 else "Disabled"
        print(f"LightningModule training_step executed successfully with PLR {pl_status}.")
    except Exception as e:
        pytest.fail(f"training_step failed with error: {e}\nFull Traceback:\n{e.__traceback__}")
