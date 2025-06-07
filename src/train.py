# src/train.py

import os
import sys 
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict # Import open_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)
    print(f"Added project root to sys.path: {PROJECT_ROOT_DIR}")

def train(cfg: DictConfig) -> None:
    print("----------------------------------------------------")
    print("Starting training with configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------------------------")

    pl.seed_everything(cfg.seed, workers=True)

    print("Initializing DataModule...")
    # Get original CWD to resolve paths correctly, as Hydra changes CWD
    original_cwd = hydra.utils.get_original_cwd()
    with open_dict(cfg.data): # Allow modification of the config
        cfg.data.project_root = original_cwd # Set project_root to the absolute original path
    
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    print("Initializing Model (StyleGAN2ADALightningModule)...")
    module_class = hydra.utils.get_class(cfg.model._target_)
    model: pl.LightningModule = module_class(
        model_cfg=cfg.model, 
        training_cfg=cfg.training,
        data_cfg=cfg.data
    )

    print("Setting up Callbacks...")
    callbacks = []
    
    num_gpus_from_cfg = cfg.training.get('devices', 1)
    if isinstance(num_gpus_from_cfg, (list, tuple)):
        num_gpus = len(num_gpus_from_cfg)
    elif isinstance(num_gpus_from_cfg, int):
        num_gpus = num_gpus_from_cfg
    elif isinstance(num_gpus_from_cfg, str) and num_gpus_from_cfg == "auto":
        num_gpus = torch.cuda.device_count() if cfg.training.accelerator == "gpu" else 1
    else: 
        num_gpus = 1 
    if num_gpus == 0: num_gpus = 1

    effective_batch_size = cfg.data.batch_size * num_gpus 
    if effective_batch_size == 0: effective_batch_size = cfg.data.batch_size

    steps_per_kimg_unit = 1000 / effective_batch_size if effective_batch_size > 0 else float('inf')
    
    checkpoint_freq_steps = cfg.training.get("checkpoint_every_n_train_steps", None)
    if checkpoint_freq_steps is None and steps_per_kimg_unit != float('inf') and cfg.training.kimg_per_tick > 0:
        checkpoint_freq_steps = int(cfg.training.kimg_per_tick * steps_per_kimg_unit)
    if not checkpoint_freq_steps or checkpoint_freq_steps <= 0 : checkpoint_freq_steps = 5000 

    checkpoint_callback = ModelCheckpoint(
        dirpath=None, 
        filename='cars-stylegan2ada-{step:07d}-{progress_kimg:.2f}kimg',
        every_n_train_steps=checkpoint_freq_steps,
        save_top_k=-1, 
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    print("Setting up Logger...")
    logger = TensorBoardLogger(save_dir=".", name=cfg.project_name, version="") 
                                                                               
    print("Initializing Trainer...")
    
    max_steps = -1 
    if cfg.training.total_kimg > 0 and effective_batch_size > 0:
        total_images_to_see = cfg.training.total_kimg * 1000
        max_steps = int(total_images_to_see / effective_batch_size)
    
    if max_steps <=0 : 
        print(f"Warning: max_steps calculated as {max_steps}. Defaulting to training for max_epochs from config if set, or 1 epoch.")
        max_steps = None 
        if cfg.training.max_epochs <= 0: 
             cfg.training.max_epochs = 1

    print(f"Effective batch size: {effective_batch_size}")
    if max_steps:
        print(f"Calculated max_steps: {max_steps} (for total_kimg: {cfg.training.total_kimg})")
    else:
        print(f"Training for max_epochs: {cfg.training.max_epochs} (max_steps not set).")

    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        max_epochs=cfg.training.max_epochs if max_steps is None else -1, 
        max_steps=max_steps if max_steps is not None else -1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        precision=cfg.training.get("precision", "32-true"), 
    )

    print(f"Starting training...")
    trainer.fit(model, datamodule=datamodule)

    print("----------------------------------------------------")
    print("Training finished.")
    print("----------------------------------------------------")

@hydra.main(config_path="../configs", config_name="main_config.yaml", version_base="1.3")
def main_hydra(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main_hydra()
