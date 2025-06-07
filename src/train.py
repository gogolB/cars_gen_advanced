# src/train.py

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from typing import List, Optional

def train(cfg: DictConfig) -> Optional[float]:
    """
    The main training pipeline.
    """
    print("----------------------------------------------------")
    print("Starting training with resolved configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------------------------")

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    print("Initializing DataModule...")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    print("Initializing Model...")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        training_cfg=cfg.training,
        batch_size=cfg.data.batch_size
    )

    print("Setting up Logger...")
    logger: pl.LightningLoggerBase = hydra.utils.instantiate(cfg.logger)

    print("Setting up Callbacks...")
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg and cfg.callbacks:
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # --- Trainer Initialization ---
    print("Initializing Trainer...")
    
    # Calculate max_steps from total_kimg.
    effective_batch_size = cfg.data.batch_size
    devices_cfg = cfg.training.trainer.get("devices", 1)
    if isinstance(devices_cfg, int):
        effective_batch_size *= devices_cfg
    elif isinstance(devices_cfg, list):
        effective_batch_size *= len(devices_cfg)
        
    max_steps = int((cfg.training.total_kimg * 1000) / effective_batch_size)
    print(f"Calculated max_steps: {max_steps} for total_kimg: {cfg.training.total_kimg}")

    # CORRECTED: Instantiate the Trainer using the nested 'trainer' block from the config
    # and override max_steps with our calculated value.
    trainer = hydra.utils.instantiate(
        cfg.training.trainer,
        max_steps=max_steps,
        logger=logger,
        callbacks=callbacks
    )

    print("Starting Training...")
    trainer.fit(model=model, datamodule=datamodule)
    print("Training finished.")

    return None

@hydra.main(version_base=None, config_path="../configs", config_name="main_config.yaml")
def main_hydra(cfg: DictConfig) -> Optional[float]:
    """Hydra entry point."""
    try:
        return train(cfg)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main_hydra()

