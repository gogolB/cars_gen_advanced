# src/train.py

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from typing import List, Optional

def train(cfg: DictConfig) -> Optional[float]:
    """
    The main training pipeline.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: The metric score for hyperparameter optimization.
    """
    print("----------------------------------------------------")
    print("Starting training with resolved configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------------------------")

    # Set seed for reproducibility
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        print(f"Seed set to {cfg.seed}")

    # Instantiate DataModule
    print("Initializing DataModule...")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate Model
    print("Initializing Model...")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        training_cfg=cfg.training,
        data_cfg=cfg.data
    )

    # Instantiate Logger from the config
    # This now correctly respects the hydra.run.dir as the CWD
    print("Setting up Logger...")
    logger: pl.LightningLoggerBase = hydra.utils.instantiate(cfg.logger)

    # Instantiate Callbacks from the config (if any)
    print("Setting up Callbacks...")
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Instantiate Trainer
    print("Initializing Trainer...")

    trainer = pl.Trainer(
        **cfg.training, # Use all parameters from the training config section
        logger=logger,
        callbacks=callbacks
    )

    # Start Training
    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    print("Training finished.")

    # Return metric score for hyperparameter optimization
    return None

@hydra.main(version_base=None, config_path="../configs", config_name="main_config.yaml")
def main_hydra(cfg: DictConfig) -> Optional[float]:
    """
    Hydra entry point.
    
    This function is called by Hydra with the fully resolved configuration.
    """
    return train(cfg)

if __name__ == "__main__":
    main_hydra()

