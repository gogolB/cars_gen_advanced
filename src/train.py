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

    # CORRECTED: This instantiation now works because the structure of
    # cfg.model (with its nested 'model_cfg' key) perfectly matches the
    # arguments of the StyleGAN2ScratchLightningModule's __init__ method.
    print("Initializing Model...")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        training_cfg=cfg.training,
        data_cfg=cfg.data
    )

    print("Setting up Logger...")
    logger: pl.LightningLoggerBase = hydra.utils.instantiate(cfg.logger)

    print("Setting up Callbacks...")
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg and cfg.callbacks:
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    print("Initializing Trainer...")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.training # Pass all training parameters from config
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
        # Optionally re-raise or handle as needed
        raise

if __name__ == "__main__":
    main_hydra()

