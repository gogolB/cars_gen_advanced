# src/datamodules.py

import os
import pandas as pd
from typing import Optional, List, Dict, Any, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
import hydra # Import hydra

class CARSStyleGANDataset(Dataset):
    def __init__(self,
                 manifest_path: str, # Path to the cleaned CSV report, relative to project_root
                 project_root: str,  # Absolute path to the project root
                 image_size: Tuple[int, int] = (256, 256),
                 num_channels: int = 1, # CARS is grayscale
                 preprocessing_cfg: Optional[DictConfig] = None,
                 augmentations: Optional[A.Compose] = None, # This will be an instantiated A.Compose object
                 output_range: Tuple[float, float] = (-1.0, 1.0) # StyleGAN often expects [-1, 1]
                ):
        super().__init__()
        
        self.project_root = os.path.abspath(project_root) # Ensure project_root is absolute

        # manifest_path is expected to be relative to project_root
        resolved_manifest_path = os.path.join(self.project_root, manifest_path)
        
        try:
            self.manifest_df = pd.read_csv(resolved_manifest_path)
            self.actual_manifest_path_loaded = resolved_manifest_path # For logging
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Manifest file not found. Attempted path: '{resolved_manifest_path}'. "
                f"This path was constructed from project_root='{self.project_root}' and "
                f"manifest_path_param='{manifest_path}'. "
                "Ensure validate_dataset.py was run, the cleaned report exists, "
                "and these paths are correctly configured relative to your project's root directory."
            )

        self.image_paths = self.manifest_df['path'].tolist() # These paths are relative to project_root
        
        # --- Label Handling and Mapping ---
        if 'class_label' in self.manifest_df.columns:
            string_labels = self.manifest_df['class_label'].tolist()
            self.unique_labels = sorted(list(set(string_labels)))
            self.label_map = {label_str: i for i, label_str in enumerate(self.unique_labels)}
            self.numeric_labels = [self.label_map[lbl_str] for lbl_str in string_labels]
            print(f"Found class labels. Mapping: {self.label_map}")
        else:
            self.labels = [0] * len(self.image_paths) # Default label if not present
            self.numeric_labels = self.labels # Use default if no class_label column
            self.label_map = {0: 0} # Default map
            print("No 'class_label' column found in manifest. Using default label 0 for all images.")
        # --- End Label Handling ---

        self.image_size = image_size
        self.num_channels = num_channels
        self.preprocessing_cfg = preprocessing_cfg if preprocessing_cfg else DictConfig({})
        self.augmentations = augmentations # This is now expected to be an A.Compose object
        self.output_range = output_range

        print(f"CARSStyleGANDataset: Loaded {len(self.image_paths)} images from {self.actual_manifest_path_loaded}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        relative_image_path = self.image_paths[idx]
        # Construct absolute path using the stored absolute project_root
        image_abs_path = os.path.normpath(os.path.join(self.project_root, relative_image_path))
        
        numeric_label = self.numeric_labels[idx] # Use the mapped numeric label

        image_numpy_uint16 = cv2.imread(image_abs_path, cv2.IMREAD_UNCHANGED)

        if image_numpy_uint16 is None:
            raise ValueError(f"CARSStyleGANDataset Error: Image at {image_abs_path} (relative: {relative_image_path}) loaded as None.")

        current_image_float = image_numpy_uint16.astype(np.float32)

        if self.preprocessing_cfg.get('apply_per_image_percentile_norm', True):
            p_low_config = self.preprocessing_cfg.get('norm_perc_low', 1.0)
            p_high_config = self.preprocessing_cfg.get('norm_perc_high', 99.0)
            
            p_low = np.clip(p_low_config, 0.0, 100.0)
            p_high = np.clip(p_high_config, 0.0, 100.0)
            if p_low >= p_high:
                p_low = min(p_low, p_high - 1e-3)
                p_high = max(p_low + 1e-3, p_high)
                p_low = np.clip(p_low, 0.0, 99.999)
                p_high = np.clip(p_high, 0.001, 100.0)

            val_low, val_high = np.percentile(current_image_float, (p_low, p_high))
            denominator = val_high - val_low
            if denominator < 1e-8: denominator = 1e-8
            
            current_image_float = (current_image_float - val_low) / denominator
            current_image_float = np.clip(current_image_float, 0.0, 1.0)
        else:
            current_image_float = (current_image_float / 65535.0)
            current_image_float = np.clip(current_image_float, 0.0, 1.0)
        
        image_numpy_0_1_range = current_image_float.astype(np.float32)

        if image_numpy_0_1_range.ndim == 2 and self.num_channels == 1:
            image_numpy_0_1_range = np.expand_dims(image_numpy_0_1_range, axis=-1)
        elif image_numpy_0_1_range.ndim == 2 and self.num_channels > 1:
            image_numpy_0_1_range = np.stack([image_numpy_0_1_range]*self.num_channels, axis=-1)

        if self.augmentations: # self.augmentations is now an A.Compose object
            augmented = self.augmentations(image=image_numpy_0_1_range)
            image_tensor = augmented['image'] 
        else:
            img_resized = cv2.resize(image_numpy_0_1_range, self.image_size, interpolation=cv2.INTER_LINEAR)
            if img_resized.ndim == 2: img_resized = np.expand_dims(img_resized, axis=-1)
            image_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float()

        if self.output_range == (-1.0, 1.0):
            image_tensor = image_tensor * 2.0 - 1.0
        elif self.output_range == (0.0, 1.0):
            pass 
        else:
            old_min, old_max = 0.0, 1.0
            new_min, new_max = self.output_range
            image_tensor = (image_tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
            image_tensor = torch.clamp(image_tensor, new_min, new_max)
        
        return image_tensor, torch.tensor(numeric_label, dtype=torch.long) # Use numeric_label


class CARSStyleGANDataModule(pl.LightningDataModule):
    def __init__(self,
                 manifest_path: str = "data/data_validation_reports/dataset_validation_report_cleaned.csv",
                 project_root: str = ".", 
                 image_size: List[int] = [256, 256],
                 num_channels: int = 1,
                 preprocessing_cfg: Optional[DictConfig] = None,
                 train_aug_cfgs: Optional[List[Any]] = None, # Can be List of DictConfigs or already instantiated objects
                 val_aug_cfgs: Optional[List[Any]] = None,   
                 output_range: List[float] = [-1.0, 1.0],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 val_split_ratio: float = 0.0, 
                 seed: int = 42
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['train_aug_cfgs', 'val_aug_cfgs']) 

        self.train_dataset_instance: Optional[Dataset] = None 
        self.val_dataset_instance: Optional[Dataset] = None   
        
        self.image_size_tuple = tuple(self.hparams.image_size)
        self.output_range_tuple = tuple(self.hparams.output_range)

        self._raw_train_aug_cfgs = train_aug_cfgs
        self._raw_val_aug_cfgs = val_aug_cfgs
        
        self.train_augmentations_pipeline = self._create_augmentations(self._raw_train_aug_cfgs)
        self.val_augmentations_pipeline = self._create_augmentations(self._raw_val_aug_cfgs)


    def _create_augmentations(self, aug_configs_or_objects: Optional[List[Any]]) -> Optional[A.Compose]:
        if not aug_configs_or_objects:
            default_resize_cfg = {'_target_': 'albumentations.Resize', 'height': self.image_size_tuple[0], 'width': self.image_size_tuple[1], 'interpolation': cv2.INTER_LINEAR}
            default_totensor_cfg = {'_target_': 'albumentations.pytorch.ToTensorV2'}
            
            album_augs = [
                hydra.utils.instantiate(default_resize_cfg),
                hydra.utils.instantiate(default_totensor_cfg)
            ]
            return A.Compose(album_augs)

        album_augs = []
        for item_cfg_or_obj in aug_configs_or_objects:
            if isinstance(item_cfg_or_obj, (A.BasicTransform, A.Compose)):
                album_augs.append(item_cfg_or_obj)
            elif isinstance(item_cfg_or_obj, (DictConfig, dict)):
                try:
                    album_augs.append(hydra.utils.instantiate(item_cfg_or_obj))
                except Exception as e:
                    target_str = item_cfg_or_obj.get("_target_", "N/A") 
                    print(f"Warning: Could not instantiate augmentation with _target_ '{target_str}'. Config: {item_cfg_or_obj}. Error: {e}")
            else:
                print(f"Warning: Skipping unknown augmentation item type: {type(item_cfg_or_obj)} - {item_cfg_or_obj}")
        
        has_totensor = any(isinstance(aug, ToTensorV2) for aug in album_augs)
        if not has_totensor:
            is_totensor_in_configs = False
            if isinstance(aug_configs_or_objects, list):
                is_totensor_in_configs = any(
                    (isinstance(cfg, (dict, DictConfig)) and cfg.get("_target_") == "albumentations.pytorch.ToTensorV2")
                    for cfg in aug_configs_or_objects
                )
            if not is_totensor_in_configs: 
                 print("Warning: ToTensorV2 not found in augmentation pipeline or configs, adding it by default.")
                 album_augs.append(ToTensorV2())
            
        return A.Compose(album_augs)

    def setup(self, stage: Optional[str] = None):
        common_dataset_args = {
            "manifest_path": self.hparams.manifest_path, 
            "project_root": self.hparams.project_root,    
            "image_size": self.image_size_tuple,
            "num_channels": self.hparams.num_channels,
            "preprocessing_cfg": self.hparams.preprocessing_cfg,
            "output_range": self.output_range_tuple,
        }
        
        full_dataset = CARSStyleGANDataset(augmentations=self.train_augmentations_pipeline, **common_dataset_args)
        
        if self.hparams.val_split_ratio > 0 and len(full_dataset) > 1 : 
            total_len = len(full_dataset)
            val_len = int(self.hparams.val_split_ratio * total_len)
            if val_len == 0 and total_len > 1 : val_len = 1 
            train_len = total_len - val_len
            
            if train_len > 0 and val_len > 0:
                self.train_dataset_instance, self.val_dataset_instance = torch.utils.data.random_split(
                    full_dataset, 
                    [train_len, val_len],
                    generator=torch.Generator().manual_seed(self.hparams.seed)
                )
                print(f"Split dataset: Train {train_len}, Val {val_len}")
            else: 
                self.train_dataset_instance = full_dataset
                self.val_dataset_instance = None 
                print(f"Using full dataset for training: {len(full_dataset)}. Insufficient data for validation split or val_split_ratio is 0.")
        else:
            self.train_dataset_instance = full_dataset
            self.val_dataset_instance = None 
            print(f"Using full dataset for training: {len(self.train_dataset_instance)}. No validation split.")

    def train_dataloader(self):
        if not self.train_dataset_instance:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset_instance,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True 
        )

    def val_dataloader(self):
        if not self.val_dataset_instance:
            return None 
        return DataLoader(
            self.val_dataset_instance,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )

