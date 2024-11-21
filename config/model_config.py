# config.py
"""
配置文件，存储所有全局配置参数
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 256
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    augmentation_factor: float = 0.5
