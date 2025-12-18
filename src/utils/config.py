"""Configuration management for Graph Neural Networks for NLP project."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
import os


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "gcn"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    activation: str = "relu"
    pooling: str = "mean"  # mean, max, attention
    use_residual: bool = True
    use_batch_norm: bool = True


@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_name: str = "synthetic_sentiment"
    max_sentence_length: int = 50
    min_sentence_length: int = 3
    word_embedding_dim: int = 300
    dependency_types: List[str] = field(default_factory=lambda: ["nsubj", "dobj", "amod", "advmod"])
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""
    project_name: str = "gnn-nlp-sentiment"
    experiment_name: str = "dependency-gcn"
    device: str = "auto"  # auto, cuda, mps, cpu
    deterministic: bool = True
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "gnn-nlp"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return OmegaConf.structured(cls(**config_dict))
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(self), f, default_flow_style=False)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        OmegaConf.set_struct(self, False)
        for key, value in updates.items():
            setattr(self, key, value)
        OmegaConf.set_struct(self, True)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return get_default_config()
