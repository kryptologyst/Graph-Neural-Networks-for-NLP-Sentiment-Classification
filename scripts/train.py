"""Main training script for Graph Neural Networks for NLP."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils.config import Config, load_config
from src.utils.device import get_device, set_deterministic, print_device_info
from src.data.processing import DependencyGraphBuilder, SentimentDataset, create_synthetic_sentiment_dataset
from src.models.gnn_models import create_model, get_model_info
from src.train.trainer import GNNTrainer, evaluate_model


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log")
        ]
    )


def create_data_loaders(
    dataset: SentimentDataset,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> tuple:
    """
    Create train/validation/test data loaders.
    
    Args:
        dataset: SentimentDataset instance
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN for NLP sentiment classification")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "graphsage", "gat", "gin"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
        # Update config with command line arguments
        config.model.name = args.model
        config.model.hidden_dim = args.hidden_dim
        config.model.num_layers = args.num_layers
        config.training.learning_rate = args.learning_rate
        config.training.batch_size = args.batch_size
        config.training.num_epochs = args.num_epochs
        config.experiment.device = args.device
        config.data.random_seed = args.random_seed
    
    logger.info("Starting GNN for NLP training...")
    logger.info(f"Configuration: {config}")
    
    # Set deterministic behavior
    set_deterministic(config.data.random_seed)
    
    # Setup device
    device = get_device(config.experiment.device)
    print_device_info(device)
    
    # Create synthetic dataset
    logger.info(f"Creating synthetic dataset with {args.num_samples} samples...")
    sentences, labels = create_synthetic_sentiment_dataset(
        num_samples=args.num_samples,
        random_seed=config.data.random_seed
    )
    
    # Create dependency graph builder
    graph_builder = DependencyGraphBuilder(
        embedding_dim=config.data.word_embedding_dim,
        max_length=config.data.max_sentence_length
    )
    
    # Create dataset
    dataset = SentimentDataset(
        sentences=sentences,
        labels=labels,
        graph_builder=graph_builder,
        max_length=config.data.max_sentence_length
    )
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        random_seed=config.data.random_seed
    )
    
    logger.info(f"Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Create model
    model = create_model(
        model_name=config.model.name,
        input_dim=config.data.word_embedding_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=2,  # Binary classification
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        activation=config.model.activation,
        pooling=config.model.pooling,
        use_residual=config.model.use_residual,
        use_batch_norm=config.model.use_batch_norm
    )
    
    # Print model info
    model_info = get_model_info(model)
    logger.info(f"Model info: {model_info}")
    
    # Create trainer
    trainer_config = {
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "num_epochs": config.training.num_epochs,
        "early_stopping_patience": config.training.early_stopping_patience,
        "gradient_clip_norm": config.training.gradient_clip_norm,
        "scheduler": config.training.scheduler,
        "optimizer": "adam"
    }
    
    trainer = GNNTrainer(
        model=model,
        device=device,
        config=trainer_config,
        checkpoint_dir=config.experiment.checkpoint_dir,
        log_dir=config.experiment.log_dir
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        save_best=True
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final results
    results = {
        "config": config.__dict__,
        "model_info": model_info,
        "dataset_stats": stats,
        "test_metrics": test_metrics,
        "training_history": history
    }
    
    results_file = os.path.join(config.experiment.log_dir, "final_results.json")
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training completed! Results saved to {results_file}")
    logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Final test F1-score: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
