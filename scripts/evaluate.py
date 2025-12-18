"""Evaluation script for trained GNN models."""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils.device import get_device
from src.data.processing import DependencyGraphBuilder, SentimentDataset, create_synthetic_sentiment_dataset
from src.models.gnn_models import create_model
from src.train.trainer import evaluate_model


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    config = checkpoint.get("config", {})
    
    # Create model
    model = create_model(
        model_name=config.get("model_name", "gcn"),
        input_dim=config.get("input_dim", 300),
        hidden_dim=config.get("hidden_dim", 64),
        output_dim=config.get("output_dim", 2),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.5),
        activation=config.get("activation", "relu"),
        pooling=config.get("pooling", "mean"),
        use_residual=config.get("use_residual", True),
        use_batch_norm=config.get("use_batch_norm", True)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model_comprehensive(
    model,
    test_loader,
    device: torch.device,
    model_name: str = "Unknown"
):
    """Comprehensive model evaluation."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating {model_name} model...")
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (Micro): {metrics['f1_micro']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    
    if "auc" in metrics:
        logger.info(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained GNN models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create test dataset
    logger.info(f"Creating test dataset with {args.test_samples} samples...")
    sentences, labels = create_synthetic_sentiment_dataset(
        num_samples=args.test_samples,
        random_seed=42
    )
    
    # Create dependency graph builder
    graph_builder = DependencyGraphBuilder()
    
    # Create test dataset
    test_dataset = SentimentDataset(
        sentences=sentences,
        labels=labels,
        graph_builder=graph_builder
    )
    
    # Create test loader
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    metrics = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name=checkpoint.get("config", {}).get("model_name", "Unknown")
    )
    
    # Save results
    if args.output_file:
        results = {
            "checkpoint": args.checkpoint,
            "model_config": checkpoint.get("config", {}),
            "test_samples": args.test_samples,
            "metrics": metrics,
            "device": str(device)
        }
        
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {args.output_file}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
