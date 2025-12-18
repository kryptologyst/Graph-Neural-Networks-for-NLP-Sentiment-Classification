"""Training utilities for GNN models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights for
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()
    
    def restore_checkpoint(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class MetricsTracker:
    """Track and compute various metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with new batch."""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.predictions or not self.targets:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Convert log probabilities to predictions
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
            pred_probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
        else:
            pred_labels = (predictions > 0.5).astype(int)
            pred_probs = predictions
        
        metrics = {
            "accuracy": accuracy_score(targets, pred_labels),
            "f1_macro": f1_score(targets, pred_labels, average="macro"),
            "f1_micro": f1_score(targets, pred_labels, average="micro"),
            "precision": precision_score(targets, pred_labels, average="macro", zero_division=0),
            "recall": recall_score(targets, pred_labels, average="macro", zero_division=0),
            "loss": np.mean(self.losses)
        }
        
        # Add AUC if binary classification
        if len(np.unique(targets)) == 2:
            try:
                if pred_probs.ndim > 1:
                    metrics["auc"] = roc_auc_score(targets, pred_probs[:, 1])
                else:
                    metrics["auc"] = roc_auc_score(targets, pred_probs)
            except ValueError:
                metrics["auc"] = 0.0
        
        return metrics


class GNNTrainer:
    """Trainer for GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GNN model to train
            device: Device to train on
            config: Training configuration
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = nn.NLLLoss()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 10),
            min_delta=config.get("early_stopping_min_delta", 0.0)
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_name = self.config.get("scheduler", "cosine").lower()
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get("num_epochs", 100)
            )
        elif scheduler_name == "step":
            step_size = self.config.get("step_size", 30)
            gamma = self.config.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get("gradient_clip_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["gradient_clip_norm"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            metrics_tracker.update(logits.detach(), batch.y.detach(), loss.item())
        
        return metrics_tracker.compute_metrics()
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Compute loss
                loss = self.criterion(logits, batch.y)
                
                # Update metrics
                metrics_tracker.update(logits.detach(), batch.y.detach(), loss.item())
        
        return metrics_tracker.compute_metrics()
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        num_epochs: int,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics["loss"], self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if save_best and epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics["loss"])
        
        # Save final model
        self.save_checkpoint(num_epochs, val_metrics["loss"], is_final=True)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "history": self.history
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        filename = "final_model.pt" if is_final else f"checkpoint_epoch_{epoch}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Save training history as JSON
        history_file = os.path.join(self.log_dir, "training_history.json")
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module = nn.NLLLoss()
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        criterion: Loss function
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            
            metrics_tracker.update(logits.detach(), batch.y.detach(), loss.item())
    
    return metrics_tracker.compute_metrics()
