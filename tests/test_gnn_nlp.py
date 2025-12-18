"""Unit tests for the GNN NLP project."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device import get_device, set_deterministic
from src.data.processing import DependencyGraphBuilder, SentimentDataset, create_synthetic_sentiment_dataset
from src.models.gnn_models import create_model, get_model_info, SentenceGCN, SentenceGAT
from src.utils.config import Config, get_default_config


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device_auto(self):
        """Test automatic device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_get_device_cpu(self):
        """Test CPU device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_deterministic(self):
        """Test deterministic behavior setup."""
        set_deterministic(42)
        # Test that random states are set
        assert torch.initial_seed() == 42


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        sentences, labels = create_synthetic_sentiment_dataset(num_samples=10, random_seed=42)
        
        assert len(sentences) == 10
        assert len(labels) == 10
        assert all(label in [0, 1] for label in labels)
        assert all(isinstance(sentence, str) for sentence in sentences)
    
    @patch('spacy.load')
    def test_dependency_graph_builder(self, mock_spacy_load):
        """Test dependency graph builder."""
        # Mock spaCy model
        mock_nlp = Mock()
        mock_token = Mock()
        mock_token.text = "test"
        mock_token.has_vector = True
        mock_token.vector = np.random.randn(300)
        mock_token.i = 0
        mock_token.head.i = 0
        mock_token.is_space = False
        mock_token.dep_ = "root"
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        builder = DependencyGraphBuilder()
        graph_data, tokens, dependencies = builder.sentence_to_graph("test")
        
        assert isinstance(graph_data, torch.Tensor) or hasattr(graph_data, 'x')
        assert len(tokens) >= 0
        assert isinstance(dependencies, list)


class TestModels:
    """Test GNN model implementations."""
    
    def test_create_model_gcn(self):
        """Test GCN model creation."""
        model = create_model(
            model_name="gcn",
            input_dim=300,
            hidden_dim=64,
            output_dim=2,
            num_layers=2
        )
        
        assert isinstance(model, SentenceGCN)
        assert model.input_dim == 300
        assert model.hidden_dim == 64
        assert model.output_dim == 2
    
    def test_create_model_gat(self):
        """Test GAT model creation."""
        model = create_model(
            model_name="gat",
            input_dim=300,
            hidden_dim=64,
            output_dim=2,
            num_layers=2
        )
        
        assert isinstance(model, SentenceGAT)
        assert model.input_dim == 300
        assert model.hidden_dim == 64
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = create_model(
            model_name="gcn",
            input_dim=300,
            hidden_dim=64,
            output_dim=2,
            num_layers=2
        )
        
        # Create dummy input
        x = torch.randn(5, 300)  # 5 nodes, 300 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        # Forward pass
        output = model(x, edge_index)
        
        assert output.shape == (2,)  # Binary classification
        assert torch.allclose(torch.sum(torch.exp(output)), torch.tensor(1.0), atol=1e-6)
    
    def test_get_model_info(self):
        """Test model information extraction."""
        model = create_model(
            model_name="gcn",
            input_dim=300,
            hidden_dim=64,
            output_dim=2,
            num_layers=2
        )
        
        info = get_model_info(model)
        
        assert "model_name" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["total_parameters"] > 0


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'training')
        assert hasattr(config, 'experiment')
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = Config()
        
        assert config.model.name == "gcn"
        assert config.model.hidden_dim == 64
        assert config.data.word_embedding_dim == 300
        assert config.training.learning_rate == 0.001


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training_setup(self):
        """Test complete training setup."""
        # Create synthetic data
        sentences, labels = create_synthetic_sentiment_dataset(num_samples=20, random_seed=42)
        
        # Create model
        model = create_model(
            model_name="gcn",
            input_dim=300,
            hidden_dim=32,
            output_dim=2,
            num_layers=1
        )
        
        # Test model info
        info = get_model_info(model)
        assert info["total_parameters"] > 0
        
        # Test device
        device = get_device("cpu")
        model = model.to(device)
        
        # Test forward pass with dummy data
        x = torch.randn(3, 300).to(device)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).to(device)
        
        output = model(x, edge_index)
        assert output.shape == (2,)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = get_default_config()
        
        # Test to_yaml (would need actual file writing)
        # This is a placeholder for actual file I/O testing
        assert hasattr(config, 'to_yaml')
        assert hasattr(config, 'from_yaml')


if __name__ == "__main__":
    pytest.main([__file__])
