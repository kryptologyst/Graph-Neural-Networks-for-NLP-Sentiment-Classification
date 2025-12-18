#!/usr/bin/env python3
"""Quick demo script to showcase the modernized GNN NLP project."""

import sys
from pathlib import Path
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.device import get_device, set_deterministic, print_device_info
from src.data.processing import DependencyGraphBuilder, create_synthetic_sentiment_dataset
from src.models.gnn_models import create_model, get_model_info
from src.utils.config import get_default_config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run a quick demonstration of the modernized GNN NLP project."""
    print("=" * 60)
    print("Graph Neural Networks for NLP - Modernized Project Demo")
    print("=" * 60)
    
    # 1. Device setup
    print("\n1. Device Configuration")
    print("-" * 30)
    device = get_device("auto")
    print_device_info(device)
    
    # 2. Deterministic setup
    print("\n2. Setting up deterministic behavior...")
    set_deterministic(42)
    print("✓ Deterministic seeding configured")
    
    # 3. Configuration
    print("\n3. Configuration Management")
    print("-" * 30)
    config = get_default_config()
    print(f"✓ Model: {config.model.name}")
    print(f"✓ Hidden dim: {config.model.hidden_dim}")
    print(f"✓ Learning rate: {config.training.learning_rate}")
    print(f"✓ Device: {config.experiment.device}")
    
    # 4. Data pipeline
    print("\n4. Data Pipeline")
    print("-" * 30)
    print("Creating synthetic sentiment dataset...")
    sentences, labels = create_synthetic_sentiment_dataset(num_samples=50, random_seed=42)
    print(f"✓ Created {len(sentences)} sentences")
    print(f"✓ Class distribution: {labels.count(0)} negative, {labels.count(1)} positive")
    
    # 5. Dependency parsing
    print("\n5. Dependency Graph Construction")
    print("-" * 30)
    try:
        graph_builder = DependencyGraphBuilder()
        sample_sentence = sentences[0]
        print(f"Sample sentence: '{sample_sentence}'")
        
        graph_data, tokens, dependencies = graph_builder.sentence_to_graph(sample_sentence)
        print(f"✓ Parsed into graph with {graph_data.num_nodes} nodes and {graph_data.edge_index.size(1)} edges")
        print(f"✓ Tokens: {tokens}")
        print(f"✓ Dependencies: {len(dependencies)}")
        
    except Exception as e:
        print(f"⚠ Dependency parsing requires spaCy model: {e}")
        print("  Install with: python -m spacy download en_core_web_sm")
    
    # 6. Model creation
    print("\n6. Model Architecture")
    print("-" * 30)
    try:
        model = create_model(
            model_name="gcn",
            input_dim=300,
            hidden_dim=64,
            output_dim=2,
            num_layers=2
        )
        
        model_info = get_model_info(model)
        print(f"✓ Model: {model_info['model_name']}")
        print(f"✓ Parameters: {model_info['total_parameters']:,}")
        print(f"✓ Architecture: {model_info['num_layers']} layers, {model_info['hidden_dim']} hidden dim")
        
        # Test forward pass
        if 'graph_data' in locals():
            model.eval()
            with torch.no_grad():
                output = model(graph_data.x, graph_data.edge_index)
                print(f"✓ Forward pass successful: output shape {output.shape}")
        
    except Exception as e:
        print(f"⚠ Model creation failed: {e}")
        print("  This requires PyTorch Geometric installation")
    
    # 7. Project structure
    print("\n7. Project Structure")
    print("-" * 30)
    project_root = Path(__file__).parent
    print(f"✓ Project root: {project_root}")
    
    # Check key directories
    key_dirs = ["src", "configs", "scripts", "demo", "tests"]
    for dir_name in key_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"⚠ {dir_name}/ directory missing")
    
    # 8. Next steps
    print("\n8. Next Steps")
    print("-" * 30)
    print("To get started with the full project:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download spaCy model: python -m spacy download en_core_web_sm")
    print("3. Train a model: python scripts/train.py")
    print("4. Run the demo: streamlit run demo/app.py")
    print("5. Evaluate models: python scripts/evaluate.py --checkpoint checkpoints/final_model.pt")
    
    print("\n" + "=" * 60)
    print("Demo completed! The project is ready for development.")
    print("=" * 60)


if __name__ == "__main__":
    main()
