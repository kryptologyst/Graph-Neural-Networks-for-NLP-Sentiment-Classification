"""Data processing utilities for NLP graph construction."""

import torch
import spacy
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from torch_geometric.data import Data, Batch
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentenceData:
    """Container for sentence and its metadata."""
    text: str
    label: int
    graph: Optional[Data] = None
    tokens: Optional[List[str]] = None
    dependencies: Optional[List[Tuple[int, int, str]]] = None


class DependencyGraphBuilder:
    """Builds dependency graphs from sentences using spaCy."""
    
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        embedding_dim: int = 300,
        include_edge_types: bool = True,
        bidirectional: bool = True,
        max_length: int = 50
    ):
        """
        Initialize the dependency graph builder.
        
        Args:
            model_name: spaCy model name
            embedding_dim: Dimension of word embeddings
            include_edge_types: Whether to include dependency type as edge features
            bidirectional: Whether to make edges bidirectional
            max_length: Maximum sentence length
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.include_edge_types = include_edge_types
        self.bidirectional = bidirectional
        self.max_length = max_length
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"spaCy model {model_name} not found. Please install it with: python -m spacy download {model_name}")
            raise
    
    def sentence_to_graph(self, sentence: str) -> Tuple[Data, List[str], List[Tuple[int, int, str]]]:
        """
        Convert a sentence to a dependency graph.
        
        Args:
            sentence: Input sentence text
            
        Returns:
            Tuple of (graph_data, tokens, dependencies)
        """
        # Process sentence with spaCy
        doc = self.nlp(sentence)
        
        # Extract tokens and limit length
        tokens = [token.text for token in doc if not token.is_space]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            logger.warning(f"Sentence truncated to {self.max_length} tokens")
        
        # Build node features (word embeddings)
        node_features = []
        token_indices = {}
        
        for i, token in enumerate(doc):
            if i >= self.max_length:
                break
            if token.is_space:
                continue
                
            token_indices[token.i] = len(node_features)
            
            # Use spaCy's word vector if available, otherwise random
            if token.has_vector:
                node_features.append(token.vector)
            else:
                # Use random embedding for unknown words
                node_features.append(np.random.randn(self.embedding_dim))
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edge indices and features
        edge_indices = []
        edge_features = []
        dependencies = []
        
        for token in doc:
            if token.i >= self.max_length or token.is_space:
                continue
                
            # Skip self-loops and ensure both tokens are in our vocabulary
            if token.i != token.head.i and token.i in token_indices and token.head.i in token_indices:
                src_idx = token_indices[token.i]
                dst_idx = token_indices[token.head.i]
                dep_type = token.dep_
                
                # Add forward edge
                edge_indices.append([src_idx, dst_idx])
                dependencies.append((src_idx, dst_idx, dep_type))
                
                # Add edge features if requested
                if self.include_edge_types:
                    edge_feat = self._encode_dependency_type(dep_type)
                    edge_features.append(edge_feat)
                
                # Add backward edge if bidirectional
                if self.bidirectional:
                    edge_indices.append([dst_idx, src_idx])
                    dependencies.append((dst_idx, src_idx, f"{dep_type}_rev"))
                    
                    if self.include_edge_types:
                        edge_feat_rev = self._encode_dependency_type(f"{dep_type}_rev")
                        edge_features.append(edge_feat_rev)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        if edge_features:
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_attr = None
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_features)
        )
        
        return graph_data, tokens, dependencies
    
    def _encode_dependency_type(self, dep_type: str) -> np.ndarray:
        """Encode dependency type as a feature vector."""
        # Simple one-hot encoding of common dependency types
        dep_types = [
            "nsubj", "dobj", "amod", "advmod", "prep", "pobj", "det", "aux",
            "cop", "conj", "cc", "punct", "root", "acl", "relcl", "xcomp",
            "ccomp", "mark", "case", "compound", "nummod", "appos", "nmod"
        ]
        
        # Add reverse types
        dep_types.extend([f"{dt}_rev" for dt in dep_types])
        
        # Create one-hot vector
        if dep_type in dep_types:
            idx = dep_types.index(dep_type)
            encoding = np.zeros(len(dep_types))
            encoding[idx] = 1.0
        else:
            # Unknown dependency type
            encoding = np.zeros(len(dep_types))
            encoding[-1] = 1.0  # Mark as unknown
        
        return encoding
    
    def batch_sentences_to_graphs(self, sentences: List[str]) -> List[Data]:
        """Convert a batch of sentences to graphs."""
        graphs = []
        for sentence in sentences:
            graph_data, _, _ = self.sentence_to_graph(sentence)
            graphs.append(graph_data)
        return graphs


class SentimentDataset:
    """Dataset for sentiment classification using dependency graphs."""
    
    def __init__(
        self,
        sentences: List[str],
        labels: List[int],
        graph_builder: DependencyGraphBuilder,
        max_length: int = 50
    ):
        """
        Initialize the sentiment dataset.
        
        Args:
            sentences: List of sentence texts
            labels: List of corresponding labels (0=negative, 1=positive)
            graph_builder: DependencyGraphBuilder instance
            max_length: Maximum sentence length
        """
        self.sentences = sentences
        self.labels = labels
        self.graph_builder = graph_builder
        self.max_length = max_length
        
        # Validate inputs
        if len(sentences) != len(labels):
            raise ValueError("Number of sentences and labels must match")
        
        # Build graphs
        self.graphs = []
        self.tokens_list = []
        self.dependencies_list = []
        
        logger.info(f"Building graphs for {len(sentences)} sentences...")
        for i, sentence in enumerate(sentences):
            try:
                graph_data, tokens, dependencies = self.graph_builder.sentence_to_graph(sentence)
                self.graphs.append(graph_data)
                self.tokens_list.append(tokens)
                self.dependencies_list.append(dependencies)
            except Exception as e:
                logger.error(f"Error processing sentence {i}: {sentence}. Error: {e}")
                # Create empty graph as fallback
                empty_graph = Data(
                    x=torch.zeros(1, self.graph_builder.embedding_dim),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    num_nodes=1
                )
                self.graphs.append(empty_graph)
                self.tokens_list.append([""])
                self.dependencies_list.append([])
        
        logger.info(f"Successfully built {len(self.graphs)} graphs")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> SentenceData:
        """Get a single sample."""
        return SentenceData(
            text=self.sentences[idx],
            label=self.labels[idx],
            graph=self.graphs[idx],
            tokens=self.tokens_list[idx],
            dependencies=self.dependencies_list[idx]
        )
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get dataset statistics."""
        num_nodes = [graph.num_nodes for graph in self.graphs]
        num_edges = [graph.edge_index.size(1) for graph in self.graphs]
        
        return {
            "num_samples": len(self),
            "num_classes": len(set(self.labels)),
            "class_distribution": {i: self.labels.count(i) for i in set(self.labels)},
            "avg_nodes_per_graph": np.mean(num_nodes),
            "avg_edges_per_graph": np.mean(num_edges),
            "max_nodes": max(num_nodes),
            "max_edges": max(num_edges),
            "min_nodes": min(num_nodes),
            "min_edges": min(num_edges),
        }


def create_synthetic_sentiment_dataset(
    num_samples: int = 1000,
    random_seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Create a synthetic sentiment dataset for demonstration.
    
    Args:
        num_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sentences, labels)
    """
    np.random.seed(random_seed)
    
    # Positive sentiment templates
    positive_templates = [
        "I love this amazing product!",
        "This is absolutely fantastic and wonderful.",
        "What a great experience this has been.",
        "I am so happy with this purchase.",
        "This exceeded all my expectations completely.",
        "Outstanding quality and excellent service.",
        "I would definitely recommend this to everyone.",
        "Perfect solution for my needs.",
        "Incredible value for the money spent.",
        "This made my day so much better."
    ]
    
    # Negative sentiment templates
    negative_templates = [
        "I hate this terrible product completely.",
        "This is absolutely awful and disappointing.",
        "What a horrible experience this has been.",
        "I am so frustrated with this purchase.",
        "This failed to meet any expectations.",
        "Poor quality and terrible service.",
        "I would never recommend this to anyone.",
        "Worst solution I have ever seen.",
        "Terrible value for the money spent.",
        "This ruined my day completely."
    ]
    
    sentences = []
    labels = []
    
    for _ in range(num_samples):
        if np.random.random() < 0.5:
            # Generate positive sample
            template = np.random.choice(positive_templates)
            label = 1
        else:
            # Generate negative sample
            template = np.random.choice(negative_templates)
            label = 0
        
        sentences.append(template)
        labels.append(label)
    
    return sentences, labels
