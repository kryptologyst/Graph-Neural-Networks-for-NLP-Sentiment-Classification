"""Graph Neural Network models for NLP tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import TransformerConv, GINConv
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """Attention-based graph pooling."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling to node features.
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch assignment for each node
            
        Returns:
            Pooled features [batch_size, input_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Compute attention weights
        attn_weights = F.softmax(self.attention(x), dim=0)
        
        # Apply attention pooling
        pooled = global_add_pool(x * attn_weights, batch)
        return pooled


class BaseGNNModel(nn.Module):
    """Base class for GNN models."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        pooling: str = "mean",
        use_residual: bool = True,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Pooling method
        if pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "max":
            self.pooling = global_max_pool
        elif pooling == "sum":
            self.pooling = global_add_pool
        elif pooling == "attention":
            self.pooling = AttentionPooling(hidden_dim)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.batch_norms = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get node embeddings without final classification."""
        return self.forward(x, edge_index, batch)


class SentenceGCN(BaseGNNModel):
    """Graph Convolutional Network for sentence classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GCN layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node
            
        Returns:
            Log probabilities [batch_size, output_dim]
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and x.size(1) == self.hidden_dim else None
            
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            # Add residual connection
            if residual is not None:
                x = x + residual
        
        # Global pooling
        if isinstance(self.pooling, AttentionPooling):
            pooled = self.pooling(x, batch)
        else:
            pooled = self.pooling(x, batch)
        
        # Classification
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)


class SentenceGraphSAGE(BaseGNNModel):
    """GraphSAGE for sentence classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(self.input_dim, self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim))
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GraphSAGE layers."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and x.size(1) == self.hidden_dim else None
            
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Global pooling
        if isinstance(self.pooling, AttentionPooling):
            pooled = self.pooling(x, batch)
        else:
            pooled = self.pooling(x, batch)
        
        # Classification
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)


class SentenceGAT(BaseGNNModel):
    """Graph Attention Network for sentence classification."""
    
    def __init__(self, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(self.input_dim, self.hidden_dim // num_heads, heads=num_heads, dropout=self.dropout))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GATConv(self.hidden_dim, self.hidden_dim // num_heads, heads=num_heads, dropout=self.dropout))
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GAT layers."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and x.size(1) == self.hidden_dim else None
            
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Global pooling
        if isinstance(self.pooling, AttentionPooling):
            pooled = self.pooling(x, batch)
        else:
            pooled = self.pooling(x, batch)
        
        # Classification
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights from all layers."""
        attention_weights = []
        
        for conv in self.convs:
            # Forward pass to get attention weights
            out, attn = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(attn)
            x = out
        
        return attention_weights


class SentenceGIN(BaseGNNModel):
    """Graph Isomorphism Network for sentence classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )))
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GIN layers."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and x.size(1) == self.hidden_dim else None
            
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Global pooling
        if isinstance(self.pooling, AttentionPooling):
            pooled = self.pooling(x, batch)
        else:
            pooled = self.pooling(x, batch)
        
        # Classification
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)


def create_model(model_name: str, **kwargs) -> BaseGNNModel:
    """
    Create a model instance by name.
    
    Args:
        model_name: Name of the model ("gcn", "graphsage", "gat", "gin")
        **kwargs: Model configuration parameters
        
    Returns:
        Model instance
    """
    model_classes = {
        "gcn": SentenceGCN,
        "graphsage": SentenceGraphSAGE,
        "gat": SentenceGAT,
        "gin": SentenceGIN
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_classes.keys())}")
    
    return model_classes[model_name](**kwargs)


def get_model_info(model: BaseGNNModel) -> Dict[str, Any]:
    """Get information about a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "output_dim": model.output_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "use_residual": model.use_residual,
        "use_batch_norm": model.use_batch_norm
    }
