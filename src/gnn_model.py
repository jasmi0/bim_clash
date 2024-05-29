"""
Graph Neural Network Model for BIM Clash Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class ClashDetectionGNN(nn.Module):
    """
    Graph Neural Network for detecting clashes in BIM models
    Uses Graph Attention Networks for better relationship modeling
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4
    ):
        """
        Initialize Clash Detection GNN
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden units
            out_channels: Number of output features (1 for binary clash detection)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            heads: Number of attention heads for GAT layers
        """
        super(ClashDetectionGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.conv_input = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # Hidden layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        
        # Output layer
        self.conv_output = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        # Edge prediction layers (for clash detection between pairs of nodes)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads if _ < num_layers - 1 else hidden_channels))
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features (batch_size, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_features)
            
        Returns:
            Node embeddings and clash predictions
        """
        # Input layer
        x = self.conv_input(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i + 1](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_output(x, edge_index)
        x = self.batch_norms[-1](x)
        x = F.elu(x)
        
        return x
    
    def predict_clashes(self, x, edge_index, edge_attr=None):
        """
        Predict clashes between pairs of nodes (with sigmoid for inference)
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            
        Returns:
            Clash predictions for each edge (probabilities)
        """
        # Get logits
        logits = self.predict_clash_logits(x, edge_index, edge_attr)
        
        # Apply sigmoid for probabilities
        clash_probs = torch.sigmoid(logits)
        
        return clash_probs
    
    def predict_clash_logits(self, x, edge_index, edge_attr=None):
        """
        Predict clash logits (for training with BCEWithLogitsLoss)
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            
        Returns:
            Clash logits for each edge (before sigmoid)
        """
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index, edge_attr)
        
        # Predict clashes for each edge
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Concatenate embeddings of connected nodes
        edge_embeddings = torch.cat([
            node_embeddings[source_nodes],
            node_embeddings[target_nodes]
        ], dim=1)
        
        # Return logits (no sigmoid)
        logits = self.edge_mlp(edge_embeddings)
        
        return logits


class ClashDetectionModel:
    """Wrapper class for training and inference"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        pos_weight: float = None,
        device: str = None
    ):
        """Initialize model wrapper"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ClashDetectionGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        ).to(self.device)
        
        # Add L2 regularization (weight_decay) to prevent overfitting
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Use BCEWithLogitsLoss with optional pos_weight for imbalanced data
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight:.4f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss without pos_weight")
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def train_step(self, data, labels):
        """Single training step with gradient clipping for stability"""
        self.model.train()
        self.optimizer.zero_grad()
        
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        # Apply label smoothing to prevent overconfidence
        # Smooth labels: 1 -> 0.95, 0 -> 0.05
        smoothed_labels = labels * 0.9 + 0.05
        
        # Forward pass - get logits for BCEWithLogitsLoss
        logits = self.model.predict_clash_logits(
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )
        
        # Calculate loss (BCEWithLogitsLoss applies sigmoid internally)
        loss = self.criterion(logits.squeeze(), smoothed_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, data):
        """Make predictions"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict_clashes(
                data.x,
                data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None
            )
        
        return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")
