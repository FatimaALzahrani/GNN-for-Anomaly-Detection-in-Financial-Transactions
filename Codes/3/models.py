import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling class imbalance
    
    Args:
        pred: Model predictions
        target: True labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        torch.Tensor: Focal loss
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()

class ImbalancedGCN(torch.nn.Module):
    """
    Graph Convolutional Network for imbalanced fraud detection
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(ImbalancedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ImbalancedGAT(torch.nn.Module):
    """
    Graph Attention Network for imbalanced fraud detection
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.3):
        super(ImbalancedGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def create_model(model_type, in_channels, out_channels=2, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('gcn' or 'gat')
        in_channels: Number of input features
        out_channels: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        torch.nn.Module: Created model
    """
    if model_type.lower() == 'gcn':
        return ImbalancedGCN(
            in_channels=in_channels,
            hidden_channels=kwargs.get('hidden_channels', 32),
            out_channels=out_channels,
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type.lower() == 'gat':
        return ImbalancedGAT(
            in_channels=in_channels,
            hidden_channels=kwargs.get('hidden_channels', 16),
            out_channels=out_channels,
            heads=kwargs.get('heads', 2),
            dropout=kwargs.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_config(model_type):
    """
    Get default configuration for a model type
    
    Args:
        model_type: Type of model ('gcn' or 'gat')
    
    Returns:
        dict: Model configuration
    """
    from config import MODEL_CONFIG
    return MODEL_CONFIG.get(model_type.lower(), {})