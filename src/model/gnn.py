import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

def build_gnn(num_classes: int = 10, in_channels: int = 3, hidden_channels: int = 64, **kwargs) -> nn.Module:
    """Simple Graph Convolutional Network for super‑pixel graph data.
    Expects a torch_geometric.data.Data object with x (node features) and edge_index.
    The forward method returns class logits.
    """
    class GNNBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.fc = nn.Linear(hidden_channels, num_classes)
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)  # (B, hidden_channels)
            return self.fc(x)
    return GNNBackbone()
