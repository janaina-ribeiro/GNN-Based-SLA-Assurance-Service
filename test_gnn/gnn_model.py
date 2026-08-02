from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GraphConv

try:
    from torch_geometric_temporal.nn import TGCN, A3TGCN, DCRNN
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    TGCN = A3TGCN = DCRNN = None


class DelayGNN(nn.Module):
    """
    Class for a Graph Neural Network (GNN) model to predict network delays.
    
    Supports both spatial and spatio-temporal architectures:
    - Spatial: gat, sage, graphsaint, pinsage
    - Spatio-Temporal: tgcn, a3tgcn, dcrnn
    
    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int, optional
        Number of hidden units in each GNN layer. Default is 64.
    num_layers : int, optional
        Number of GNN layers. Default is 2.
    num_classes : int, optional
        Number of output classes. Default is 2.
    conv_type : str, optional
        Type of convolution: 'gat', 'sage', 'graphsaint', 'pinsage', 'tgcn', 'a3tgcn', 'dcrnn'. Default is 'sage'.
    dropout : float, optional
        Dropout rate between layers. Default is 0.2.
    gat_heads : int, optional
        Number of attention heads for GAT convolution. Default is 2.
    Returns
    -------
    torch.nn.Module
        The GNN model.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        conv_type: str = "sage",
        dropout: float = 0.2,
        gat_heads: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        conv_type = conv_type.lower()
        
        spatial_types = {"gat", "sage", "graphsaint", "pinsage"}
        temporal_types = {"tgcn", "a3tgcn", "dcrnn"}
        supported = spatial_types | temporal_types
        
        if conv_type not in supported:
            raise ValueError(f"conv_type must be one of {supported}")
        
        if conv_type in temporal_types and not TEMPORAL_AVAILABLE:
            raise ImportError(
                f"'{conv_type}' requires torch_geometric_temporal. "
                "Install with: pip install torch-geometric-temporal"
            )
        
        self.conv_type = conv_type
        self.dropout = dropout
        self.gat_heads = gat_heads
        self.is_temporal = conv_type in temporal_types

        if conv_type == "gat":
            self.actual_hidden = (hidden_channels // gat_heads) * gat_heads
            if self.actual_hidden == 0:
                self.actual_hidden = gat_heads
        else:
            self.actual_hidden = hidden_channels

        if self.is_temporal:
            self.temporal_conv = self._make_temporal_conv(conv_type, in_channels, self.actual_hidden)
            self.h = None
        else:
            self.convs = nn.ModuleList()
            for layer in range(num_layers):
                in_c = in_channels if layer == 0 else self.actual_hidden
                out_c = self.actual_hidden
                self.convs.append(self._make_conv(conv_type, in_c, out_c))
        
        self.classifier = nn.Linear(self.actual_hidden, num_classes)

    def _make_conv(
        self, conv_type: str, in_channels: int, out_channels: int
    ) -> nn.Module:
        """Create spatial convolution layer."""
        if conv_type == "sage":
            return SAGEConv(in_channels, out_channels, aggr="mean")
        
        if conv_type == "pinsage":
            return SAGEConv(in_channels, out_channels, aggr="mean")
        
        if conv_type == "graphsaint":
            return GraphConv(in_channels, out_channels)

        if conv_type == "gat":
            out_per_head = max(1, out_channels // self.gat_heads)
            return GATConv(
                in_channels,
                out_per_head,
                heads=self.gat_heads,
                concat=True,
                dropout=self.dropout,
            )
        
        raise ValueError(f"Unknown conv_type: {conv_type}")

    def _make_temporal_conv(
        self, conv_type: str, in_channels: int, out_channels: int
    ) -> nn.Module:
        """Create spatio-temporal convolution layer."""
        if conv_type == "tgcn":
            return TGCN(in_channels=in_channels, out_channels=out_channels)
        
        elif conv_type == "a3tgcn":
           
            return A3TGCN(
                in_channels=in_channels,
                out_channels=out_channels,
                periods=1,  
            )
        
        elif conv_type == "dcrnn":
            return DCRNN(
                in_channels=in_channels,
                out_channels=out_channels,
                K=2, 
            )
        
        else:
            raise ValueError(f"Unknown temporal conv_type: {conv_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        For temporal models, processes the input as a sequence and uses recurrent connections.
        For spatial models, applies multiple GNN layers.
        """
        if self.is_temporal:
            
            try:
                h = self.temporal_conv(
                    x,
                    edge_index,
                    edge_weight=edge_weight,
                    H=self.h,
                )
            except TypeError:
                h = self.temporal_conv(x, edge_index, H=self.h)

            self.h = h.detach()  
            x = h
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        else:
            for conv in self.convs:
                x = conv(x, edge_index, edge_weight=edge_weight) if hasattr(conv, 'edge_weight') else conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def reset_hidden_state(self):
        """Reset hidden state for temporal models (call between batches/sequences)."""
        if self.is_temporal:
            self.h = None
