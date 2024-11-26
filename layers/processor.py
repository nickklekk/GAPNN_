import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
# DGL中自带的三个模型
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv

import layers


class SymGatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.SymGatedGCN(hidden_features, hidden_features, batch_norm) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h,e


class GraphGatedGCN(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GatedGCN_1d(hidden_features, hidden_features, batch_norm) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
            # h = F.relu(h)
            # e = F.relu(e)
        return h, e


class GAT_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout=0.0, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        print(f'Using dropout:', dropout)
        self.convs = nn.ModuleList([
            GATConv(hidden_features, hidden_features, num_heads=self.num_heads, feat_drop=dropout, attn_drop=0) for _ in range(num_layers)
        ])
        self.linears = nn.ModuleList([
            nn.Linear(self.num_heads * hidden_features, hidden_features) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)-1):
            heads = self.convs[i](graph, h)
            h = torch.cat(tuple(heads[:,j,:] for j in range(self.num_heads)), dim=1)
            h = self.linears[i](h)
            h = F.relu(h)
        heads = self.convs[-1](graph, h)
        h = torch.cat(tuple(heads[:,j,:] for j in range(self.num_heads)), dim=1)
        h = self.linears[-1](h)
        return h, e


class GCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            GraphConv(hidden_features, hidden_features, weight=True, bias=True) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)-1):
            h = F.relu(self.convs[i](graph, h))
        h = self.convs[-1](graph, h)
        return h, e


class SAGE_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(hidden_features, hidden_features, 'mean', feat_drop=dropout) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)-1):
            h = F.relu(self.convs[i](graph, h))
        h = self.convs[-1](graph, h)
        return h, e


class PATHNN_processor(nn.Module):
    def __init__(self, num_layers, node_feats, edge_feats, hidden_feats, hidden_edge_feats, dropout):  
        # super().__init__()  
        # self.node_conv = GraphConv(node_feats, hidden_feats)
        # self.convs = nn.ModuleList([
        #     GraphConv(hidden_features, hidden_features, weight=True, bias=True) for _ in range(num_layers)
        # ])  
        # self.edge_conv = GraphConv(edge_feats, hidden_edge_feats)  
        # self.batch_norm = nn.BatchNorm1d(hidden_feats) if batch_norm else None  
        # self.dropout = nn.Dropout(dropout) if dropout is not None else None  
        super(PATHNN_processor, self).__init__()  
        self.layers = nn.ModuleList()  
          
        

    def forward(self, g, h, e):  
        # # 节点特征变换  
        # h = F.relu(self.node_conv(g, h))  
    
        # # 边特征变换  
        # e = F.relu(self.edge_conv(g, e)) 

        # for i in range(len(self.convs)):
        #     h, e = self.convs[i](graph, h, e)
            # h = F.relu(h)
            # e = F.relu(e)
     
          
        for layer in self.layers:  
            h, e = layer(g, h, e)  
        return h, e
 

