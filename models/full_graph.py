import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn import GraphConv    

import layers


class GraphGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc):
        super().__init__()
        #self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.linear_pe = nn.Linear(nb_pos_enc + 2, hidden_features) # PE + degree_in + degree_out 
        #self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GraphGatedGCN(num_layers, hidden_features, batch_norm)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)
    
    # def softmax(x):
    #     X_exp = torch.exp(x)
    #     partition = X_exp.sum(1, keepdim = True)
    #     return X_exp / partition

    def forward(self, graph, x, e, pe):
        x = self.linear_pe(pe)

        # e = torch.cat((e, e), dim=0) 
        
        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class SageModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear_pe = nn.Linear(nb_pos_enc + 2, hidden_features) # PE + degree_in + degree_out 
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.SAGE_processor(num_layers, hidden_features, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        x = self.linear_pe(pe)
        # x = self.linear1_node(pe)
        # x = torch.relu(x)
        # x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class GCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear_pe = nn.Linear(nb_pos_enc + 2, hidden_features) # PE + degree_in + degree_out 
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GCN_processor(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        x = self.linear_pe(pe)
        # x = self.linear1_node(pe)
        # x = torch.relu(x)
        # x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class GATModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear_pe = nn.Linear(nb_pos_enc + 2, hidden_features) # PE + degree_in + degree_out 
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GAT_processor(num_layers, hidden_features, dropout=dropout, num_heads=3)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        x = self.linear_pe(pe)       
        # x = self.linear1_node(pe)
        # x = torch.relu(x)
        # x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores  


class SymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear_pe = nn.Linear(nb_pos_enc + 2, hidden_features) # PE + degree_in + degree_out 
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.SymGatedGCN_processor(num_layers, hidden_features, batch_norm)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        x = self.linear_pe(pe) 
        # x = self.linear1_node(pe)
        # x = torch.relu(x)
        # x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores

class PathNNModel(nn.Module):
    def __init__(self,node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True ):
        super().__init__()
        self.num_layers = num_layers
        self.directed = directed
        self.linear_pe = nn.Linear(nb_pos_enc + 2, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.PATHNN_processor(num_layers, node_features, edge_features, hidden_features, hidden_edge_features,dropout=dropout)

        self.hidden_edge_scores = hidden_edge_scores
        self.nb_pos_enc = nb_pos_enc
        self.layers = nn.ModuleList()
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)
        
        
        # current_node_feats = node_features
        # current_edge_feats = edge_features

        # for i in range(num_layers):
        #     layer = layers.PATHNN_processor(current_node_feats, current_edge_feats, hidden_features, hidden_edge_features, batch_norm, dropout)
        #     self.layers.append(layer)
        #     current_node_feats = hidden_features
        #     current_edge_feats = hidden_edge_features

        # 位置编码  
        self.pos_enc = nn.Embedding(nb_pos_enc, hidden_features) if nb_pos_enc > 0 else None  
          
        # 隐藏层边得分  
        self.edge_scores = nn.Linear(hidden_edge_features, 1) if hidden_edge_scores else None  
  
    def forward(self, graph, x, e, pe):  
        # x = self.linear_pe(pe)
        # 初始化节点和边特征  
    
        # 前向传播  
        # for i, layer in enumerate(self.layers):  
        #     x, e = layer(g, x, e)  
              
        #     # 如果需要，添加位置编码  
        #     if self.pos_enc and paths is not None:  
        #         path_lengths = [len(path) for path in paths]  
        #         max_length = max(path_lengths)  
        #         pos_encodings = self.pos_enc(torch.arange(max_length).to(h.device))  
        #         for path_idx, path in enumerate(paths):  
        #             path_length = path_lengths[path_idx]  
        #             h[path] += pos_encodings[:path_length]  # 添加位置编码  
          
        # 如果需要，计算隐藏层边得分  
        x = self.linear_pe(pe)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores 


# class PathNNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim,cutoff, n_classes, dropout, device, residuals= True, encode_disatances = False):
#         super(PathNN, self).__init__()
#         self.cutoff = cutoff
#         self.device = device
#         self.residuals = residuals
#         self.dropout = dropout
#         self.encode_disatances = encode_disatances

#         #Feature Encoder that projects initial node representation to d-dim space
#         self.feature_encoder = Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
#                                           Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
#         conv_class = PathConv

#         #1 shared LSTM across layers
#         if encode_distances : 
#             self.distance_encoder = nn.Embedding(cutoff, hidden_dim)
#             self.lstm = nn.LSTM(input_size = hidden_dim * 2, hidden_size = hidden_dim , batch_first=True, bidirectional = False, num_layers = 1, bias = True)
#         else : 
#             self.lstm = nn.LSTM(input_size = hidden_dim , hidden_size = hidden_dim , batch_first=True, bidirectional = False, num_layers = 1, bias = True)
        
#         self.convs = nn.ModuleList([])
#         for _ in range(self.cutoff - 1) : 
#             bn = nn.BatchNorm1d(hidden_dim)
#             self.convs.append(conv_class(hidden_dim, self.lstm, bn, residuals = self.residuals, dropout = self.dropout))

#         self.hidden_dim = hidden_dim
#         self.linear1 = Linear(hidden_dim, hidden_dim)
#         self.linear2 = Linear(hidden_dim, n_classes)

#         self.reset_parameters()

#     def reset_parameters(self):
#         for c in self.feature_encoder.children():
#             if hasattr(c, 'reset_parameters'):
#                 c.reset_parameters()
#         self.lstm.reset_parameters()
#         for conv in self.convs : 
#             conv.reset_parameters()
#         self.linear1.reset_parameters()
#         self.linear2.reset_parameters()     
#         if hasattr(self, "distance_encoder") : 
#             nn.init.xavier_uniform_(self.distance_encoder.weight.data)

#     def forward(self, data):
#          #Projecting init node repr to d-dim space
#         # [n_nodes, hidden_size]
#         h = self.feature_encoder(data.x)

#         #Looping over layers
#         for i in range(self.cutoff-1) :
#             if self.encode_distances : 
#                 #distance encoding with shared distance embedding
#                 # [n_paths, path_length, hidden_size]
#                 dist_emb = self.distance_encoder(getattr(data, f"sp_dists_{i+2}"))
#             else : 
#                 dist_emb = None
#             # [n_nodes, hidden_size]
#             h = self.convs[i](h, getattr(data, f"path_{i+2}"), dist_emb)
        
#         #Readout sum function
#         h = global_add_pool(h, data.batch)

#         #Prediction
#         h = F.relu(self.linear1(h))
#         h = F.dropout(h, training=self.training, p=self.dropout)
#         scores = self.linear2(h)
#         return scores






# class VQGraphModel(nn.Module):


    
#         return 0