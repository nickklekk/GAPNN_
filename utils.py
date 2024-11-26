import os
import pickle
import random

import torch
import torch.nn.functional as F
import numpy as np
import dgl

from scipy import sparse as sp 
# from torch_geometric.utils import add_self_loops

import algorithms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_seed(seed=42):
    """Set random seed to enable reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        A number used to set the random seed

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


# def get_paths(start, neighbors, num_nodes):
#     """Return all the possible walks from a current node of length
#     num_nodes.
    
#     Parameters
#     ----------
#     start : int
#         Index of the starting node
#     neighbors : dict
#         Dictionary with the list of neighbors for each node
#     num_nodes : int
#         Length of the walks to be returned

#     Returns
#     -------
#     list
#         a list of all the possible walks, where each walk is also
#         stored in a list with num_nodes consecutive nodes
#     """
#     if num_nodes == 0:
#         return [[start]]
#     paths = []
#     for neighbor in neighbors[start]:
#         next_paths = get_paths(neighbor, neighbors, num_nodes-1)
#         for path in next_paths:
#             path.append(start)
#             paths.append(path)
#     return paths


def preprocess_graph(g, data_path, idx):
    g = g.int()
    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = g.edata['overlap_length'].float()
    ol_sim = g.edata['overlap_similarity']
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    ol_sim = (ol_sim - ol_sim.mean()) / ol_sim.std()
    g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    if 'y' not in g.edata:
        # TODO: Debug, or just delete this whole part eventually
        print('Deprecated - labels generated while creating DGL graph')
        try:
            nodes_gt, edges_gt = get_correct_ne(idx, data_path)
            # g.ndata['y'] = torch.tensor([1 if i in nodes_gt else 0 for i in range(g.num_nodes())], dtype=torch.float)
            g.edata['y'] = torch.tensor([1 if i in edges_gt else 0 for i in range(g.num_edges())], dtype=torch.float)
        except FileNotFoundError:
            # print("Solutions not generated")
            succs = pickle.load(open(f'{data_path}/info/{idx}_succ.pkl', 'rb'))
            edges = pickle.load(open(f'{data_path}/info/{idx}_edges.pkl', 'rb'))
            pos_str_edges, neg_str_edges = algorithms.get_gt_graph(g, succs, edges)
            edges_gt = pos_str_edges | neg_str_edges
            if 'solutions' not in os.listdir(data_path):
                os.mkdir(os.path.join(data_path, 'solutions'))
            pickle.dump(edges_gt, open(f'{data_path}/solutions/{idx}_edges.pkl', 'wb'))
            g.edata['y'] = torch.tensor([1 if i in edges_gt else 0 for i in range(g.num_edges())], dtype=torch.float)

    return g


def add_positional_encoding(g, pe_dim):
    """
        Initializing positional encoding with k-RW-PE
    """

    g.ndata['in_deg'] = g.in_degrees().float()
    g.ndata['out_deg'] = g.out_degrees().float()

    type_pe = 'PR'

    if type_pe == 'RW':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A @ Dinv  
        M = RW
        # Iterate
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(pe_dim-1):
            M_power = M_power @ M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pe'] = PE  

    if type_pe == 'PR':
        # k-step PageRank features
        A = g.adjacency_matrix(scipy_fmt="csr")
        D = A.sum(axis=1) # out degree
        Dinv = 1./ (D+1e-9); Dinv[D<1e-9] = 0 # take care of nodes without outgoing edges
        Dinv = sp.diags(np.squeeze(np.asarray(Dinv)), dtype=float) # D^-1 
        P = (Dinv @ A).T 
        n = A.shape[0]
        One = np.ones([n])
        x = One/ n
        PE = [] 
        alpha = 0.95 
        for _ in range(pe_dim): 
            x = alpha* P.dot(x) + (1.0-alpha)/n* One 
            PE.append(torch.from_numpy(x).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pe'] = PE  

    return g


def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'


def get_walks(idx, data_path):
    walk_path = os.path.join(data_path, f'solutions/{idx}_gt.pkl')
    walks = pickle.load(open(walk_path, 'rb'))
    return walks


def get_correct_ne(idx, data_path):
    nodes_path = os.path.join(data_path, f'solutions/{idx}_nodes.pkl')
    edges_path = os.path.join(data_path, f'solutions/{idx}_edges.pkl')
    nodes_gt = pickle.load(open(nodes_path, 'rb'))
    edges_gt = pickle.load(open(edges_path, 'rb'))
    return nodes_gt, edges_gt


def get_info(idx, data_path, type):
    info_path = os.path.join(data_path, 'info', f'{idx}_{type}.pkl')
    info = pickle.load(open(info_path, 'rb'))
    return info


def unpack_data(data, info_all, use_reads):
    idx, graph = data
    idx = idx if isinstance(idx, int) else idx.item()
    pred = info_all['preds'][idx]
    succ = info_all['succs'][idx]
    if use_reads:
        reads = info_all['reads'][idx]
    else:
        reads = None
    edges = info_all['edges'][idx]
    return idx, graph, pred, succ, reads, edges


def load_graph_data(num_graphs, data_path, use_reads):
    info_all = {
        'preds': [],
        'succs': [],
        'reads': [],
        'edges': [],
    }
    for idx in range(num_graphs):
        info_all['preds'].append(get_info(idx, data_path, 'pred'))
        info_all['succs'].append(get_info(idx, data_path, 'succ'))
        if use_reads:
            info_all['reads'].append(get_info(idx, data_path, 'reads'))
        info_all['edges'].append(get_info(idx, data_path, 'edges'))
    return info_all


def print_graph_info(idx, graph):
    """Print the basic information for the graph with index idx."""
    print('\n---- GRAPH INFO ----')
    print('Graph index:', idx)
    print('Number of nodes:', graph.num_nodes())
    print('Number of edges:', len(graph.edges()[0]))


def print_prediction(walk, current, neighbors, actions, choice, best_neighbor):
    """Print summary of the prediction for the current position."""
    print('\n-----predicting-----')
    print('previous:\t', None if len(walk) < 2 else walk[-2])
    print('current:\t', current)
    print('neighbors:\t', neighbors[current])
    print('actions:\t', actions.tolist())
    print('choice:\t\t', choice)
    print('ground truth:\t', best_neighbor)


def calculate_tfpn(edge_predictions, edge_labels):
    edge_predictions = torch.round(torch.sigmoid(edge_predictions))
    TP = torch.sum(torch.logical_and(edge_predictions==1, edge_labels==1)).item()
    TN = torch.sum(torch.logical_and(edge_predictions==0, edge_labels==0)).item()
    FP = torch.sum(torch.logical_and(edge_predictions==1, edge_labels==0)).item()
    FN = torch.sum(torch.logical_and(edge_predictions==0, edge_labels==1)).item()
    return TP, TN, FP, FN


def calculate_metrics(TP, TN, FP, FN):
    try:
        recall = TP / (TP + FP)
    except ZeroDivisionError:
        recall = 0
    try: 
        precision = TP / (TP + FN)
    except ZeroDivisionError:
        precision = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN) )
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, precision, recall, f1


def node_difficulty_measurer(data, label):
    neighbor_label, _ = add_self_loops(data.edge_index)
    neighbor_label[1] = label[neighbor_label[1]]
    # 节点的邻节点分布
    neighbor_label = torch.transpose(neighbor_label, 0, 1)
    index, count = torch.unique(neighbor_label,sorted=True, return_counts=True, dim=0)
    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()
    # 计算节点的邻居信息熵
    neighbor_class = neighbor_class[data.train_id]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)
    return local_difficulty.to(device)

def sort_training_nodes(data, label, embedding, alpha = 0.5):
    # 节点难度设置
    node_difficulty = difficulty_measurer(data, label, embedding, alpha)
    # 用torch进行分类
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset