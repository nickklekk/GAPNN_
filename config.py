################################################################################

# Edit these three dictionaries to specify graphs to train/validation/test
# Assemblies will be constructed only for the graphs in the test_dict

# To train/validate/test on multiple chromosomes, put the as separate
# entries in the dictionaries
# E.g., to train on 1 chr19 graph and 2 chr20 graphs: 
# _train_dict = {'chr19': 1, 'chr20': 2}

# To test on real chromosome put "_r" suffix. Don't put value higher than 1,
# since there is only 1 real HiFi dataset for each chromosomes
# E.g., to test on real chr21:
# _test_dict = {'chr21_r': 1}

_train_dict = {'chr19': 5}
_valid_dict = {'chr19': 2}
_test_dict = {'chr1_r': 1}

################################################################################

def get_config():
    return {
        'train_dict': _train_dict,
        'valid_dict': _valid_dict,
        'test_dict' : _test_dict
    }

#  class PathNNLayer(nn.Module):  
#     def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, batch_norm, dropout):  
#         super(PathNNLayer, self).__init__()  
#         self.batch_norm = batch_norm  
#         self.dropout = dropout  
          
#         self.linear_node = nn.Linear(node_features, hidden_features)  
#         self.linear_edge = nn.Linear(edge_features, hidden_edge_features)  
          
#         if batch_norm:  
#             self.bn_node = nn.BatchNorm1d(hidden_features)  
#             self.bn_edge = nn.BatchNorm1d(hidden_edge_features)  
          
#         self.dropout_layer = nn.Dropout(dropout) if dropout else None  
  
#     def forward(self, g, h, e):  
#         # Node update  
#         h = self.linear_node(h)  
#         if self.batch_norm:  
#             h = self.bn_node(h)  
#         h = F.relu(h)  
#         if self.dropout_layer:  
#             h = self.dropout_layer(h)  
          
#         # Edge update using src and dst node features  
#         g.srcdata['h'] = h  
#         g.dstdata['h'] = h  
#         g.edata['e'] = self.linear_edge(e)  
#         if self.batch_norm:  
#             g.edata['e'] = self.bn_edge(g.edata['e'])  
#         g.edata['e'] = F.relu(g.edata['e'])  
#         if self.dropout_layer:  
#             g.edata['e'] = self.dropout_layer(g.edata['e'])  
          
#         # Message passing  
#         g.update_all(dgl.function.u_mul_e('h', 'e', 'm'), dgl.function.sum('m', 'h'))  
          
#         return h, g.edata['e']



# class PathNNProcessor(nn.Module):  
#     def __init__(self, num_layers, node_features, edge_features, hidden_features, hidden_edge_features, batch_norm, dropout):  
#         super(PathNNProcessor, self).__init__()  
#         self.layers = nn.ModuleList()  
          
#         current_node_feats = node_features  
#         current_edge_feats = edge_features  
          
#         for i in range(num_layers):  
#             layer = PathNNLayer(current_node_feats, current_edge_feats, hidden_features, hidden_edge_features, batch_norm, dropout)  
#             self.layers.append(layer)  
#             current_node_feats = hidden_features  
#             current_edge_feats = hidden_edge_features  
  
#     def forward(self, g, h, e):  
#         for layer in self.layers:  
#             h, e = layer(g, h, e)  
#         return h, e


# class PathNNModel(nn.Module):  
#     def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None):  
#         super(PathNNModel, self).__init__()  
#         self.linear_pe = nn.Linear(nb_pos_enc, hidden_features)  # Position Encoding  
#         self.processor = PathNNProcessor(num_layers, node_features + nb_pos_enc, edge_features, hidden_features, hidden_edge_features, batch_norm, dropout)  
#         self.predictor = nn.Linear(hidden_features, hidden_edge_scores)  # Edge score predictor  
  
#     def forward(self, graph, x, e, pe):  
#         x = torch.cat([x, pe], dim=1)  # Concatenate position encoding with node features  
#         x, e = self.processor(graph, x, e)  
#         scores = self.predictor(x)  # Predict edge scores based on node features  
#         return scores



# {
#   "model_name" : "resnet18",
#   "client_num" : 10,
#   "type" : "cifar",
#   "global_epochs" : 20,
#   "local_epochs" : 3,
#   "k" : 6,
#   "batch_size" : 32,
#   "lr" : 0.001,
#   "momentum" : 0.0001,
#   "lambda" : 0.1 
# }


# # 全局聚合模型
# # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
# def model_aggregate(self, weight_accumulator):
#   # 遍历服务器的全局模型
#   for name, data in self.global_model.state_dict().items():
#     # 更新每一层乘上学习率
#     update_per_layer = weight_accumulator[name] * self.conf["lambda"]
#     # 累加和
#     if data.type() != update_per_layer.type():
#       	# 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
#       	data.add_(update_per_layer.to(torch.int64))
#       else:
#         data.add_(update_per_layer)


# # 评估函数
#     def model_eval(self):
#         self.global_model.eval()    # 开启模型评估模式（不修改参数）
#         total_loss = 0.0
#         correct = 0
#         dataset_size = 0
#         # 遍历评估数据集合
#         for batch_id, batch in enumerate(self.eval_loader):
#             data, target = batch
#             # 获取所有的样本总量大小
#             dataset_size += data.size()[0]
#             # 存储到gpu
#             if torch.cuda.is_available():
#                 data = data.cuda()
#                 target = target.cuda()
#             # 加载到模型中训练
#             output = self.global_model(data)
#             # 聚合所有的损失 cross_entropy交叉熵函数计算损失
#             total_loss += torch.nn.functional.cross_entropy(
#                 output,
#                 target,
#                 reduction='sum'
#             ).item()
#             # 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
#             pred = output.data.max(1)[1]
#             # 统计预测结果与真实标签target的匹配总个数
#             correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
#         acc = 100.0 * (float(correct) / float(dataset_size))    # 计算准确率
#         total_1 = total_loss / dataset_size                     # 计算损失值
#         return acc, total_1
# ————————————————

#                             版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
# 原文链接：https://blog.csdn.net/weixin_43988498/article/details/119540093


# def local_train(self, model):
#         # 整体的过程：拉取服务器的模型，通过部分本地数据集训练得到
#         for name, param in model.state_dict().items():
#             # 客户端首先用服务器端下发的全局模型覆盖本地模型
#             self.local_model.state_dict()[name].copy_(param.clone())
#         # 定义最优化函数器用于本地模型训练
#         optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

#         # 本地训练模型
#         self.local_model.train()        # 设置开启模型训练（可以更改参数）
#         # 开始训练模型
#         for e in range(self.conf["local_epochs"]):
#             for batch_id, batch in enumerate(self.train_loader):
#                 data, target = batch
#                 # 加载到gpu
#                 if torch.cuda.is_available():
#                     data = data.cuda()
#                     target = target.cuda()
#                 # 梯度
#                 optimizer.zero_grad()
#                 # 训练预测
#                 output = self.local_model(data)
#                 # 计算损失函数 cross_entropy交叉熵误差
#                 loss = torch.nn.functional.cross_entropy(output, target)
#                 # 反向传播
#                 loss.backward()
#                 # 更新参数
#                 optimizer.step()
#             print("Epoch %d done" % e)
#         # 创建差值字典（结构与模型参数同规格），用于记录差值
#         diff = dict()
#         for name, data in self.local_model.state_dict().items():
#             # 计算训练后与训练前的差值
#             diff[name] = (data - model.state_dict()[name])
#         print("Client %d local train done" % self.client_id)
#         # 客户端返回差值
#         return diff


#     # 读取配置文件
#     with open(args.conf, 'r') as f:
#         conf = json.load(f)

#     # 获取数据集, 加载描述信息
#     train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

#     # 开启服务器
#     server = Server(conf, eval_datasets)
#     # 客户端列表
#     clients = []

#     # 添加10个客户端到列表
#     for c in range(conf["no_models"]):
#         clients.append(Client(conf, server.global_model, train_datasets, c))

#     # 全局模型训练
#     for e in range(conf["global_epochs"]):
#         print("Global Epoch %d" % e)
#         # 每次训练都是从clients列表中随机采样k个进行本轮训练
#         candidates = random.sample(clients, conf["k"])
#         print("select clients is: ")
#         for c in candidates:
#             print(c.client_id)

#         # 权重累计
#         weight_accumulator = {}

#         # 初始化空模型参数weight_accumulator
#         for name, params in server.global_model.state_dict().items():
#             # 生成一个和参数矩阵大小相同的0矩阵
#             weight_accumulator[name] = torch.zeros_like(params)

#         # 遍历客户端，每个客户端本地训练模型
#         for c in candidates:
#             diff = c.local_train(server.global_model)

#             # 根据客户端的参数差值字典更新总体权重
#             for name, params in server.global_model.state_dict().items():
#                 weight_accumulator[name].add_(diff[name])

#         # 模型参数聚合
#         server.model_aggregate(weight_accumulator)

#         # 模型评估
#         acc, loss = server.model_eval()
# ————————————————

#                             版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
# 原文链接：https://blog.csdn.net/weixin_43988498/article/details/119540093