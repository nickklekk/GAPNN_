import math
import torch
import numpy as np

weighhts = torch.randn(784, 10) / math.sqrt(784)
weighhts.requires_grad_()
bias = torch.zeros(10, requires_grad_ = True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

scores = np.array([123, 456, 789])
softmax = np.exp(scores) / np.sum(np.exp(scores))
# 为了针对数值溢出，将每一个输出值减去输出值中最大的值
scores -= np.max(scores)
p = np.exp(scores) / np.sum(np.exp(scores))

