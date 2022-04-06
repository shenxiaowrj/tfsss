# _*_coding:utf-8_*_
#shenxiao
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

# 关于word embedding,以序列建模为例
#abs 构建输入
# 考虑source sentence 和 target sentence
# 构建序列，序列的字符以其词表中的索引的形式表示
batch_size = 2

'''随机整数生成函数 torch.randint
最小值
最大值
batch的大小
'''
# src_len = torch.randint(2,5,(batch_size,))  #很奇怪 可以随便加逗号
# tgt_len = torch.randint(2,5,(batch_size,))
# print(src_len)
# print(tgt_len)

# tensor([4, 4])
# tensor([4, 2])
#
# tensor([2, 2])
# tensor([3, 4])
# 每次都是随机进行变化 因为bacth_size设置为2 所以为两个数
'''代表的具体含义是:第一个句子的长度 第二个句子的长度'''

#随机变化不太好，所以写成固定的形式
src_len = torch.Tensor([2,4]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)

#序列的最大长度
max_num_src_words = 8
max_num_tgt_words = 8

#利用上边实现的单词索引构造成源句子和目标句子 并且做padding 默认值为0
#首先实现这个句子

# src_seq = [torch.randint(1, max_num_src_words, (L,)) for L in src_len]
# tgt_seq = [torch.randint(1, max_num_tgt_words, (L,)) for L in tgt_len]

# [tensor([5, 7]), tensor([2, 7, 7, 7])]
# [tensor([4, 5, 1, 1]), tensor([3, 6, 6])]

#之后增加padding 功能
'''使用torch.nn.functional.pad 函数
要pad进去的值
pad之后的长度
'''

#序列的最大长度
max_num_src_len = 5
max_num_tgt_len = 5

# src_seq = [F.pad(torch.randint(1, max_num_src_words, (L,)),(0, max_num_src_len-L)) for L in src_len]
# tgt_seq = [F.pad(torch.randint(1, max_num_tgt_words, (L,)),(0,max_num_tgt_len-L)) for L in tgt_len]

# [tensor([6, 1, 0, 0, 0]), tensor([2, 2, 1, 1, 0])]
# [tensor([3, 3, 4, 6, 0]), tensor([7, 5, 6, 0, 0])]

#接下来，我们需要将分开的tensor变成二维的tensor 方便做批量计算
#先用unqueeze增加一个维度
#之后在增加的维度上进行concat

src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)),(0, max_num_src_len-L)) ,0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)),(0,max_num_tgt_len-L)) ,0) for L in tgt_len])

# tensor([[3, 1, 0, 0, 0],
#         [2, 4, 1, 3, 0]])
# tensor([[3, 5, 5, 4, 0],
#         [2, 4, 7, 0, 0]])

# print(src_seq)
# print(tgt_seq)

# 构造word embedding
# 首先看一下torch.nn.embedding工具
# 首先是单词表的最大长度
# 之后是模型的特征维度

model_dim = 5
src_embedding_table = nn.Embedding(max_num_src_words+1 , model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1 , model_dim)
#再一个实例之后条用一个（） 就是采用forward形式
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
# print(src_embedding_table)
# print(src_embedding_table.weight)
# print(src_seq)
# print(src_embedding)

# Embedding(9, 5)
# Parameter containing:
# tensor([[ 0.7967,  0.1918,  0.5071, -0.0493, -0.6663],
#         [-1.8225,  2.1890, -0.1750, -0.4994,  1.2719],
#         [ 2.2231, -0.5911, -0.2809, -1.1796, -0.0734],
#         [-1.8649, -0.5550,  0.7715,  1.9463, -0.0761],
#         [-1.0537,  0.1263,  0.0989, -0.0646, -0.8993],
#         [-1.1266,  0.7425,  0.3982, -0.8240, -1.9145],
#         [ 1.2064, -0.1607, -0.2356, -1.7646,  0.4217],
#         [ 0.9517, -0.0028,  0.4348, -0.5360,  0.3449],
#         [-1.2251,  0.3034, -0.8297, -0.4218, -0.3074]], requires_grad=True)
#随机生成每一个词的初始化的特征矩阵

'''构建出来一个九乘以五的矩阵 第一行留给padding列 剩下的是每个词占一列'''#shenxiao
#第0列是谁是怎么固定的？在embadding源码里边设置好的吧

# tensor([[4, 3, 0, 0, 0],
#         [6, 3, 5, 7, 0]])
# tensor([[[-1.0537,  0.1263,  0.0989, -0.0646, -0.8993],
#          [-1.8649, -0.5550,  0.7715,  1.9463, -0.0761],
#          [ 0.7967,  0.1918,  0.5071, -0.0493, -0.6663],
#          [ 0.7967,  0.1918,  0.5071, -0.0493, -0.6663],
#          [ 0.7967,  0.1918,  0.5071, -0.0493, -0.6663]],
#
#         [[ 1.2064, -0.1607, -0.2356, -1.7646,  0.4217],
#          [-1.8649, -0.5550,  0.7715,  1.9463, -0.0761],
#          [-1.1266,  0.7425,  0.3982, -0.8240, -1.9145],
#          [ 0.9517, -0.0028,  0.4348, -0.5360,  0.3449],
#          [ 0.7967,  0.1918,  0.5071, -0.0493, -0.6663]]],
#        grad_fn=<EmbeddingBackward>)

# 每一行代表一个词的特征维度 这是设置的特征维度为5
#shenxiao 2022-04-04

