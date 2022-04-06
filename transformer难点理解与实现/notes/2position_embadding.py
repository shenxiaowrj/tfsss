# _*_coding:utf-8_*_
#shenxiao
#书接上回，先来弄一下复习一下word_embadding

import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

src_len = torch.Tensor([2,4]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)

#批量大小
batch_size = 2

#单词表大小
max_num_src_words = 8
max_num_tgt_words = 8

#序列的最大的长度
max_src_seq_len = 5
max_tgt_seq_len = 5

#单词索引构成的句子 并作了padding  padding的值为0
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_len-L)),0)
                     for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L,)),(0,max_tgt_seq_len-L)),0)
                     for L in tgt_len])

# print(src_seq)
# print(tgt_seq)

#一个词的特征维度数
model_dim = 8
#构造embedding
src_embedding_table = nn.Embedding(max_num_src_words+1,model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1,model_dim)
# print(src_embedding_table)
# print(tgt_embedding_table)
#实例化类
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
# print(src_embedding)
# print(tgt_embedding)

'''接下来 我们开始构建position embedding'''#shenxiao

#最大的位置编码的长度
    #这个值是不是要和最大的序列的长度严格保持一样啊？
max_position_len = 5

#构造position embadding
#首先定义公式里的后面的那个部分
pos_mat = torch.arange(max_position_len).reshape((-1,1))  #将其转为竖着的矩阵
# print(pos_mat)
# i_mat = torch.arange(0, 8, 2).reshape(1,-1) / model_dim    #这里的reshape就体现出了上一个reshape的作用
                                                    #其让二者进行相除，以此来对矩阵的形式进行变化
                                                    #不对 不对 这里使用的是点积的形式和倒数来构成除法
                                                    #而这样reshape的原因是利用广播机制来进行处理
                                                    #broadcast 首先将两个向量比较少的那个维度广播成比较大的那个维度 之后再进行点积运算
#上边那是i 之后加入pow对 上边的值进行指数的运算
i_mat = torch.pow(10000 , torch.arange(0 , 8, 2).reshape((1 ,-1)) /model_dim)
# print(i_mat)
# i_mat = torch.arange(0 , 8, 2).reshape((1 ,-1)) / model_dim
# print(i_mat)
pe_embedding_table = torch.zeros(max_position_len,model_dim)
# print(pe_embedding_table)
# a = torch.sin(pos_mat / i_mat)
# print(a)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# print(pe_embedding_table)
# RuntimeError: The expanded size of the tensor (3) must match the existing size (4) at non-singleton dimension 1.  Target sizes: [5, 3].  Tensor sizes: [5, 4]
# 写错了 model_dim 是8

#构建pe_embadding 借用pytorch的embadding API 直接构建出来一个embadding
#之后利用pytorch API 中的nn.Parameter 传入构建好的位置参数 pe_embedding_table
#以此来构建一个完整的pe_embadding 之后就可以传入对应的位置参数 以此可以来构建src tgt不同的embadding
pe_embadding = nn.Embedding(max_position_len,model_dim)
pe_embadding.weight = nn.Parameter(pe_embedding_table)
# print(pe_embadding)
# src_embedding = pe_embadding(src_seq)
#注意 这里传入的应该是位置参数 而不是src_seq


#制造位置标号
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)),0) for _ in tgt_len]).to(torch.int32)
# # 注意 这里的是位置编码 所以max(max_pisition_len) 而不应该是max(src_len) 或者 max(tgt_len)
# src_pos = torch.cat([torch.unsqueeze(torch.arange(max(max_position_len)),0) for _ in src_len]).to(torch.int32)
# print(src_pos)

src_pe_embedding = pe_embadding(src_pos) #传入位置参数
tgt_pe_embedding = pe_embadding(tgt_pos) #传入位置参数

# print(src_pe_embedding)
# print(tgt_pe_embedding)
#shenxiao 2022-04-04



