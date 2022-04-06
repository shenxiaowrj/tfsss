# _*_coding:utf-8_*_
#shenxiao
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

src_len = torch.Tensor([2,4]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)

batch_size = 2

#单词表的大小
max_num_src_words = 8
max_num_tgt_words = 8

#序列的最大的长度
max_src_seq_len = 5
max_tgt_seq_len = 5

#用单词索引构成的句子 并作了padding padding的值为0
# src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L) ),(0,max_num_src_words-L)),0) for L in src_len])
# tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L)) ,(0,max_num_src_words-L)),0)for L in tgt_len])
#把max_src_seq_len改为max(src_len)了
'''WHY？？？'''
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,) ),(0,max(src_len)-L)),0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L,)) ,(0,max(tgt_len)-L)),0)for L in tgt_len])
'''我去 少加L后面的,就会报错 好神奇 why???'''#shenxiao

#一个单词的特征维数
model_dim = 8

#构造embadding
src_embadding_table = nn.Embedding(max_num_src_words+1,model_dim)
tgt_embadding_table = nn.Embedding(max_num_src_words+1,model_dim)

#填入序列 实例化
src_embadding = src_embadding_table(src_seq)
tgt_embadding = tgt_embadding_table(tgt_seq)

#shenxiao

#最大的位置编码的长度
max_position_len = 5

#构建pe_embadding_table
#位置编码公式
pos_mat = torch.arange(max_position_len).reshape(-1,1)

i_mat = torch.pow(10000,torch.arange(0 , model_dim ,2).reshape(1,-1) / model_dim)

pe_embadding_table = torch.zeros(max_position_len,model_dim)

pe_embadding_table[:,0::2] = torch.sin(pos_mat / i_mat) #偶数列
pe_embadding_table[:,1::2] = torch.cos(pos_mat / i_mat) #奇数列

#构建pe_embadding
pe_embadding = nn.Embedding(max_position_len,model_dim)
pe_embadding.weight = nn.Parameter(pe_embadding_table)

#制造位置序列标号
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for _ in src_len])
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)),0) for _ in tgt_len])

#传入位置序列 实例化embadding
src_pe_embadding = pe_embadding(src_pos)
tgt_pe_embedding = pe_embadding(tgt_pos)

#shenxiao
##################################################################################################

'''构造encoder的self—attention mask'''#shenxiao
#根据公式
#query (batch_size,max(src_len),model_dim(embadding))
#key (batch_size,max(src_len),model_dim(embadding))
#所以 mask_ma (batch_size,max(src_len).max(src_len))

#构造有效的句子的位置编码矩阵 同时按句子中的最大长度 继续pad 0
# valid_encoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len])

# print(valid_encoder_pos)
# print(valid_encoder_pos.shape)
# tensor([[1., 1., 0., 0.],
#         [1., 1., 1., 1.]])
# torch.Size([2, 4])
# batch_size 乘以 max(src_len)

#用连接矩阵的形式 构造出关联性 也就是矩阵乘以矩阵的转置 所以给valid_encoder_pos在第二维增加一维
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len]),2)
# A = [torch.ones(L) for L in src_len]
# print(A)
# [tensor([1., 1.]), tensor([1., 1., 1., 1.])]
#构造连接矩阵
valid_encoder_pos_metrix = torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(1,2))
                                                                        #构建一个转置
# print(valid_encoder_pos_metrix)
# tensor([[[1., 1., 0., 0.],
#          [1., 1., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#
#         [[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]]])

#构建无效矩阵编码
invalid_encoder_self_matrix = 1-valid_encoder_pos_metrix

#转换成布尔型 即为mask_encoder_self_attention
mask_encoder_self_attention = invalid_encoder_self_matrix.to(torch.bool)
# print(mask_encoder_self_attention)
# tensor([[[False, False,  True,  True],
#          [False, False,  True,  True],
#          [ True,  True,  True,  True],
#          [ True,  True,  True,  True]],
#
#         [[False, False, False, False],
#          [False, False, False, False],
#          [False, False, False, False],
#          [False, False, False, False]]])
#false代表我们不需要对其进行mask操作 而 true则说明我们需要对其进行mask操作

#构建一个score score也就是query乘以key的值
score = torch.randn(batch_size,max(src_len),max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention,-1e9)
prob = F.softmax(masked_score,-1)

print(src_len)
print('原始得分','\n',score)
print('掩码之后的分数','\n',masked_score)
print('masked self-attention机制之后得到的概率 也就是权重','\n',prob)
'''
tensor([2, 4], dtype=torch.int32)
原始得分 
 tensor([[[ 0.2541, -0.7361, -0.8365,  1.2543],
         [ 0.6589,  0.5759, -0.5924, -1.2744],
         [-0.4680,  0.1420,  1.1173,  0.5491],
         [ 1.0489,  0.0275,  0.1438,  2.0903]],

        [[ 0.7284,  0.1582, -0.6804, -0.1074],
         [-0.2196,  0.2299,  0.1721, -0.5507],
         [-0.2754, -1.5439, -0.4684,  1.1258],
         [-0.1284, -1.1821,  1.1660, -0.5087]]])
掩码之后的分数 
 tensor([[[ 2.5414e-01, -7.3609e-01, -1.0000e+09, -1.0000e+09],
         [ 6.5895e-01,  5.7588e-01, -1.0000e+09, -1.0000e+09],
         [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
         [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09]],

        [[ 7.2840e-01,  1.5823e-01, -6.8037e-01, -1.0743e-01],
         [-2.1964e-01,  2.2988e-01,  1.7205e-01, -5.5072e-01],
         [-2.7536e-01, -1.5439e+00, -4.6838e-01,  1.1258e+00],
         [-1.2838e-01, -1.1821e+00,  1.1660e+00, -5.0868e-01]]])
masked self-attention机制之后得到的概率 也就是权重 
 tensor([[[0.7291, 0.2709, 0.0000, 0.0000],
         [0.5208, 0.4792, 0.0000, 0.0000],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.2500, 0.2500, 0.2500, 0.2500]],      #因为对其进行掩码操作 所以其概率最后是平均的
                                                 #之后还会对loss进行一次mask 所以这里虽然概率是有值的 但也不影响
        [[0.4458, 0.2520, 0.1090, 0.1932],
         [0.2099, 0.3290, 0.3105, 0.1507],
         [0.1622, 0.0456, 0.1337, 0.6585],
         [0.1760, 0.0614, 0.6423, 0.1203]]])
'''

#这就是encoder之中的self-attention mask
#shenxiao 2022-04-05
