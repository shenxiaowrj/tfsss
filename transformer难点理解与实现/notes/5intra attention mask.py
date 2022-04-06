# _*_coding:utf-8_*_
# from 4encoder self-attention mask import *
import torch
import torch.nn.functional as F

#有空格的引用会出错

#也就是memory base multi head cross attention的mask操作
#为什么要对其进行mask呢？
#原序列 和 目标序列的长度 可能是不一样的 但是我们是对其pad成 相同的维度 所以pad的部分 我们要做一个mask

#构造intra-attrntion mask
# 公式 ： Q * K^T
# shape : [batch_size , tgt_seq_len ,src_seq_len]
###思路
    # 首先，我们要构造 真实的源序列和目标序列的位置矩阵
    # 之后， 我们来将两个矩阵相乘 构造连接矩阵 来判断对应A序列中每一个词 对于B序列中所有词的位置的有效性分析
    # 再之后，我们构建无效的连接矩阵  将无效的连接矩阵转换成 bool型 即为构造的intra-attention mask

src_len = torch.Tensor([2,4]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)

#真实的双序列的位置矩阵
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L) ,(0,max(src_len)-L)),0)for L in src_len]),2)
valid_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len)-L)),0) for L in tgt_len]),2)

#构造连接矩阵
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos,valid_encoder_pos.transpose(1,2))

#构造无效连接矩阵
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix

#构造mask
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)

# print('掩码注意力','\n',mask_cross_attention)
# 掩码注意力
#  tensor([[[False, False,  True,  True],
#          [False, False,  True,  True],
#          [False, False,  True,  True],
#          [False, False,  True,  True]],
#
#         [[False, False, False, False],
#          [False, False, False, False],
#          [False, False, False, False],
#          [ True,  True,  True,  True]]])

# print(valid_decoder_pos)
# print(valid_encoder_pos)
# print('有效的连接矩阵','\n',valid_cross_pos_matrix)

# tensor([[[1.],
#          [1.],
#          [1.],
#          [1.]],
#
#         [[1.],
#          [1.],
#          [1.],
#          [0.]]])
# tensor([[[1.],
#          [1.],
#          [0.],
#          [0.]],
#
#         [[1.],
#          [1.],
#          [1.],
#          [1.]]])

# 有效的连接矩阵
#  tensor([[[1., 1., 0., 0.],  #源序列的第一个值 和 目标序列的第一列 的有效位置判定
#          [1., 1., 0., 0.],   #源序列的第二值 和 目标序列的第一列的有效位置的判定
#          [1., 1., 0., 0.],
#          [1., 1., 0., 0.]],
#
#         [[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [0., 0., 0., 0.]]])#源序列的第二个序列的 最后一个单词 和目标序列的第二个序列 的有效位置判定


