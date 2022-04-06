# _*_coding:utf-8_*_
#decoder self-attention mask
#也就是对casual multi head self attention做的mask
#其是具备因果性的
#为什么要对其做掩码呢？ 因为预测的时候只能让它看到这一步的输入 不能看到序列之中之后的部分 也就是说不能让其看到其要预测的值

import torch
import torch.nn.functional as F

tgt_len = torch.Tensor([4,3]).to(torch.int32)

#思路，
    #首先,构建真实的tgt的位置编码的下三角矩阵
    #之后，构建无效的位置编码矩阵
    #再之后，构建mask矩阵

#torch API 中的上下三角矩阵
shang = [torch.triu(torch.ones(L,L)) for L in tgt_len]
xia = [torch.tril(torch.ones(L,L)) for L in tgt_len]
# print(shang)
# print()
# print(xia)
# [tensor([[1., 1., 1., 1.],       #上三角矩阵
#         [0., 1., 1., 1.],
#         [0., 0., 1., 1.],
#         [0., 0., 0., 1.]]), tensor([[1., 1., 1.],
#         [0., 1., 1.],
#         [0., 0., 1.]])]
#
# [tensor([[1., 0., 0., 0.],      #下三角矩阵
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]]), tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])]

A = [torch.ones(L)   for L in tgt_len]
# print(A)
# [tensor([1., 1., 1., 1.]), tensor([1., 1., 1.])]
B = [torch.ones(L,L) for L in tgt_len]
# print(B)
# [tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]]), tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])]

#所以 tri 的作用是 将原本有值的地方变为0 以此构成三角矩阵  这也是为什么 ones之中要加两个 L 的原因

#构建解码器真实的下三角的位置矩阵
C = [F.pad(torch.ones(L,L),(0,max(tgt_len)-L,0,max(tgt_len)-L)) for L in tgt_len]
# print(C)
# [tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]]), tensor([[1., 1., 1., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 0.],
#         [0., 0., 0., 0.]])]
D = [torch.unsqueeze(F.pad(torch.ones(L,L),(0,max(tgt_len)-L,0,max(tgt_len)-L)),0)for L in tgt_len]
# print(D)
# [tensor([[[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]]]), tensor([[[1., 1., 1., 0.],
#          [1., 1., 1., 0.],
#          [1., 1., 1., 0.],
#          [0., 0., 0., 0.]]])] #Add a Dimension
E = torch.cat([torch.unsqueeze(F.pad(torch.ones(L,L),(0,max(tgt_len)-L,0,max(tgt_len)-L)),0) for L in tgt_len])
# print(E)
# tensor([[[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]],
#
#         [[1., 1., 1., 0.],
#          [1., 1., 1., 0.],
#          [1., 1., 1., 0.],
#          [0., 0., 0., 0.]]])
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.ones(L,L),(0,max(tgt_len)-L, 0 ,max(tgt_len)-L)),0) for L in tgt_len])
# print(valid_decoder_tri_matrix)
# tensor([[[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]],
#
#         [[1., 1., 1., 0.],
#          [1., 1., 1., 0.],
#          [1., 1., 1., 0.],
#          [0., 0., 0., 0.]]])
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
# print(invalid_decoder_tri_matrix)
# tensor([[[False, False, False, False],
#          [False, False, False, False],
#          [False, False, False, False],
#          [False, False, False, False]],
#
#         [[False, False, False,  True],
#          [False, False, False,  True],
#          [False, False, False,  True],
#          [ True,  True,  True,  True]]])

'''到此为止 decoder self-attention就构建成功了'''
#我们尝试来随机创建一个score 之后来看一下它的效果
batch_size = 2
score = torch.randn(batch_size,max(tgt_len),max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix,-1e9)
prob = F.softmax(masked_score,-1)
# print(score)
# print(masked_score)
# print(prob)
# tensor([[[-1.2019,  0.0592, -0.0414,  0.5191],
#          [ 0.2317,  0.2858, -0.0952, -1.4039],
#          [ 1.0549,  0.1233,  1.2312,  0.7038],
#          [ 0.3727, -0.4999, -1.3865, -0.2444]],
#
#         [[-1.2272,  0.5128,  1.0317,  0.1729],
#          [-0.5762,  1.3481,  1.1285,  0.1868],
#          [ 1.1724,  0.3744,  1.0796,  0.0741],
#          [ 0.0547, -1.7879,  0.0249,  1.2367]]])
# tensor([[[-1.2019e+00,  5.9229e-02, -4.1446e-02,  5.1907e-01],
#          [ 2.3170e-01,  2.8579e-01, -9.5224e-02, -1.4039e+00],
#          [ 1.0549e+00,  1.2331e-01,  1.2312e+00,  7.0381e-01],
#          [ 3.7273e-01, -4.9993e-01, -1.3865e+00, -2.4442e-01]],
#
#         [[-1.2272e+00,  5.1281e-01,  1.0317e+00, -1.0000e+09],
#          [-5.7625e-01,  1.3481e+00,  1.1285e+00, -1.0000e+09],
#          [ 1.1724e+00,  3.7439e-01,  1.0796e+00, -1.0000e+09],
#          [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09]]])
# tensor([[[0.0751, 0.2652, 0.2398, 0.4200],
#          [0.3365, 0.3552, 0.2427, 0.0656],
#          [0.3039, 0.1197, 0.3625, 0.2139],
#          [0.4696, 0.1962, 0.0809, 0.2533]],
#
#         [[0.0615, 0.3502, 0.5884, 0.0000],
#          [0.0749, 0.5131, 0.4120, 0.0000],
#          [0.4234, 0.1906, 0.3859, 0.0000],
#          [0.2500, 0.2500, 0.2500, 0.2500]]])   #此向量全部无效 概率平均 但是之后loss还有一个mask 所以虽然有概率 但是不影响结果


