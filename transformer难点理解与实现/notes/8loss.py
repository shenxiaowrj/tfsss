# _*_coding:utf-8_*_
#loss
import torch
import torch.nn as nn
import torch.nn.functional as F

#transformer 的loss函数
#若以机器翻译任务来举例
#本质上这是一个分类任务 decoder 抽取特征之后预测概率 然后和目标的标签算一个交叉熵loss cross entropy

#假设我们通过transformer模型预测出来的概率
#batch_size = 2 sequence_length = 3 vocab_size = 4 (单词表的数目 也就是字典之中的单词)\
batch_size = 2
sequence_length = 3
vocab_size = 4

logits = torch.randn(batch_size,sequence_length,vocab_size)
'''
# print(logits)
# tensor([[[ 1.1726, -0.0477, -0.7850,  1.7727],
#          [-2.5861,  0.0189, -1.5228,  0.1151],
#          [-0.1904, -0.1449, -0.0993,  1.5766]],
# 
#         [[-0.6028, -0.9924, -0.3561, -1.4772],
#          [ 0.1675, -0.0280,  0.7265, -0.6946],
#          [-0.3928,  0.8626, -1.3397, -0.7480]]])
'''
#随机生成一个label 一般来说 label是某一个单词 但是若是是一个概率的话 那也是可以支持的
#最小值 0 最大值 vocab_size shape (batch_size*sequence_length)

#第一种情况 对于序列上的每个位置都有一个label  这个label是单词表中索引
label = torch.randint(low=0,high=vocab_size,size=(batch_size,sequence_length))
'''
# print(label)
# tensor([[0, 2, 3],
#         [2, 0, 3]])
'''
#pytorch 的形状编写 默认的要求： 第一个是batch_size 第二个是 类别class 也就是对应这里的vocab_size
#所以要对logits进行转置
logits = logits.transpose(1,2)   #transpose 函数对于里边的数的顺序是没有要求的

#调用cross_entropy 计算loss targets的既可以是整型的类别 也可以是概率 都是支持的
loss = F.cross_entropy(input=logits,target=label)
'''
# print(loss)
# tensor(0.9675)
'''
#这里loss是个标量 我们知道反向传播的时候 一开始需要一个标量去开头 （自动求导时）
#这个loss是如何计算得到的呢? 我们知道 batch_size * sequence_length 也就是单词的总数 每一个单词都会有一个 loss 这里默认 将这六个loss相加之后 取平均值
#还可以做加法 和 全部输出
#全部输出
loss1 = F.cross_entropy(input=logits,target=label,reduction='none')
'''
print(loss1)
tensor([[2.5045, 2.2801, 0.4022],
        [2.2039, 1.1700, 2.3367]])
'''
#相加
loss2 = F.cross_entropy(input=logits,target=label,reduction='sum')
'''
print(loss2)
tensor(14.0750)
'''

#现在我们指定的 真实的序列长度是 3 3
#若是真实的序列长度是 2 3 呢？ 我们就需要写mask 或者 使用ignore_index

#手写mask的方式
tgt_len = torch.Tensor([2,3]).to(torch.int32)
'''
a1 = [torch.ones(L) for L in tgt_len]
# print(a1)
# [tensor([1., 1.]), tensor([1., 1., 1.])]
a2 = [F.pad(torch.ones(L),(0,max(tgt_len)-L)) for L in tgt_len]
# print(a2)
# [tensor([1., 1., 0.]), tensor([1., 1., 1.])]
a3 = [torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len)-L)),0) for L in tgt_len]
# print(a3)
# [tensor([[1., 1., 0.]]), tensor([[1., 1., 1.]])]
'''
mask = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len)-L)),0) for L in tgt_len])
# print(mask)
# tensor([[1., 1., 0.],
#         [1., 1., 1.]])
# print(mask.shape)
# torch.Size([2, 3])

loss_mask = F.cross_entropy(input=logits,target=label,reduction='none') * mask
# print(loss_mask)
# tensor([[1.2838, 1.2838, 0.0000],
#         [1.2838, 1.2838, 1.2838]])

#调用ignore_index
label[0,2] = -100
loss_mask1 = F.cross_entropy(input=logits,target=label,reduction='none',ignore_index=-100)
#ignore_index 默认等于 -100
# print(loss_mask1)
# tensor([[0.9688, 1.0662, 0.0000],
#         [2.2988, 0.5738, 0.8384]])

#效果相同
#这只是改了一个地方啊 如何全部序列 一起改呢？ 很简单 直接 pad 成 -100 就好了！
mask1 = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len)-L),value=-100),0) for L in tgt_len])
#pad 值默认为 0
loss_mask2 = F.cross_entropy(input=logits,target=label,reduction='none') * mask1
# print(loss_mask2)
# tensor([[0.8097, 1.2988, -0.0000],
#         [1.3087, 1.4318, 0.3488]])

#效果几乎相同 只是多了一个负号而已 因为是-100 不是100 应该不影响什么东西

#shenxiao 2022-04-07
