# _*_coding:utf-8_*_
#以构建关系矩阵的方式 实现构建mask
import torch
a = torch.randint(10,size=(4, 1))
print(a)
b = a - a.T
print(b)
c = (a - a.T) == 0
print(c)
# tensor([[1],
#         [3],
#         [9],
#         [3]])
# tensor([[ 0, -2, -8, -2],
#         [ 2,  0, -6,  0],
#         [ 8,  6,  0,  6],
#         [ 2,  0, -6,  0]])
# tensor([[ True, False, False, False],    1 只对 1 有关
#         [False,  True, False,  True],    3 对两个 3 有关 所以有两个地方都是true
#         [False, False,  True, False],    9 只对 9 相关
#         [False,  True, False,  True]])   3 对两个 3 相关 所以有两个位置都是true

# 这就是同类关系矩阵 对embedding进行编码之后 直接进行同类关系矩阵操作 矩阵中为0的值 表示两个值相关 若是不为0 则表示不相关
# 对于这些同类关系矩阵中不相关的值 要进行mask操作 也就是将之变为一个比较大的负数 也就是将他的能量变为负无穷
