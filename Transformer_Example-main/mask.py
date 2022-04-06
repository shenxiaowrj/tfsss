import torch

from data import zidian_x, zidian_y


def mask_pad(data):
    # b句话,每句话50个词,这里是还没embed的

    # data = [b, 50]
    # 判断每个词是不是<PAD>
    mask = data == zidian_x['<PAD>']

    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)

    # 在计算注意力时,是计算50个词和50个词相互之间的注意力,所以是个50*50的矩阵
    # 是pad的列是true,意味着任何词对pad的注意力都是0
    # 但是pad本身对其他词的注意力并不是0
    # 所以是pad的行不是true

    # 复制n次
    # [b, 1, 1, 50] -> [b, 1, 50, 50]
    mask = mask.expand(-1, 1, 50, 50)

    return mask


def mask_tril(data):
    # b句话,每句话50个词,这里是还没embed的
    # data = [b, 50]

    # 50*50的矩阵表示每个词对其他词是否可见
    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
    # [1, 50, 50]
    """
    [[0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]"""
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))
    # print('tril','\n',tril)

    # 判断y当中每个词是不是pad,如果是pad则不可见
    # [b, 50]
    mask = data == zidian_y['<PAD>']

    # print('mask1','\n',mask)

    # 变形+转型,为了之后的计算
    # [b, 1, 50]
    mask = mask.unsqueeze(1).long()

    # print('mask2','\n',mask)
    # mask和tril求并集
    # [b, 1, 50] + [1, 50, 50] -> [b, 50, 50]
    mask = mask + tril

    # print('mask3','\n',mask)
    # 转布尔型
    mask = mask > 0
    #这是什么呀？ 转了一次

    # print('mask4','\n',mask)

    # 转布尔型,增加一个维度,便于后续的计算
    '''仅仅是增加了一个维度吗?'''
    mask = (mask == 1).unsqueeze(dim=1)
    #这是什么呀？  为什么还要再转一次？

    # print('mask5','\n',mask)
    return mask


'''
tril 
 tensor([[[0, 1, 1,  ..., 1, 1, 1],
         [0, 0, 1,  ..., 1, 1, 1],
         [0, 0, 0,  ..., 1, 1, 1],
         ...,
         [0, 0, 0,  ..., 0, 1, 1],
         [0, 0, 0,  ..., 0, 0, 1],
         [0, 0, 0,  ..., 0, 0, 0]]])
mask1 
 tensor([[False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False,  True,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True]])
mask2 
 tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
          1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
          1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1]]])
mask3 
 tensor([[[0, 1, 1,  ..., 2, 2, 2],
         [0, 0, 1,  ..., 2, 2, 2],
         [0, 0, 0,  ..., 2, 2, 2],
         ...,
         [0, 0, 0,  ..., 1, 2, 2],
         [0, 0, 0,  ..., 1, 1, 2],
         [0, 0, 0,  ..., 1, 1, 1]],

        [[0, 1, 1,  ..., 2, 2, 2],
         [0, 0, 1,  ..., 2, 2, 2],
         [0, 0, 0,  ..., 2, 2, 2],
         ...,
         [0, 0, 0,  ..., 1, 2, 2],
         [0, 0, 0,  ..., 1, 1, 2],
         [0, 0, 0,  ..., 1, 1, 1]],

        [[0, 1, 1,  ..., 2, 2, 2],
         [0, 0, 1,  ..., 2, 2, 2],
         [0, 0, 0,  ..., 2, 2, 2],
         ...,
         [0, 0, 0,  ..., 1, 2, 2],
         [0, 0, 0,  ..., 1, 1, 2],
         [0, 0, 0,  ..., 1, 1, 1]],

        ...,

        [[0, 1, 1,  ..., 2, 2, 2],
         [0, 0, 1,  ..., 2, 2, 2],
         [0, 0, 0,  ..., 2, 2, 2],
         ...,
         [0, 0, 0,  ..., 1, 2, 2],
         [0, 0, 0,  ..., 1, 1, 2],
         [0, 0, 0,  ..., 1, 1, 1]],

        [[0, 1, 1,  ..., 2, 2, 2],
         [0, 0, 1,  ..., 2, 2, 2],
         [0, 0, 0,  ..., 2, 2, 2],
         ...,
         [0, 0, 0,  ..., 1, 2, 2],
         [0, 0, 0,  ..., 1, 1, 2],
         [0, 0, 0,  ..., 1, 1, 1]],

        [[0, 1, 1,  ..., 2, 2, 2],
         [0, 0, 1,  ..., 2, 2, 2],
         [0, 0, 0,  ..., 2, 2, 2],
         ...,
         [0, 0, 0,  ..., 1, 2, 2],
         [0, 0, 0,  ..., 1, 1, 2],
         [0, 0, 0,  ..., 1, 1, 1]]])
mask4 
 tensor([[[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]],

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]],

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]],

        ...,

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]],

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]],

        [[False,  True,  True,  ...,  True,  True,  True],
         [False, False,  True,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]]])
mask5 
 tensor([[[[False,  True,  True,  ...,  True,  True,  True],
          [False, False,  True,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True]]],


        [[[False,  True,  True,  ...,  True,  True,  True],
          [False, False,  True,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True]]],


        [[[False,  True,  True,  ...,  True,  True,  True],
          [False, False,  True,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True]]],


        ...,


        [[[False,  True,  True,  ...,  True,  True,  True],
          [False, False,  True,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True]]],


        [[[False,  True,  True,  ...,  True,  True,  True],
          [False, False,  True,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True]]],


        [[[False,  True,  True,  ...,  True,  True,  True],
          [False, False,  True,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          ...,
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True],
          [False, False, False,  ...,  True,  True,  True]]]])
'''