# _*_coding:utf-8_*_
#首先假设一点常量：
import torch
import torch.nn as nn
import torch.nn.functional as F
batch_size = 1
input_channel = 3
image_h = 8                                                                             #shenxiao
image_w = 8
image = torch.randn(batch_size,input_channel,image_h,image_w)

model_dim = 8       #每一个位置抽取的特征的维度的数目

pacth_size = 4      #每一个小块儿的长度
patch_depth = pacth_size*pacth_size*input_channel    #根据论文中的要求定义的每个patch的深度 要和weight进行连接
             #这是patch的面积 也就相当于卷积核的面积 然后乘以 input_channel 也就是输入的通道数 这样就构成了 patch操作的深度

weight = torch.randn(patch_depth,model_dim)                      #shenxiao
# print(weight.shape)
# torch.Size([48, 8])
# 这个权重具体的意思是什么呢？
# weight 就是 patch2embadding 的乘法矩阵 也就是我们要将 patch_depth的大小 映射到 model_dim 的大小

#第一种 DNN perspective
def image2emb_naive(image,patch_size,weight):
    #image shape : batch_size * input_channels * h * w
    # patch = F.unfold(input=image,kernel_size=patch_size,stride=pacth_size)
    # print(patch.shape)
    # torch.Size([1, 48, 4])
    # 转置以下更好看
    patch = F.unfold(input=image,kernel_size=patch_size,stride=pacth_size).transpose(-1,-2)
    '''这一步就是 从image2patch的过程 '''
    print(patch.shape)
    # torch.Size([1, 4, 48])
    #这个数是怎么来的呢？                                             #shenxiao
        #4 : 8 * 8的图片 分成 4 * 4的块儿 一共能分成 4 块儿 这个4 也就是 patch的块儿数
        #48: 其实就是patch_depth ：patch_size * patch_size * input_channels
    patch_embadding = patch @ weight # @ 代表矩阵相乘
    '''这一步是patch2embadding的过程'''
    return patch_embadding

patch_embadding_naive = image2emb_naive(image,pacth_size,weight)

#第二种 CNN perspective
def image2emb_conv(image,kernel,stride):
    conv_output = F.conv2d(image,kernel,stride=stride)    #第一步：2d convolution over image
    batch_size,output_channels,output_height,output_wide = conv_output.shape                  #shenxiao
    patch_embadding = conv_output.reshape(batch_size,output_channels,(output_height * output_wide)).transpose(-1,-2)
    #第二步：flattern the output feature
    return patch_embadding
# weight = torch.randn(patch_depth,model_dim) # patch_depth 是卷积核的面积乘以输入通道数 model_dim 是输出的通道数
kernel = weight.transpose(0,1).reshape((-1,input_channel,pacth_size,pacth_size)) #kernal 也就是将weight拆分出来

patch_embadding_conv = image2emb_conv(image,kernel=kernel,stride=pacth_size)

print(patch_embadding_naive)
print(patch_embadding_conv)
# torch.Size([1, 4, 48])                          #shenxiao
# 这俩最后实现的效果是一样的
'''
tensor([[[ -4.4449,   3.9271,   3.2570,  -0.9595,   0.4202,   0.5007,  -7.2013,
            5.3649],
         [-19.6209,  -5.5153,   2.0163,   1.1719,  -5.1006,  -5.9877,   0.4322,
            3.6984],
         [  2.7960,  -4.2914,   2.8355,  -3.3591,   2.3218,  -2.7431,  -6.6145,
           -2.2431],
         [ -2.6745,   2.1395,   3.4503,  -0.9731, -15.3321,  -2.1057,   3.7707,
           -0.2576]]])
tensor([[[ -4.4449,   3.9271,   3.2570,  -0.9595,   0.4203,   0.5007,  -7.2013,
            5.3649],
         [-19.6209,  -5.5153,   2.0163,   1.1719,  -5.1006,  -5.9877,   0.4322,
            3.6984],
         [  2.7960,  -4.2914,   2.8355,  -3.3591,   2.3218,  -2.7431,  -6.6145,
           -2.2431],
         [ -2.6745,   2.1395,   3.4503,  -0.9731, -15.3321,  -2.1057,   3.7707,
           -0.2576]]])
'''
#2022-04-07 星期四    shenxiao