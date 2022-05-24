# shenxiao 2022-05-24


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

'''
convnext中的每一个大层是由不同数目（深度不同）的block构成的 block内的总体架构是相同的 只是特征维度等不同
所以我们先来构建一个block
block的组成部分：
#depth wiseth 7*7 96 
    深度可分离的群卷积 这个卷积对于每一个通道单独进行的 group=input_channel
#layer norm

#1*1 的卷积 （point wise卷积）
    其实就不是卷积 因为卷积的操作是对一块儿特征区域内的像素进行提取变为一个 (即一块儿区域变为一个)
    而这里区域为一 也就是（从一个变为一个） 所以就丧失了卷积操作所宣称的局部特征提取（空间上的局部关联性）
    其就可以看作是一个全连接层
    作用：
        常规卷积的作用是：空间融合 和 通道融合
        1*1卷积的作用是： 通道融合 （only）
            也就是相当于mlp 比如说将上一层特征维度384映射到下一层就是96

#gelu

#1*1 的卷积 （point wise卷积）

# 残差连接 前后两个张量特征维度相同
    
        
'''
# 不管定义多么大的模块儿 都需要继承自nn.Module类 这是一个主要的父类
# nn.Module是pytorch框架的重要的一部分

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # depth wise conv 深度可分离卷积
        # group = dim(input_dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv

        # layernorm
        # 输入张量的维度：batch_size * height * width * channel
        # 算均值和方差： 计算范围 batch_size * height * width
        # 均值和方差的大小: channel
        # 算出均值和方差之后做归一化 将值计算成正态分布
        # 引入仿射变换的参数 w(权重) b(偏置)
        # 将刚才转化成正态分布的参数 移到另外的一个分布
        self.norm = LayerNorm(dim, eps=1e-6)

        # point wise conv 这里用mlp实现的 而没用 1*1 卷积
        # 1*1 卷积可以这么写
            # self.pwconv3 = nn.Conv2d(in_channels=dim,out_channels=4*dim,kernel_size=(1,1))
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers

        self.act = nn.GELU()
            # self.pwconv4 = nn.Conv2d(4*dim, dim, kernel_size=(1,1))
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # depth wise
        x = self.dwconv(x)
        # 转置
        '''
        pytorch中两个转置函数
        transpose 一次性只能在两个维度之间进行转置
        permute 可以对很多维度进行转置
        '''
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # layernorm
        x = self.norm(x)
        # point wise conv 扩大四倍
        x = self.pwconv1(x)
        # 激活函数 gelu
        x = self.act(x)
        # point wise conv 减小四倍
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # 转置
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # 残差
        x = input + self.drop_path(x)
        return x

"""
stem
    lr
stage1
    lr
    downsampling
stage2
    lr
    downsampling
stage3
    lr
    sownsampling
stage4

maxpool

head
"""
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    """
    stem (和swin-transoformer划分块儿的方式一样)
    
    res2
        downsample1 (对图片来说下采样 对channel来说上采样)
    res3
        downsample2
    res4
        downsample3
    res5
    
    head
    
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],   # dims指的是每一个stage中输入通道的特征维数
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        """
        nn.Modulelist() 也就是Module的列表  一个大的Sequential就可以构成一个Moudle
        nn.Modulelist() 可以扩大存放好多个module
        """
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # stem层
            # 输入通道数为3 （rgb） 输出通道数为96
            # 卷积核为4*4 stride为4 pide默认为0
            # 的一个卷积层
        # 和vision_transforemr 划分patch的层一样 借用划分patch的操作想法 将image划分成patch
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        # 将stem也加入到下采样层中 所以下面的downsample就会有四个
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(

                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            # 在每个下采样之前都有一个layer norm层  下采样是通过2*2 stride=2卷积操作实现的
            # 卷积只需要指定输入通道数的大小 输出通道数的大小就行了 图片的尺寸是自动算出来的 当然你也可以根据公式来算一算
            """和swin-transformer中的patch_merging很相似 这个是啥操作来？ 我忘记了"""
            self.downsample_layers.append(downsample_layer)
        ### 顺序： stem dp1 dp2 dp3


        # 四个stage
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks

        # drop_out的比例 每一个stage之后的比例是不一样的
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0

        # 两层循环
        # 对4个stage进行循环
        """
        这两个双层嵌套for循环的意思我明白 但是具体的代码实现过程 我不明白
        我不懂它是如何可以根据遍历深度 来将block叠加的
        这是一个python中的语法糖 一种简写的方式吗？
        """
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            # 下一层循环，对深度 depth进行循环
            ###### 加一个星号 使得列表可以展开
            )
            self.stages.append(stage)
            cur += depths[i]   # 记录深度的 作用:对于drop_out层有作用 根据不同的深度 来实现不同的drop_out的比例 深度越深 drop_out的比例越大
                                                    # 符合：上层提取的是基础的信息，下层提取的是高层的语义信息，上层变动应该慢，下层应该尽快的变动


        # 用于做maxpool的层归一化
        '''
        这一点很奇怪 为什么层归一化还能用做最大池化
        还是说最大池化是用mean函数实现的 这里的作用只是一个layter norm
        '''
        '''他们说层归一化的好处是让参数更稳定'''
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        # 对于权重进行归一化
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
            # 常用的0归一化
    """可以学习一下他归一化的写法"""

# 因为模型要做不同的任务 可能只需要作为一个backbone。一个特征提取器
# 所以要把特征提取部分 和 完整的做分类的部分 写成两个函数

    def forward_features(self, x):
        # 每一个下采样作用在对应的每一个stage之前
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
            # 全局池化，将四维转化为二维

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

"""
layer norm出现的三种情况:
stem之后
每个downsampling之前
最后一层全连接(分类头)之后    
"""
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

# shenxiao 2022-05-24