# _*_coding:utf-8_*_
#首先先回顾一下第二种卷积的方式来进行image2embadding的操作
import torch
import torch.nn as nn
import torch.nn.functional as F                   #shenxiao

#首先生成一个图片
batch_size = 1
input_channel = 3
image_h = 8
image_w = 8
image = torch.randn(batch_size,input_channel,image_h,image_w)

#patch 的常量
patch_size = 4

#每一个位置的特征的数量 #类似于mlp中隐藏层中神经元的数量 在数学之中是什么？ 更多的项数 x?                  #shenxiao
model_dim = 8
#patch的深度
patch_depth = patch_size*patch_size*input_channel
#矩阵相乘变换矩阵 权重
weight = torch.randn(patch_depth,model_dim)    #
#卷积中的核 其实也就是将weight拆开
kernel = weight.transpose(0,1).reshape((-1,input_channel,patch_size,patch_size))
# print(kernel.shape)
# torch.Size([8, 3, 4, 4])
#为什么wight 和 kernel 不能写在一起 更准确的说 kernel是对weight的拆分 为什么不能直接就不去拆分呢？
#不对，不对。weight还有一个生成权重矩阵的功能 patch_depth 只是一个数 代表patch之后的 之后的什么呢？
                #patch核的面积乘以输入通道的数目 为什么是这样的呢？ 它有什么直观的含义呢？

#需要特别注意的是 这里kernel（conv）和kernel_size（naive） 是不同的
    #kernel 是一个矩阵
    #kernel_size 是一个数 就是指定卷积核的大小 比如这里就是 4 * 4              #shenxiao
def image2emb_conv(image,kernel,stride):
    conv_output = F.conv2d(image,kernel,stride=stride)   #在
    batch_size,output_channels,output_height,output_wide = conv_output.shape
    patch_embadding = conv_output.reshape(batch_size,output_channels,(output_height*output_wide)).transpose(-1,-2) #这是一个展平的操作
    return patch_embadding

patch_embadding_conv = image2emb_conv(image,kernel=kernel,stride=patch_size)
# print(patch_embadding_conv.shape)
# torch.Size([1, 4, 8])
# batch_size patch_size model_dim

#step 2 prepend CLS token embedding
#模仿bert的写法来创建的
cls_token_embadding = torch.randn(batch_size,1,model_dim,requires_grad=True)      #可训练的类别标号序列
                #这里的1 代表的是token_embadding 的序列的数目
# print(cls_token_embadding.shape)
# torch.Size([1, 1, 8])
'''
1 4 8 concat 1 1 8 = 1 5 8
'''                   #shenxiao

token_embadding = torch.cat([cls_token_embadding,patch_embadding_conv],dim=1)    #将类别序列和embadding序列拼起来 在维度1上进行拼
# print(token_embadding.shape)
# torch.Size([1, 5, 8])

#step3 add position embedding
max_num_token = 16     #vocab_size
position_embadding_table = torch.randn(max_num_token,model_dim,requires_grad=True)    #作者在文章中实验好几种position embedding 最后选择了这个可学习的
# print(position_embadding_table.shape)
# torch.Size([16, 8])

#现在要取出table中的值
seq_len = token_embadding.shape[1]    # 5
'''cv领域之中一般都把图片改成相同的大小 所以一般不考虑 mask'''
# A = position_embadding_table[:seq_len] # 取出前seq_len列表中的位置编码（向量）
# print(A.shape)
# torch.Size([5, 8])

position_embadding = torch.tile(position_embadding_table[:seq_len],[token_embadding.shape[0],1,1])
'''torch.tile API 复制自身就为1 复制几次就为几 复制是对自身进行操作的'''   #shenxiao
# print(position_embadding.shape)
# torch.Size([1, 5, 8])
#现在是二维  现在复制成三维
token_embadding = token_embadding + position_embadding
#1 5 8 + 1 5 8 = 1 5 8

#step4 pass embaedding to Transformer  直接调用pytorch实现的transformer的encoder层
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=6)
encoder_output = transformer_encoder(token_embadding)
# print(encoder_output.shape)
# torch.Size([1, 5, 8])
#step5 do classification
num_classes = 10
label = torch.randint(num_classes,(batch_size,))    #又出现这个问题了 为什么一定要加个逗号？
cls_token_output = encoder_output[:,0,:]     #batch_size  (patch_size+cls_position)的进行抽取特征之后的版本 model_dim
# 切片的使用 这个目的是取出中间的那一个维度的数值

linear_layer = nn.Linear(in_features=model_dim,out_features=num_classes)
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits,label)
print(loss)
#2022-04-07 上半部分
#2022-04-08 下半部分  星期四      shenxiao