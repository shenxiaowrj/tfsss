# _*_coding:utf-8_*_
#下面来具体实现vision_transformer
#网络架构
#DNN PERSESPECTIVE                                                         #shenxiao
    #image2path
    #path2embadding
#CNN perspective
    #2d convolution over image                                         #shenxiao
    #flatten the output feature map
'''
以上两种操作是并列的 两种操作的结果最后应该是一样的
但是实现的思路是不同的
第一个是原文中的 对于图片进行分块儿 每16*16个pixel组合成 一个patch 对于一张图片进行这样的patch分割
    操作实现的API 是torch中的unfold 也就是可以以类似卷积的方式的每一个窗口提取出来 这样做kernal_size=stride的卷积
    也就是每个patch之间不重叠的卷积的时候 就可以很自然 把从卷积之后提取出来的东西 每一个patch '
    现在每一个patch就相当于一个单词

第二个是做一个二维的卷积 卷积的维度分别是：     之后再将 高宽两个维度合并 并且继续进行展平     #shenxiao
'''
#CLASS token embadding
    #和bert的架构类似
    #但是很奇怪 一般来说 是将之看作是query 但是这个东西又加入了 position embadding 同时还能算注意力分数
    #不好理解 有一点问题在这里
#POSITION embadding

#TRANSFORMER encoder
    #和之前transformer之中的encoder一样
    #multi-head self attention
    #layer norm & residual
    #feed-forward network
    #layer norm & residual
#classification head
    #对于encoder抽取出来的特征 或者是 对于每个图片patch的序号的列表 进行分类

#疑问： question
    #这里的encoder怎么抽取的特征啊？ mlp？？？

#2022-04-07 星期四 shenxiao