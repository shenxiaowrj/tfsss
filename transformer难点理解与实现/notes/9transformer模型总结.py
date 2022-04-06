# _*_coding:utf-8_*_
#transformer模型总结
'''
#特点：
一，没有先验假设
    卷积神经网络的先验假设是局部位置关联性， 通过局部的像素点 就能学好任务
        以此可以让它充分模型的局部信息 通过卷积层的加深获得全局信息
    循环神经网络的先验假设是时间序列上的关联性 上一时刻输出 才能进入下一时刻  自回归
        以此可以让它获得时序性很强的序列的时序信息
    transformer没有置入先验假设，
        所以它可以很简单的既可以学习到很强的空间上的信息
        也可以学习到很强的时许上的信息 并且相对于卷积神经网络来说 它可以做到并行
随之带来的缺点就是
二，数据量的要求和先验假设的程度成反比
    也就是说，transformer不是万能的 不能无脑的使用它
    transformer要想取得一个很好的效果 必须需要一个数量大的数据集 并且其下需要的数据的数量是远大于cnn rnn的
三，核心计算在于自注意力机制，平方复杂度
    tranformer在建模很长的时许信息的时候 其计算的复杂度 是成指数进行增加的
    而相比之下 rnn 在每一次时序建模的时候的空间复杂度 都是相同的

    所以后面出了很多论文来加入一些先验假设
        比如
            注意力机制算出的权重不是随机的 而应该是对角的
            对权重做一些哈希算法
            这些都是为了降低注意力机制的计算复杂度
'''

'''
#模型架构
    encoder
        position embadding            
        multi head self attention
        layer norm & residual
        feedforward neural network 仅有一个隐藏层的多层感知机
    
            首先来复习之前比较或的双卷积架构
                dips-wise卷积  也就是可分离通道的卷积
                加上
                1*1的卷积
                    dips-wise卷积做的是像素在空间上的混合
                    而1*1的卷积做的是在通道上的混合
                    二者的功能不同 但是二者结合可以达到大卷积的效果 并且可以降低计算量
            
            同样 这里的 多头自注意力卷积和feedforward层也是类似的架构
                multi head self attention
                    用来做每个单词对应在其序列位置上的相似度的混合
                feedforward nn
                    用来做每个单词的每个位置上的特征维度的卷积
            
            为什么要加入position embadding呢？
                因为transformer本身并没有先验假设 其不能做到对空间位置进行建模 所以我们需要加入position embadding
                以此来分辨 类似 我吃西红柿 西红柿吃我 这种词完全相同 但是位置不同的序列
            
    decoder
        position embadding
        casual multi head self attention
        layer norm & residual
        memory base multi head cross attention
        layer norm & residual 
        feedforward  n n
        layer norm & residual
    
            casual multi head self attention 也就是说 带有因果的多头自注意力机制
            
            memory base multi head cross attention 也就是将encoder 的输出作为key 和 value 将decoder的输入作为 query进行的自注意力运算
    
    '''

'''
#使用类型(之后的transformer族的一系列架构的分类)
encoder only 
    bert 分类任务（情感分类） 非流式任务（也就是 全部喂入数据之后再输出）
decoder only 
    gpt系列 语言建模 自回归生成任务 流式任务（也就是一边输入 一边进行输出）
encoder-decoder 
    机器翻译 语音识别

语音识别和机器翻译都是属于 流式任务的代表 它们的难点在于 其一边输入一边输出 输入的数据不完整 也要预测出 后面的话 以此来增强实时性
    反应速度
    增强预测能力

    '''
