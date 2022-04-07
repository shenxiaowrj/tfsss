# _*_coding:utf-8_*_
#构建scaled self-attention
#构建一个scaled 的乘积的 注意力机制
import torch
import torch.nn.functional as F
model_dim = 8
def scaled_dot_attention(Q,K,V,attn_mask):
    score = torch.bmm(Q,K.transpose(-2,-1)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask,-1e9)
    prob = F.softmax(masked_score,-1)
    context = torch.bmm(prob,V)
    return context
#   Q K V 的shape (batch_size*num_head,seq_len,model_dim/num_head)
    #当然 在 intra self-attention Q 就是 tgt_len 而不是 seq_len


#再来回顾一下 transformers 的源码
'''
Transformer 类 : source_transformer.py
__init__:要去实例化一个tansformer  需要传入 
        d_model
        nhead
        num_encoder_layers
        num_decoder_layers
        dim_feedforward
        dropout
forward:要去调用transformer的话 则需要传入
        src: 源序列的词向量
        tgt: 目标序列的词向量
        src_msk: 源序列的掩码 也即是encoder self-attention mask
        tgt_msk: 目标序列的掩码 也即是decoder self-attenton mask
        memory_msk:带桥梁的 可连接的 intra attention mask
之后就可以得到解码器的输出
    这个输出没有进入softmax
    所以 要去加一个softmax 再加一个 全连接 进行一个概率的预测
    
'''
'''
具体实现的地方：
activation.py中：
MultiheadAttention 类
最终调用的是上一层的 source_functional.py
multi_head_attention_forward 函数
scaled_dot_attention
这个函数 做的是 计算scaled_dot_attention
还有加性的attention

以及三种mask的计算
'''