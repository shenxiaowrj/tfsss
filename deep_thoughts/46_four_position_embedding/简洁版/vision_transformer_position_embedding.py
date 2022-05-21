# _*_coding:utf-8_*_
# 2. 1d absolute trainable embedding
import torch
import torchvision
import torch.nn as nn

def create_1d_absolute_trainable_embeddings(n_pos_vec, dim):
    #n_pos_vec: torch.arange(n_pos, dtype=torch.float)

    position_embedding = nn.Embedding(num_embeddings=n_pos_vec.numel(),embedding_dim=dim)
    nn.init.constant_(position_embedding.weight, 0.)


    return position_embedding

