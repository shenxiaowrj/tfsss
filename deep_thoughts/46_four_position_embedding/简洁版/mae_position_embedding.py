# _*_coding:utf-8_*_
import torch

def create_1d_absolute_sin_cos_embeddings(n_pos_vec, dim):
    # n_pos_vec: torch.arange(n_pos)

    assert dim % 2 == 0, "wrong dimension"

    position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float32)

    omega = torch.arange(dim//2, dtype=torch.float32)
    omega /= dim / 2.
    omega = 1. / (10000 ** omega)

    out = n_pos_vec[:, None] @ omega[None, :] # @ 表示：矩阵乘法 (而不是点积 product)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # 填充
    position_embedding[:, 0::2] = emb_sin
    position_embedding[:, 1::2] = emb_cos

    return position_embedding



def create_2d_absolute_sin_cos_embeddings(height, width, dim):
    assert dim & 4 == 0,"wrong dimension"

    position_embedding = torch.zeros(height*width, dim)

    coords = torch.stack(torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32))) # [2, height, width]

    height_embedding = create_1d_absolute_sin_cos_embeddings(torch.flatten(coords[0]), dim // 2) # [height*width, dim//2]
    width_embedding = create_1d_absolute_sin_cos_embeddings(torch.flatten(coords[1]), dim // 2) # [height*width, dim//2]

    position_embedding[:, :dim//2] = height_embedding
    position_embedding[:, dim//2:] = width_embedding

    return position_embedding



