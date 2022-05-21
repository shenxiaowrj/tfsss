# _*_coding:utf-8_*_
# 2022-05-18

# 1.   1d absolute sin_cos constant embedding
import torch


def create_1d_absolute_sin_cos_embeddings(n_pos_vec, dim):
    # n_pos_vec: torch.arange(n_pos)

    assert dim % 2 == 0, "wrong dimension"

    position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float32)

    omega = torch.arange(dim//2, dtype=torch.float32)
    omega /= dim / 2.
    omega = 1. / (10000 ** omega)

    out = n_pos_vec[:, None] @ omega[None, :] # @ 表示：矩阵乘法

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # 填充
    position_embedding[:, 0::2] = emb_sin
    position_embedding[:, 1::2] = emb_cos

    return position_embedding



if __name__ == '__main__':
    n_pos = 4
    dim = 4
    n_pos_vec = torch.arange(n_pos, dtype=torch.float32)
    pe = create_1d_absolute_sin_cos_embeddings(n_pos_vec, dim=dim)
    print(pe)
