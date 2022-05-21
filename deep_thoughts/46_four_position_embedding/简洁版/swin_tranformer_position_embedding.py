# _*_coding:utf-8_*_
# 3. 2d relative bias trainable embedding
import torch
import torch.nn as nn

def create_2d_relative_bias_trainable_embedding(n_head, height, width, dim):
    pass
    # width: 5, [0, 1, 2, 3, 4], bias=[-width+1, width-1], 2*width-1
    # height: 5, [0, 1, 2, 3, 4], bias=[-height+1, height-1], 2*height-1
    position_embedding = nn.Embedding(num_embeddings=(2*width-1)*(2*height-1), embedding_dim=n_head)
    nn.init.constant_(tensor=position_embedding.weight, val=0.)

    def get_relative_position_index(height, width):
        coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width))) # [2, height,width]
        coords_flatten = torch.flatten(coords, 1) # [2, height*width]

        relative_coords_bias = coords_flatten[:, :, None] - coords_flatten[:, None, :] # [2, height*width, height*width]

        relative_coords_bias[0, :, :] += height-1
        relative_coords_bias[1, :, :] += width-1

        # A: 2d, B:1d, B[i*cols+j] = A[i,j]
        relative_coords_bias[0, :, :] *= relative_coords_bias[1, :, :].max() +1

        return relative_coords_bias.sum(0) # [height*width, height*width]

    relative_position_bias = get_relative_position_index(height=height, width=width)
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width, height*width, n_head) # [height*width, height*width, n_head]

    bias_embedding = bias_embedding.permute(2, 0, 1).unsqueeze(0)

    return bias_embedding

