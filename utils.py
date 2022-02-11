"""
utils.py
"""

import math
import torch
import torch.nn.functional as F


def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = torch.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)


def augment_xy_data_by_8_fold(xy_data, training=False):
    # xy_data.shape = [B, N, 2]
    # x,y shape = [B, N, 1]

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)

    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    # data_augmented.shape = [B, N, 16]
    if training:
        data_augmented = torch.cat(
            (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=2
        )
        return data_augmented

    # data_augmented.shape = [8*B, N, 2]
    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return data_augmented


def data_augment(batch):
    batch = augment_xy_data_by_8_fold(batch, training=True)
    theta = []
    for i in range(8):
        theta.append(
            torch.atan(batch[:, :, i * 2 + 1] / batch[:, :, i * 2]).unsqueeze(-1)
        )
    theta.append(batch)
    batch = torch.cat(theta, dim=2)
    return batch
