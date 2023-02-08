import torch
from torch import Tensor


class FakeSummaryWriter:
    def add_scalars(self, *args, **kwargs):
        pass


def compute_gradient_cosine_similarity_fast(one_hot: Tensor, g: Tensor) -> Tensor:
    square_sum = torch.sum(one_hot ** 2)
    value_norm = torch.sqrt(1 - (2 * one_hot) + square_sum)
    g_norm = g.norm()
    norm = (g - torch.matmul(one_hot, g)) / (value_norm * g_norm)
    return norm


def compute_gradient_cosine_similarity_slow(one_hot: Tensor, g: Tensor) -> Tensor:
    scores = []
    for i in range(len(one_hot)):
        v = -one_hot
        v[i] = 1 - one_hot[i]
        score = torch.cosine_similarity(v, g, dim=0)
        scores.append(score)
    return torch.stack(scores)
