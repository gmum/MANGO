import torch
from torch import Tensor


def initialize_logit_at_original_value(stochastic_mask: Tensor, quantized_mask: Tensor, original_mask: Tensor,
                                       value: float = 10.0):
    logits = torch.zeros_like(stochastic_mask).float()
    logits[original_mask] = value
    logits[~stochastic_mask] = -float('inf')
    logits[quantized_mask] = 1.0
    return logits


def initialize_logit_normal(stochastic_mask: Tensor, quantized_mask: Tensor):
    logits = torch.normal(0, 1, size=stochastic_mask.shape).to(stochastic_mask.device)
    logits[~stochastic_mask] = -float('inf')
    logits[quantized_mask] = 1.0
    return logits


def logits_to_one_hot(logits: Tensor, stochastic_mask: Tensor, quantized_mask: Tensor):
    one_hot_encodings = torch.softmax(logits, dim=1)
    one_hot_encodings = torch.masked_fill(one_hot_encodings, ~stochastic_mask, 0.0)
    one_hot_encodings = torch.masked_fill(one_hot_encodings, quantized_mask, 1.0)
    return one_hot_encodings


def initialize_one_hot_normal(stochastic_mask: Tensor, quantized_mask: Tensor):
    logits = initialize_logit_normal(stochastic_mask, quantized_mask)
    return logits_to_one_hot(logits, stochastic_mask, quantized_mask)
