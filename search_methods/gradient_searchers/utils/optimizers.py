from abc import ABC, abstractmethod

import torch
from torch import Tensor


def scale_to_proper_one_hot_distribution(grad: Tensor, stochastic_mask: Tensor) -> Tensor:
    return grad - (torch.sum(grad, dim=1, keepdim=True) / torch.sum(stochastic_mask, dim=1, keepdim=True))


class InputOptimizerBase(ABC):
    def reset(self):
        pass

    @abstractmethod
    def updated(self, value: Tensor, grad: Tensor, stochastic_mask: Tensor, quantized_mask: Tensor) -> Tensor:
        ...


class SGDInputOptimizer(InputOptimizerBase):
    def __init__(self, lr: float = 1.0, scale: bool = False):
        self.lr = lr
        self.scale = scale

    def updated(self, value: Tensor, grad: Tensor, stochastic_mask: Tensor, quantized_mask: Tensor) -> Tensor:
        if self.scale:
            grad = scale_to_proper_one_hot_distribution(grad, stochastic_mask)
        grad[~stochastic_mask] = 0.0
        return value + self.lr * grad


class AdamInputOptimizer(InputOptimizerBase):
    def __init__(self, lr: float, scale: bool = False, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 ams_grad: bool = False):
        self.lr = lr
        self.scale = scale
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.v_hat = 0.0
        self.ams_grad = ams_grad

    def reset(self):
        self.m = 0.0
        self.v = 0.0
        self.v_hat = 0.0
        self.t = 0

    def updated(self, value: Tensor, grad: Tensor, stochastic_mask: Tensor, quantized_mask: Tensor) -> Tensor:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        _m = self.m / (1 - self.beta1 ** self.t)
        _v = self.v / (1 - self.beta2 ** self.t)

        if self.ams_grad:
            if isinstance(self.v_hat, float):
                self.v_hat = torch.zeros_like(_v)
            self.v_hat = torch.max(_v, self.v_hat)
            grad = _m / (torch.sqrt(self.v_hat) + self.epsilon)
        else:
            grad = _m / (torch.sqrt(_v) + self.epsilon)

        if self.scale:
            grad = scale_to_proper_one_hot_distribution(grad, stochastic_mask)
        grad[~stochastic_mask] = 0.0
        return value + self.lr * grad
