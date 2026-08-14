"""Gradient Reversal Layer：forward 恒等，backward 乘 -lambda。"""
from __future__ import annotations

import torch
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_domain: float) -> torch.Tensor:
        ctx.lambda_domain = float(lambda_domain)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_domain * grad_output, None


def grad_reverse(input_tensor: torch.Tensor, lambda_domain: float) -> torch.Tensor:
    return _GradientReversalFunction.apply(input_tensor, float(lambda_domain))


class GradientReversal(torch.nn.Module):
    """可配置 lambda 的 GRL 模块。"""

    def __init__(self, lambda_domain: float = 1.0):
        super().__init__()
        self.lambda_domain = float(lambda_domain)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return grad_reverse(input_tensor, self.lambda_domain)

    def set_lambda(self, lambda_domain: float) -> None:
        self.lambda_domain = float(lambda_domain)
