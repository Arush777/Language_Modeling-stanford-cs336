from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from math import sqrt

import math

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        ):
            if lr < 0.0:
                raise ValueError(f"Invalid lr: {lr}")
            if eps < 0.0:
                raise ValueError(f"Invalid eps: {eps}")
            b1, b2 = betas
            if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
                raise ValueError(f"Invalid betas: {betas}")
            if weight_decay < 0.0:
                raise ValueError(f"Invalid weight_decay: {weight_decay}")

            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)
        @torch.no_grad()
        def step(self,closure=None):
            loss=None
            if closure is not None:
                with torch.enable_grad():
                    loss=closure()
            for group in self.param_groups:
                lr=group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    t = state["step"]

                    # decoupled weight decay (PyTorch-style)
                    if wd != 0.0:
                        p.data.add_(p.data, alpha=-lr * wd)
                        
                    # moment updates
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - (beta1 ** t)
                    bias_correction2 = 1.0 - (beta2 ** t)

                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

            return loss

    return AdamW

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    decay_ratio=(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
    
    #hyperparamter
    coeff=0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    #torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_l2_norm, norm_type=2)
    
    #from scratch
    params=list(parameters)
    
    grads=[p.grad for p in params if p.grad is not None]
    
    if len(grads)==0:
        return 
    
    total_sq = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for g in grads:
        total_sq+=g.pow(2).sum()
    total_norm = torch.sqrt(total_sq)
    if total_norm >= max_l2_norm:
        scale=max_l2_norm/(total_norm + 1e-12)
        for p in params :
            if p.grad is not None:
                p.grad.mul_(scale)
    else :
        return