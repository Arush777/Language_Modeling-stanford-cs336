
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
from .nn import softmax
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scores=Q @ K.transpose(-2,-1)
    d_k=Q.shape[-1]
    normalized_scores = scores / sqrt(d_k)
    
    if mask is not None:
        normalized_scores=normalized_scores.masked_fill(~mask,-1e9)
        
    weights=softmax(normalized_scores , dim=-1)
    
    return weights @ V

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    
    x = in_query_or_key                      # (..., T, d_k)
    
    # 1) split into pairs (even dims, odd dims)
    x_even = x[..., 0::2]                    # (..., T, d_k/2)
    x_odd  = x[..., 1::2]                    # (..., T, d_k/2)

    # 2) make the frequency vector for each pair index
    # freq[j] = theta^(-2j/d_k)
    j = torch.arange(d_k // 2, device=x.device, dtype=x.dtype)   # (d_k/2,)
    theta_j = theta ** (-2.0 * j / d_k)                             # (d_k/2,)

    # 3) angle for token position p and pair index j is p * freq[j]
    pos = token_positions.to(dtype=x.dtype)[..., None]           # (..., T, 1)
    angles = pos * theta_j                                          # (..., T, d_k/2)

    # 4) rotate each pair using cos/sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    while cos.ndim < x_even.ndim:  #(B, T, d/2) into (B, 1, T, d/2) so it broadcasts over heads.
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    y_even = x_even * cos - x_odd * sin
    y_odd  = x_even * sin + x_odd * cos

    # 5) put pairs back into the original shape
    y = torch.empty_like(x)                  # (..., T, d_k)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    B, T, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]

    d_k_head = d_k // num_heads
    d_v_head = d_v // num_heads

    # 1) project once
    Q = in_features @ q_proj_weight.transpose(-2, -1)  # (B, T, d_k)
    K = in_features @ k_proj_weight.transpose(-2, -1)  # (B, T, d_k)
    V = in_features @ v_proj_weight.transpose(-2, -1)  # (B, T, d_v)

    # 2) split into heads -> (B, H, T, d_head)
    Q = Q.view(B, T, num_heads, d_k_head).transpose(1, 2)
    K = K.view(B, T, num_heads, d_k_head).transpose(1, 2)
    V = V.view(B, T, num_heads, d_v_head).transpose(1, 2)

    causal = torch.tril(torch.ones(T, T, device=in_features.device, dtype=torch.bool))

    # 4) loop over heads (now each head gets different slice)
    heads = []
    for i in range(num_heads):
        qi = Q[:, i, :, :]   # (..., T, d_k_head)
        ki = K[:, i, :, :]   # (..., T, d_k_head)
        vi = V[:, i, :, :]   # (..., T, d_v_head)
        head_i = scaled_dot_product_attention(qi, ki, vi, mask=causal)
        heads.append(head_i)

    # 5) concat heads: list of (..., T, d_v_head) -> (..., T, d_v)
    multi_head = torch.cat(heads, dim=-1)  # (..., T, d_v)

    # 6) output projection back to d_model
    out = multi_head @ o_proj_weight.transpose(-2, -1)  # (..., T, d_model)

    return out


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    B, T, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]

    d_k_head = d_k // num_heads
    d_v_head = d_v // num_heads

    # 1) project once
    Q = in_features @ q_proj_weight.transpose(-2, -1)  # (B, T, d_k)
    K = in_features @ k_proj_weight.transpose(-2, -1)  # (B, T, d_k)
    V = in_features @ v_proj_weight.transpose(-2, -1)  # (B, T, d_v)

    # 2) split into heads -> (B, H, T, d_head)
    Q = Q.view(B, T, num_heads, d_k_head).transpose(1, 2)
    K = K.view(B, T, num_heads, d_k_head).transpose(1, 2)
    V = V.view(B, T, num_heads, d_v_head).transpose(1, 2)
    
    #add rope
    # apply RoPE to Q and K (RoPE dim = d_k_head)
    if token_positions is None:
        T = in_features.shape[1]
        B = in_features.shape[0]
        token_positions = torch.arange(T).unsqueeze(0).expand(B, T)
        
    Q = rope(d_k_head, theta, max_seq_len, Q, token_positions)  # (B, H, T, d_k_head)
    K = rope(d_k_head, theta, max_seq_len, K, token_positions)
    
    causal = torch.tril(torch.ones(T, T, device=in_features.device, dtype=torch.bool))

    # 4) loop over heads (now each head gets different slice)
    heads = []
    for i in range(num_heads):
        qi = Q[..., i, :, :]   # (..., T, d_k_head)
        ki = K[..., i, :, :]   # (..., T, d_k_head)
        vi = V[..., i, :, :]   # (..., T, d_v_head)
        head_i = scaled_dot_product_attention(qi, ki, vi, mask=causal)
        heads.append(head_i)

    # 5) concat heads: list of (..., T, d_v_head) -> (..., T, d_v)
    multi_head = torch.cat(heads, dim=-1)  # (..., T, d_v)

    # 6) output projection back to d_model
    out = multi_head @ o_proj_weight.transpose(-2, -1)  # (..., T, d_model)

    return out


