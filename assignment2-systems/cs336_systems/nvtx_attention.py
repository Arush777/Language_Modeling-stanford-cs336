"""NVTX-annotated SDPA for Nsight Systems (handout §2.1.4)."""

from __future__ import annotations

import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

from cs336_basics.nn_utils import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


def install_annotated_attention() -> None:
    import cs336_basics.model as model

    model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
