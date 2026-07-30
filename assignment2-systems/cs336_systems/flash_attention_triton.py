"""
FlashAttention-2 Triton forward + compiled PyTorch backward.

Why Triton?
  The PyTorch tiled version still launches many small kernels and moves tiles
  through the PyTorch dispatcher. A fused Triton kernel keeps Q/K/V tiles in
  SRAM (on-chip), runs the online-softmax loop there, and writes only O and L
  to HBM — that is the IO win.

Backward (handout flash_backward):
  Implemented with torch.compile'd PyTorch recompute (Eq. 13–19), not Triton.
  Optional full Triton backward is out of scope unless needed for leaderboard.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from cs336_systems.flash_attention_pytorch import get_flash_backward_compiled


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    One program instance = one query tile for one batch index.
    Launch grid: (T_q, batch)  — handout requirement.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # --- block pointers into this batch's (N, D) matrices ---
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Qi once; stay in SRAM for the whole key loop
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # Accumulators in fp32 (handout precision tip)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    # Query indices for causal mask within this tile
    q_start = query_tile_index * Q_TILE_SIZE
    q_offs = q_start + tl.arange(0, Q_TILE_SIZE)

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(0, n_k_tiles):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # S = Qi @ Kj^T * scale   (Q_TILE_SIZE, K_TILE_SIZE)
        S = tl.dot(Qi, tl.trans(Kj)) * scale

        # Zero-padded OOB keys from boundary_check would score S=0 and
        # pollute softmax unless we mask them. Causal alone only hides
        # them when k_idx >= N_KEYS > q_idx; non-causal needs this always.
        k_offs = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        valid = k_offs[None, :] < N_KEYS
        if is_causal:
            valid = valid & (q_offs[:, None] >= k_offs[None, :])
        S = tl.where(valid, S, -1e6)

        # Online softmax
        m_ij = tl.max(S, axis=1)
        mi_new = tl.maximum(mi, m_ij)
        P_tilde = tl.exp(S - mi_new[:, None])
        alpha = tl.exp(mi - mi_new)
        li = alpha * li + tl.sum(P_tilde, axis=1)
        # Cast P to V's dtype before multiply (handout tip)
        P_cast = P_tilde.to(Vj.dtype)
        Oi = Oi * alpha[:, None] + tl.dot(P_cast, Vj)
        mi = mi_new

        # Advance K,V tiles along the sequence axis
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Finalize O and L
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    """Triton FA2 forward + compiled PyTorch recompute backward."""

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton FA requires CUDA tensors"
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        batch, Nq, d = Q.shape
        Nk = K.shape[1]

        # Tile sizes: powers of 2, >= 16 (handout).
        #
        # SRAM budget (A100): this kernel keeps Qi, Kj, Vj, Oi, S, P in shared
        # memory. With Bq=Bk=64 and D=128 Triton asks for ~181KB but only
        # ~167KB is available → OutOfResources. So for large D we *must*
        # shrink the sequence tiles (still >= 16).
        #   D<=64  → 64×64 is fine (validated on this A100 job)
        #   D>=128 → cap at 32×32 (16×16 when the seq itself is tiny)
        max_tile = 32 if d >= 128 else 64
        q_tile = 16 if Nq <= 128 else max_tile
        k_tile = 16 if Nk <= 128 else max_tile
        q_tile = min(q_tile, max(16, triton.next_power_of_2(Nq)))
        k_tile = min(k_tile, max(16, triton.next_power_of_2(Nk)))

        O = torch.empty_like(Q)
        L = torch.empty(batch, Nq, device=Q.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(d)

        T_q = triton.cdiv(Nq, q_tile)
        flash_fwd_kernel[(T_q, batch)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Nq,
            Nk,
            scale,
            D=d,
            Q_TILE_SIZE=q_tile,
            K_TILE_SIZE=k_tile,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        backward_fn = get_flash_backward_compiled()
        dQ, dK, dV = backward_fn(Q, K, V, O, dO.contiguous(), L, ctx.is_causal)
        return dQ, dK, dV, None
