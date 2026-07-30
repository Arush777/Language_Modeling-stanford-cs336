"""
FlashAttention-2 in pure PyTorch (tiled forward + recompute backward).

First principles
----------------
Vanilla attention materializes S = QK^T / sqrt(d) of shape (Nq, Nk). That is O(N^2)
HBM traffic and memory — bad for long sequences.

FlashAttention tiles Q into blocks of size Bq and (K,V) into blocks of size Bk.
For each query tile we stream over key tiles and maintain an *online softmax*:
  m_i  = running row-max of scores
  l_i  = running sum of exp(score - m)
  O_i  = running weighted sum of V, rescaled when m updates
At the end: O = O / l, and L = m + log(l)  (log-sum-exp of the full row).

We NEVER store the full (Nq, Nk) attention matrix — only O and L (plus Q,K,V for backward).

Backward (Eq. 13–19 in the handout) recomputes P_ij = exp(S_ij - L_i) from Q,K,L
instead of reading a saved P from HBM.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _causal_mask(Bq: int, Bk: int, q_start: int, k_start: int, device: torch.device) -> Tensor:
    """True where query may attend to key (q_idx >= k_idx)."""
    q_idx = q_start + torch.arange(Bq, device=device)[:, None]
    k_idx = k_start + torch.arange(Bk, device=device)[None, :]
    return q_idx >= k_idx


def flash_forward_tiled(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    is_causal: bool = False,
    q_tile_size: int = 16,
    k_tile_size: int = 16,
) -> tuple[Tensor, Tensor]:
    """
    Algorithm 1 (handout) in PyTorch.

    Q, K, V: (batch, seq, d)  — tests use a flat leading batch (no explicit head dim).
    Returns:
      O: (batch, Nq, d)
      L: (batch, Nq)  log-sum-exp of attention scores per query
    """
    batch, Nq, d = Q.shape
    Nk = K.shape[1]
    scale = 1.0 / math.sqrt(d)
    device = Q.device
    dtype = Q.dtype

    # Output / L written tile by tile
    O = torch.zeros_like(Q)
    L = torch.zeros(batch, Nq, device=device, dtype=torch.float32)

    # Process one batch element at a time for clarity (still educational / correct).
    for b in range(batch):
        Qb = Q[b]  # (Nq, d)
        Kb = K[b]
        Vb = V[b]

        for i in range(0, Nq, q_tile_size):
            q_end = min(i + q_tile_size, Nq)
            Qi = Qb[i:q_end].to(torch.float32)  # (Bq, d)
            Bq = Qi.shape[0]

            # Online softmax state for this query tile (float32 for stability)
            Oi = torch.zeros(Bq, d, device=device, dtype=torch.float32)
            li = torch.zeros(Bq, device=device, dtype=torch.float32)
            mi = torch.full((Bq,), float("-inf"), device=device, dtype=torch.float32)

            for j in range(0, Nk, k_tile_size):
                k_end = min(j + k_tile_size, Nk)
                Kj = Kb[j:k_end].to(torch.float32)  # (Bk, d)
                Vj = Vb[j:k_end].to(torch.float32)

                # S_ij = Qi @ Kj^T / sqrt(d)   → (Bq, Bk)
                S = (Qi @ Kj.T) * scale
                if is_causal:
                    mask = _causal_mask(Bq, Kj.shape[0], i, j, device)
                    S = S.masked_fill(~mask, -1e6)

                # --- online softmax update (the core FA trick) ---
                m_ij = S.max(dim=-1).values  # rowmax of this tile
                mi_new = torch.maximum(mi, m_ij)

                # P̃ = exp(S - mi_new)
                P_tilde = torch.exp(S - mi_new[:, None])
                # Rescale previous denominator and output by exp(mi - mi_new)
                alpha = torch.exp(mi - mi_new)
                li = alpha * li + P_tilde.sum(dim=-1)
                Oi = Oi * alpha[:, None] + P_tilde @ Vj
                mi = mi_new

            # Finalize: O = diag(1/l) O ;  L = m + log(l)
            O[b, i:q_end] = (Oi / li[:, None]).to(dtype)
            L[b, i:q_end] = mi + torch.log(li)

    return O, L


def flash_backward_recompute(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    O: Tensor,
    dO: Tensor,
    L: Tensor,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Handout Eq. 13–19 (recompute P from Q,K,L; no saved attention matrix).

      S = QK^T / sqrt(d)
      P_ij = exp(S_ij - L_i)
      D_i = sum_j (O ∘ dO)_ij     [= rowsum(P ∘ dP)]
      dV = P^T dO
      dP = dO V^T
      dS_ij = P_ij * (dP_ij - D_i)
      dQ = dS K / sqrt(d)
      dK = dS^T Q / sqrt(d)
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    # Work in float32 for the reductions
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    Of, dOf, Lf = O.float(), dO.float(), L.float()

    S = torch.einsum("bqd,bkd->bqk", Qf, Kf) * scale
    if is_causal:
        Nq, Nk = S.shape[-2], S.shape[-1]
        q_idx = torch.arange(Nq, device=S.device)[None, :, None]
        k_idx = torch.arange(Nk, device=S.device)[None, None, :]
        S = S.masked_fill(q_idx < k_idx, -1e6)

    P = torch.exp(S - Lf.unsqueeze(-1))
    D = (Of * dOf).sum(dim=-1)  # (batch, Nq)

    dV = torch.einsum("bqk,bqd->bkd", P, dOf)
    dP = torch.einsum("bqd,bkd->bqk", dOf, Vf)
    dS = P * (dP - D.unsqueeze(-1))
    dQ = torch.einsum("bqk,bkd->bqd", dS, Kf) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, Qf) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


# torch.compile accelerates the dense recompute backward (handout suggestion)
_flash_backward_compiled = None


def get_flash_backward_compiled():
    global _flash_backward_compiled
    if _flash_backward_compiled is None:
        _flash_backward_compiled = torch.compile(flash_backward_recompute, fullgraph=False)
    return _flash_backward_compiled


class FlashAttentionPyTorch(torch.autograd.Function):
    """Pure-PyTorch FlashAttention-2 autograd Function (handout flash_forward (a))."""

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False):
        O, L = flash_forward_tiled(Q, K, V, is_causal=is_causal)
        # Tests look for a saved tensor of shape (batch, Nq) == L
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        backward_fn = get_flash_backward_compiled()
        dQ, dK, dV = backward_fn(Q, K, V, O, dO, L, ctx.is_causal)
        # No gradient for the is_causal boolean
        return dQ, dK, dV, None
