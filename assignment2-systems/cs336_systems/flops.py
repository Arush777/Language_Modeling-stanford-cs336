"""
Analytical FLOPs for BasicsTransformerLM (staff A2 cs336_basics.model).

Accounting convention (standard matmul FLOPs):
  A[M,K] @ B[K,N] costs 2*M*K*N FLOPs (multiply-add).

Forward pass (exact for the Linear/einsum matmuls in this architecture):
  Per TransformerBlock:
    - Attention: q/k/v projections (3 * Linear d_model→d_model),
      SDPA scores + weighted values, output projection.
    - SwiGLU: w1, w3 (d_model→d_ff), w2 (d_ff→d_model).
  Plus final ln is ignored (elementwise, negligible vs GEMMs).
  Plus lm_head Linear d_model→vocab.

Backward ≈ 2× forward GEMM FLOPs (standard training estimate).
 Optimizer AdamW step is not counted in model FLOPs (elementwise on params).

Measured throughput: FLOPs / wall_seconds → TFLOP/s.
"""

from __future__ import annotations

from dataclasses import dataclass

from cs336_systems.configs import ModelConfig


def _linear_flops(batch: int, seq: int, d_in: int, d_out: int) -> int:
    """FLOPs for (B,T,d_in) @ (d_out,d_in)^T → (B,T,d_out)."""
    return 2 * batch * seq * d_in * d_out


def _sdpa_flops(batch: int, seq: int, num_heads: int, d_head: int) -> int:
    """Causal SDPA matmul FLOPs only (scores QK^T and attn@V); softmax ignored."""
    # scores: [B,H,T,d] @ [B,H,d,T] → [B,H,T,T]
    scores = 2 * batch * num_heads * seq * seq * d_head
    # out: [B,H,T,T] @ [B,H,T,d] → [B,H,T,d]
    out = 2 * batch * num_heads * seq * seq * d_head
    return scores + out


@dataclass(frozen=True)
class FlopBreakdown:
    embedding: int  # 0 — gather, not counted as GEMM
    attention_proj: int
    attention_sdpa: int
    swiglu: int
    lm_head: int
    forward_total: int
    backward_total: int  # 2 * forward GEMMs
    train_total: int  # forward + backward (no optimizer FLOPs)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "flops_embedding": self.embedding,
            "flops_attention_proj": self.attention_proj,
            "flops_attention_sdpa": self.attention_sdpa,
            "flops_swiglu": self.swiglu,
            "flops_lm_head": self.lm_head,
            "flops_forward": self.forward_total,
            "flops_backward": self.backward_total,
            "flops_train_fwd_bwd": self.train_total,
        }


def transformer_flops(
    cfg: ModelConfig,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> FlopBreakdown:
    b, t = batch_size, seq_len
    d, d_ff, L, h = cfg.d_model, cfg.d_ff, cfg.num_layers, cfg.num_heads
    d_head = d // h

    # Per layer — CausalMultiHeadSelfAttention: q_proj, k_proj, v_proj, output_proj
    attn_proj_layer = (
        _linear_flops(b, t, d, d)  # q
        + _linear_flops(b, t, d, d)  # k
        + _linear_flops(b, t, d, d)  # v
        + _linear_flops(b, t, d, d)  # out
    )
    sdpa_layer = _sdpa_flops(b, t, h, d_head)
    # SwiGLU: w1, w3 up-proj; w2 down-proj
    swiglu_layer = (
        _linear_flops(b, t, d, d_ff)
        + _linear_flops(b, t, d, d_ff)
        + _linear_flops(b, t, d_ff, d)
    )

    attn_proj = L * attn_proj_layer
    sdpa = L * sdpa_layer
    swiglu = L * swiglu_layer
    lm_head = _linear_flops(b, t, d, vocab_size)

    forward = attn_proj + sdpa + swiglu + lm_head
    backward = 2 * forward
    return FlopBreakdown(
        embedding=0,
        attention_proj=attn_proj,
        attention_sdpa=sdpa,
        swiglu=swiglu,
        lm_head=lm_head,
        forward_total=forward,
        backward_total=backward,
        train_total=forward + backward,
    )


def flops_for_mode(breakdown: FlopBreakdown, mode: str) -> int:
    if mode == "forward":
        return breakdown.forward_total
    if mode == "forward_backward":
        return breakdown.train_total
    if mode == "train":
        # AdamW elementwise not included; same GEMM budget as fwd+bwd
        return breakdown.train_total
    raise ValueError(f"unknown mode: {mode}")


def residual_stream_activation_mib(batch: int, seq: int, d_model: int, dtype_bytes: int = 4) -> float:
    """Handout memory (d): size of residual-stream activation tensor in MiB."""
    return (batch * seq * d_model * dtype_bytes) / (1024**2)
