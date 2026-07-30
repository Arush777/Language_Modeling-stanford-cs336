"""Model size table from CS336 A2 handout §2.1.2 (Table 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


# Handout Table 1 — GPT-2-like configs. Vocab/batch/context are run-time args.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig("small", 768, 3072, 12, 12),
    "medium": ModelConfig("medium", 1024, 4096, 24, 16),
    "large": ModelConfig("large", 1280, 5120, 36, 20),
    "xl": ModelConfig("xl", 2560, 10240, 32, 32),
    "10B": ModelConfig("10B", 4608, 12288, 50, 36),
}

DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_BATCH_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_ROPE_THETA = 10_000.0
