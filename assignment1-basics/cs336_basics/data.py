from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient for np.memmap: only loads (B*(T+1)) tokens per call.
    Returns x,y of shape (B,T) as torch.long on `device`.
    """
    assert dataset.ndim == 1, "dataset must be 1D"
    n = int(dataset.shape[0])
    assert n > context_length + 1

    # random start indices on CPU (numpy)
    starts = np.random.randint(0, n - (context_length + 1), size=batch_size)

    # gather contiguous blocks (small copy only)
    x_np = np.stack([dataset[s : s + context_length] for s in starts]).astype(np.int64)
    y_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int64)

    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y
