"""ToyModel dtype probe + FP16 accumulation demo for mixed-precision writeup."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def accumulation_demo() -> None:
    print("=== mixed_precision_accumulation ===")
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("fp32 accum fp32 addends:", float(s))

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp16 accum fp16 addends:", float(s))

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp32 accum += fp16 tensor:", float(s))

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("fp32 accum += fp16.float():", float(s))


def toy_autocast_dtypes() -> None:
    print("=== ToyModel autocast FP16 dtypes ===")
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    model = ToyModel(16, 8).cuda()
    x = torch.randn(4, 16, device="cuda")
    # Inspect dtypes inside forward with hooks
    dtypes: dict[str, str] = {}

    def hook_fc1(_m, _inp, out):
        dtypes["fc1_out"] = str(out.dtype)

    def hook_ln(_m, _inp, out):
        dtypes["ln_out"] = str(out.dtype)

    def hook_fc2(_m, _inp, out):
        dtypes["fc2_out_logits"] = str(out.dtype)

    model.fc1.register_forward_hook(hook_fc1)
    model.ln.register_forward_hook(hook_ln)
    model.fc2.register_forward_hook(hook_fc2)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        dtypes["param_fc1_in_autocast"] = str(model.fc1.weight.dtype)
        logits = model(x)
        loss = logits.float().pow(2).mean()
    loss.backward()
    dtypes["logits"] = str(logits.dtype)
    dtypes["loss"] = str(loss.dtype)
    g = model.fc1.weight.grad
    dtypes["grad_fc1"] = str(g.dtype) if g is not None else "None"
    for k, v in dtypes.items():
        print(f"  {k}: {v}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accumulation", action="store_true")
    p.add_argument("--toy-dtypes", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.all or args.accumulation or not (args.toy_dtypes or args.accumulation):
        accumulation_demo()
    if args.all or args.toy_dtypes:
        toy_autocast_dtypes()


if __name__ == "__main__":
    main()
