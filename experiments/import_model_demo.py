"""Prove `layers_from_torch` works on a model nobody hand-typed into
workloads.py -- a Llama-7B-shaped MLP block (SwiGLU-style gate/up/down,
d_model=4096, d_ff=11008), traced directly from a real `nn.Module`.

Needs torch (`pip install torch`) -- skips cleanly if it isn't installed,
same convention as validate/silicon/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("skipped: needs torch (pip install torch)")
    sys.exit(0)

from simulator.compute import Hardware
from simulator.memory import Memory
from simulator.dataflow import model_step, ADAM_FP32
from simulator.import_model import layers_from_torch


class LlamaMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)) -- three Linears, no bias."""

    def __init__(self, d_model=4096, d_ff=11008):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


model = LlamaMLP()
batch, seq = 4, 2048
example = torch.randn(batch, seq, 4096)

layers = layers_from_torch(model, example)
print(f"traced {len(layers)} nn.Linear calls from a Llama-7B-shaped MLP "
      f"(batch={batch}, seq={seq}) -- nobody hand-typed these into workloads.py:")
for l in layers:
    print(f"  {l.name:12s} M={l.M:6d}  K={l.K:5d}  N={l.N:5d}")

hw = Hardware(rows=256, cols=256, clock_hz=1e9)
mem = Memory(sram_bytes=32 << 20, dram_bandwidth_bytes_s=400e9)
total, parts = model_step(layers, hw, mem, ADAM_FP32)
print(f"\ntraining step (256^2 array, 32 MiB SRAM, 400 GB/s): {total * 1e3:.3f} ms")
for p in parts:
    print(f"  {p.layer:12s} fwd {p.forward_s*1e3:7.3f}  bwd {p.backward_s*1e3:7.3f}  "
          f"opt {p.optimizer_s*1e3:6.3f} ms  [{p.backward_bound}]")
