"""Closed-form systems autotuner: the whole lever space, exhaustively, in ms.

Two things:
  1. The min-latency plan the autotuner picks for a range of hardware configs
     (it's config-dependent — that's the point of autotuning).
  2. The latency vs stored-activation-memory Pareto frontier over the lever
     space, for one config.

The lever space is backward reuse (2) × optimizer precision/residency (4) ×
recomputation (2) × microbatch size (5) = 80 configs. Every evaluation is a
closed-form `model_step`, so the full sweep is ~1.5 ms — faster than a
single generation of MONET's genetic algorithm.
"""

import os

from _common import Hardware, Memory, plt, PLOT_DIR, GPT2_BLOCK
from simulator.autotune import autotune, pareto, _candidates

CONFIGS = [
    ("edge  128² / 2 MiB / 50 GB/s",   Hardware(128, 128, 1e9), Memory(2 << 20, 50e9)),
    ("small 128² / 8 MiB / 100 GB/s",  Hardware(128, 128, 1e9), Memory(8 << 20, 100e9)),
    ("mid   256² / 16 MiB / 100 GB/s", Hardware(256, 256, 1e9), Memory(16 << 20, 100e9)),
    ("big   256² / 64 MiB / 400 GB/s", Hardware(256, 256, 1e9), Memory(64 << 20, 400e9)),
]

print("autotuned plan per hardware config:\n")
for name, hw, mem in CONFIGS:
    p = autotune(GPT2_BLOCK, hw, mem)
    print(f"  {name:32s} {p.step_ms:6.2f} ms   {p.label()}")
print(f"\n  ({p.n_evaluated} configs searched in {p.search_ms:.1f} ms each)\n")

# Pareto frontier for the mid config
name, hw, mem = CONFIGS[2]
allp = list(_candidates(GPT2_BLOCK, hw, mem))
front = pareto(GPT2_BLOCK, hw, mem)
budgets = [128, 64, 32, 16]
print(f"min-latency plan under an activation budget ({name}):")
for b in budgets:
    try:
        p = autotune(GPT2_BLOCK, hw, mem, memory_budget_mb=b)
        print(f"  ≤ {b:3d} MB acts -> {p.step_ms:6.2f} ms   {p.label()}")
    except ValueError:
        print(f"  ≤ {b:3d} MB acts -> infeasible")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter([p.activation_mb for p in allp], [p.step_ms for p in allp],
           s=18, color="#bbb", label="all 80 lever configs", zorder=2)
ax.plot([p.activation_mb for p in front], [p.step_ms for p in front],
        "o-", color="#d62728", label="Pareto frontier", zorder=3)
ax.set_xlabel("stored activation memory (MB)")
ax.set_ylabel("training step latency (ms)")
ax.set_title(f"Systems-lever Pareto frontier  ({name})")
ax.set_xscale("log", base=2)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
out = os.path.join(PLOT_DIR, "autotune.png")
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
