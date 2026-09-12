"""Adam optimizer cost vs SRAM, for three state precisions.

Holds the array (256x256) and DRAM (100 GB/s) fixed and sweeps SRAM.  For
each of fp32 / bf16 / fp8 optimizer state, plots the per-step optimizer time
when m, v are kept resident (they stop streaming to DRAM once they fit).

Two levers, both capacity-gated:
  * precision  -- halving state_bytes halves the m,v stream
  * residency  -- above the fit threshold, m,v never touch DRAM and only the
                  master weights are streamed
"""

import os

from _common import Hardware, Memory, plt, PLOT_DIR, GPT2_BLOCK
from simulator.dataflow import optimizer_seconds, Optimizer

HW = Hardware(rows=256, cols=256, clock_hz=1e9)
BW = 100e9
KIB = 1024
SRAM_MIB = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]

PRES = [("fp32", 4, "#d62728"), ("bf16", 2, "#ff7f0e"), ("fp8", 1, "#1f77b4")]


def block_opt_ms(mem, opt):
    return sum(optimizer_seconds(l, mem, opt) for l in GPT2_BLOCK) * 1e3


fig, ax = plt.subplots(figsize=(7.5, 4.5))

stream_fp32 = [block_opt_ms(Memory(m * KIB * KIB, BW), Optimizer(4, False))
               for m in SRAM_MIB]
ax.plot(SRAM_MIB, stream_fp32, "--", color="gray", label="fp32, streamed (baseline)")

print(f"{'SRAM':>7}  " + "  ".join(f"{n:>8}" for n, _, _ in PRES))
for name, sb, colour in PRES:
    ys = [block_opt_ms(Memory(m * KIB * KIB, BW), Optimizer(sb, True))
          for m in SRAM_MIB]
    ax.plot(SRAM_MIB, ys, "o-", color=colour, label=f"{name} state, resident")
for i, m in enumerate(SRAM_MIB):
    row = "  ".join(f"{block_opt_ms(Memory(m*KIB*KIB, BW), Optimizer(sb, True)):8.3f}"
                    for _, sb, _ in PRES)
    print(f"{m:5d}MiB  {row}")

ax.set_xscale("log", base=2)
ax.set_xlabel("on-chip SRAM (MiB)")
ax.set_ylabel("optimizer time per step (ms)")
ax.set_title("Adam optimizer cost vs SRAM (array 256×256, BW 100 GB/s)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
out = os.path.join(PLOT_DIR, "optimizer_cost.png")
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
