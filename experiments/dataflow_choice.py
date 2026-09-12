"""Retracted finding: per-phase dataflow does NOT change DRAM traffic.

An earlier version of this model claimed that switching the weight-gradient
GEMM to output-stationary saved up to a third of the training step, because
weight-stationary `wgrad` was thought to thrash SRAM on its token-length
operand.

A Timeloop cross-check (validate/) disproved it: a weight-stationary `wgrad`
reaches the *same* minimal DRAM traffic by tiling the contraction (token)
dimension and keeping the small `dW` accumulator resident.  The dataflow
choice only moves ~1 % of cycles, not DRAM traffic.

This script now just shows the corrected model: WS and OS `wgrad` land on the
same traffic curve.  Kept as a record of the retraction.
"""

import os

from _common import Hardware, Memory, plt, PLOT_DIR
from simulator.compute import matmul_cycles
from simulator.memory import gemm_traffic

HW = Hardware(rows=256, cols=256, clock_hz=1e9)
BW = 100e9
SRAM_MIB = [1, 2, 4, 8, 16, 32, 64, 128]

# mlp_up wgrad: dW = 768 x 3072, contraction = 8192 tokens
M, N, K = 768, 3072, 8192

xs, traffic_ms = [], []
for mib in SRAM_MIB:
    mem = Memory(mib << 20, BW)
    t = gemm_traffic(M, N, K, HW, mem).total_bytes    # dataflow-independent now
    xs.append(mib)
    traffic_ms.append(t / mem.achievable_bw * 1e3)
    print(f"{mib:4d} MiB   wgrad DRAM traffic {t/1e6:7.1f} MB "
          f"({traffic_ms[-1]:6.2f} ms @ 100 GB/s)")

ws_cyc = matmul_cycles(M, N, K, HW, "ws")
os_cyc = matmul_cycles(M, N, K, HW, "os")
print(f"\nwgrad cycles:  ws {ws_cyc:,}   os {os_cyc:,}   "
      f"(os is {100*(ws_cyc-os_cyc)/ws_cyc:.1f}% fewer -- the only real difference)")

fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(xs, traffic_ms, "o-", color="#333",
        label="wgrad — weight- and output-stationary (identical)")
ax.set_xscale("log", base=2)
ax.set_xlabel("on-chip SRAM (MiB)")
ax.set_ylabel("wgrad memory time (ms @ 100 GB/s)")
ax.set_title("wgrad DRAM traffic vs SRAM — dataflow-independent (retraction)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
out = os.path.join(PLOT_DIR, "dataflow_choice.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
