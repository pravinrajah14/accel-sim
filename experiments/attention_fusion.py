"""Naive vs fused (flash-attention-style) attention, swept over sequence
length.

Unlike `optimizer_cost.py`'s lever, fusion
here is *not* SRAM-capacity-gated -- see `simulator/attention.py`'s module
docstring for why (the on-chip working set is one tile, independent of S).
So the interesting sweep isn't SRAM, it's sequence length: naive attention's
DRAM traffic is O(S^2) (materializing the S x S score tensor), fused
attention's is O(S) (just Q, K, V, O) -- the saving *grows* with sequence
length, exactly where naive attention hurts most (see attention_scaling.py).
"""

from _common import Hardware, Memory, plt, PLOT_DIR
import os

from simulator.attention import Attention, attention_step
from simulator.workloads import BATCH, N_HEADS, D_HEAD

HW = Hardware(rows=256, cols=256, clock_hz=1e9)
MEM = Memory(sram_bytes=8 << 20, dram_bandwidth_bytes_s=100e9)
SEQS = [128, 256, 512, 1024, 2048, 4096, 8192]

xs, naive_ms, fused_ms, naive_act, fused_act = [], [], [], [], []
for seq in SEQS:
    attn = Attention("attention", batch=BATCH, seq=seq, n_heads=N_HEADS, d_head=D_HEAD)
    n = attention_step(attn, HW, MEM, fused=False)
    f = attention_step(attn, HW, MEM, fused=True)
    xs.append(seq)
    naive_ms.append(n.total_s * 1e3)
    fused_ms.append(f.total_s * 1e3)
    naive_act.append(n.activation_bytes / 1e6)
    fused_act.append(f.activation_bytes / 1e6)
    print(f"seq {seq:5d}   naive {n.total_s*1e3:9.3f} ms   fused {f.total_s*1e3:8.3f} ms   "
          f"saved {(n.total_s - f.total_s) / n.total_s * 100:5.1f}%   "
          f"activations {n.activation_bytes/1e6:8.1f} -> {f.activation_bytes/1e6:6.3f} MB")

best = max((n - f) / n * 100 for n, f in zip(naive_ms, fused_ms))
print(f"\nmax latency saved by fusion: {best:.1f}%  "
      f"(activations cut {naive_act[-1] / fused_act[-1]:.0f}x at seq={xs[-1]})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(xs, naive_ms, "o-", color="#d62728", label="naive (materializes S×S scores)")
ax1.plot(xs, fused_ms, "o-", color="#1f77b4", label="fused (flash-attention-style)")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("sequence length (tokens), batch=8")
ax1.set_ylabel("attention latency (ms, log scale)")
ax1.set_title("Latency: naive vs fused")
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=8)

ax2.plot(xs, naive_act, "o-", color="#d62728", label="naive")
ax2.plot(xs, fused_act, "o-", color="#1f77b4", label="fused")
ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.set_xlabel("sequence length (tokens), batch=8")
ax2.set_ylabel("stored activations (MB, log scale)")
ax2.set_title("Stored activations: naive vs fused")
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=8)

fig.suptitle("Attention fusion (array 256×256, 8 MiB, 100 GB/s)")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "attention_fusion.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
