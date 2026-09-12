"""Gradient accumulation / microbatching: the memory-vs-latency knob.

Split the batch into ceil(M / mb) microbatches, accumulate dW, run the
optimizer once. **Stored activations shrink linearly** with the microbatch
size -- the whole point. In exchange you pay K x the per-GEMM fixed
overhead, K x the weight re-reads, and smaller (lower arithmetic-intensity)
GEMMs, so latency rises ~5-15 %.

GPT-2 block, 8192 tokens, array 256x256, 100 GB/s, 32 MiB SRAM.
"""

import os

from _common import Hardware, Memory, plt, PLOT_DIR, GPT2_BLOCK, model_step

HW = Hardware(rows=256, cols=256, clock_hz=1e9)
MEM = Memory(32 << 20, 100e9)
MB_SIZES = [8192, 4096, 2048, 1024, 512, 256]

xs, lat, act = [], [], []
for mb in MB_SIZES:
    t, parts = model_step(GPT2_BLOCK, HW, MEM, microbatch=mb)
    xs.append(mb)
    lat.append(t * 1e3)
    act.append(sum(p.activation_bytes for p in parts) / 1e6)
    n = (8192 + mb - 1) // mb
    print(f"microbatch {mb:>5} (n={n:>2})  step {t*1e3:6.2f} ms "
          f"(+{(lat[-1]/lat[0]-1)*100:4.0f}%)   stored activations {act[-1]:6.1f} MB "
          f"({act[-1]/act[0]*100:3.0f}%)")

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
ax2 = ax1.twinx()
l1, = ax1.plot(xs, lat, "o-", color="#d62728", label="step latency")
l2, = ax2.plot(xs, act, "s-", color="#1f77b4", label="stored activations")
ax1.set_xscale("log", base=2)
ax1.invert_xaxis()
ax1.set_xlabel("microbatch size (tokens)   ←  more gradient accumulation")
ax1.set_ylabel("training step latency (ms)", color="#d62728")
ax2.set_ylabel("stored activation memory (MB)", color="#1f77b4")
ax1.set_title("Gradient accumulation: linear memory cut, ~10 % latency cost")
ax1.grid(alpha=0.3)
ax1.legend(handles=[l1, l2], loc="center right")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "microbatching.png")
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
print("note: near the SRAM tiling-penalty threshold (~4 MiB here) the smaller "
      "GEMMs can also *lower* latency, but that rests on the heuristic penalty term.")
