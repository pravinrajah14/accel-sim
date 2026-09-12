"""Sweep DRAM bandwidth; hold array size and SRAM fixed.

Lower bandwidth -> memory time grows -> the step crosses from compute-bound
to memory-bound.
"""

from _common import Hardware, Memory, step_ms, dominant_bound, plot_sweep

BW_GBS = [25, 50, 100, 200, 400, 800, 1600, 3200]
HW = Hardware(rows=128, cols=128, clock_hz=1e9)

xs, ys, bounds = [], [], []
for bw in BW_GBS:
    mem = Memory(sram_bytes=8 << 20, dram_bandwidth_bytes_s=bw * 1e9)
    xs.append(bw)
    ys.append(step_ms(HW, mem))
    bounds.append(dominant_bound(HW, mem))
    print(f"{bw:6d} GB/s  {ys[-1]:8.3f} ms  [{bounds[-1]}]")

plot_sweep(xs, ys, bounds,
           xlabel="DRAM bandwidth (GB/s)",
           title="Training-step latency vs DRAM bandwidth (array=128x128, SRAM=8 MiB)",
           fname="sweep_bandwidth.png", xlog=True)
