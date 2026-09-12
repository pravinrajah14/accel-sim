"""Sweep systolic-array edge length; hold SRAM and DRAM bandwidth fixed.

Larger array -> less compute time -> the training step crosses from
compute-bound to memory-bound.
"""

from _common import Hardware, Memory, step_ms, dominant_bound, plot_sweep

SIDES = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
MEM = Memory(sram_bytes=8 << 20, dram_bandwidth_bytes_s=100e9)

xs, ys, bounds = [], [], []
for s in SIDES:
    hw = Hardware(rows=s, cols=s, clock_hz=1e9)
    xs.append(s)
    ys.append(step_ms(hw, MEM))
    bounds.append(dominant_bound(hw, MEM))
    print(f"{s:4d}x{s:<4d}  {ys[-1]:8.3f} ms  [{bounds[-1]}]")

plot_sweep(xs, ys, bounds,
           xlabel="array edge length (PEs)",
           title="Training-step latency vs array size (SRAM=8 MiB, BW=100 GB/s)",
           fname="sweep_array_size.png", xlog=True)
