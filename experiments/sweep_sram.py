"""Sweep on-chip SRAM capacity; hold array size and DRAM bandwidth fixed.

More SRAM -> the stationary weight fits on chip -> activations stop being
re-fetched per weight tile -> the step moves back toward compute-bound.
"""

from _common import Hardware, Memory, step_ms, dominant_bound, plot_sweep

KIB = 1024
SRAM_BYTES = [b * KIB for b in (128, 256, 512, 1024, 2048, 4096, 8192,
                                16384, 32768, 65536, 131072)]
HW = Hardware(rows=128, cols=128, clock_hz=1e9)

xs, ys, bounds = [], [], []
for sb in SRAM_BYTES:
    mem = Memory(sram_bytes=sb, dram_bandwidth_bytes_s=100e9)
    xs.append(sb / KIB / 1024)  # MiB
    ys.append(step_ms(HW, mem))
    bounds.append(dominant_bound(HW, mem))
    print(f"{xs[-1]:6.2f} MiB  {ys[-1]:8.3f} ms  [{bounds[-1]}]")

plot_sweep(xs, ys, bounds,
           xlabel="on-chip SRAM (MiB)",
           title="Training-step latency vs SRAM (array=128x128, BW=100 GB/s)",
           fname="sweep_sram.png", xlog=True)
