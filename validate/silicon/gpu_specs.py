"""Datasheet specs for common training GPUs, and the mapping to an accel-sim
`Hardware` + `Memory` pair.

accel-sim models a square systolic array of `rows x cols` MAC units at
`clock_hz`.  A GPU is not a systolic array, but for a roofline-level latency
model only three numbers matter:

    peak MACs/s      <- dense fp16/bf16 tensor-core throughput / 2
    on-chip capacity <- L2 cache size (the "keep one tensor resident" store)
    off-chip BW      <- HBM / GDDR bandwidth

We turn peak MACs/s into a throughput-equivalent square array:

    rows = cols = round( sqrt( peak_macs_per_s / clock_hz ) )

so `rows * cols * clock_hz == peak_macs_per_s`.  The array *shape* only feeds
the fill/drain overhead term (K + rows + cols cycles), which is <5% of a
block GEMM here -- a documented, small mismatch.

TFLOP/s figures are DENSE, fp32-accumulate (what torch autocast does), NO
sparsity.  Treat them as +-10%.  L2 is the hardware L2 cache.  Clock is the
typical boost clock.
"""

from dataclasses import dataclass
import dataclasses
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from simulator.compute import Hardware
from simulator.memory import Memory


# Provisional, calibrated against the Tesla T4 fp16 anchor (n=1 GPU,
# 6 layers x 3 phases; see docs/silicon_anchor.md).  Fit: forward MAPE ~12%,
# step 0.87x.  Refine / average as more devices are profiled.
_COMPUTE_EFFICIENCY = 0.55    # sustained cuBLAS GEMM / datasheet tensor peak
_KERNEL_LAUNCH_S = 8e-6       # per elementwise-kernel launch latency
_UTIL_TILES = 128            # output tiles to fill the machine (skinny-GEMM knee)

# Calibrated against the isolated Adam-only anchor
# (`validate/silicon/optimizer_profile.json`, `compare_optimizer.py`'s
# `refit_elementwise`) -- n=1 GPU (Tesla T4), 4 layer sizes x 2 states (fp32,
# bf16). See docs/optimizer_state.md.
_VECTOR_FLOPS_PER_S = 4.571e9   # native (fp32) elementwise-ALU throughput
_EMULATION_PENALTY = 1.94       # bf16-on-Turing promote/emulate multiplier

# Turing (T4) and Volta (V100) have no native bf16 or fp8 tensor-core /
# vector-ALU throughput -- PyTorch promotes/emulates, which the T4 optimizer
# anchor showed costs MORE than the smaller byte count saves (measured bf16
# state -33%, not the naive "smaller bytes -> faster" +20%). Ampere+ has
# native bf16 (fp8 needs Hopper+, still untested -- see docs/optimizer_state.md).
_PRE_AMPERE = {"Tesla T4", "V100-SXM2", "V100"}

# Calibrated against a depth-isolated chained anchor
# (`validate/silicon/profiles/t4_fp16_chain_depth.json`, OLS fit in
# `compare_chain_depth.py`) -- one shape (768x768), one token count (8192),
# depths 1/2/4/6/8 on one GPU (Tesla T4). R^2 = 0.964 (forward), 0.974
# (backward). See docs/silicon_anchor.md.
#
# Forward's marginal per-layer cost was close to the isolated calibration's
# (ratio 0.89, within normal per-shape variance) -- left alone. Backward's
# was not (ratio 0.39): tried lowering compute_efficiency for backward
# first -- wrong direction (lower efficiency means SLOWER, not faster) and
# would need >1.0 to fit, which is physically meaningless. The real driver
# is `_occupancy`'s util_tiles penalty on wgrad's skinny output shape,
# calibrated from isolated timing; relieving it (util_tiles=1, no penalty)
# brings backward's residual to ~11% -- matching forward's own gap, not a
# separate 2.6x miss.
_BACKWARD_UTIL_TILES = 1
_CHAIN_OVERHEAD_FWD_S = 0.785e-3   # fixed one-time forward overhead per chain
_CHAIN_OVERHEAD_BWD_S = 1.361e-3   # fixed one-time backward overhead per chain

# TRIED AND REVERTED: a `compute_efficiency` boost for backward GEMMs that
# are already occupancy-safe, calibrated against a GEMM-size sweep
# (`validate/silicon/profiles/t4_fp16_gemm_size.json`, depth=1, fixed 8192
# tokens, square shapes 256-4096) where the implied efficiency looked like
# a clean ~1.0 plateau (0.90-1.03) across 4 sizes. It had ZERO effect on
# the very cases it was fit on: those GEMMs are MEMORY-bound under
# `_backward_seconds`'s `max(compute, memory)` (weight matrix exceeds this
# GPU's L2, triggering the traffic penalty branch), so compute_efficiency
# never reaches the final result. Checking the arithmetic confirmed it:
# matching the measured time as memory-bound would need ~451 GB/s on a
# 320 GB/s GPU -- impossible. The real gap is in `gemm_traffic`'s
# unvalidated penalty-term heuristic, not compute_efficiency. See
# docs/silicon_anchor.md.


@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_tflops: float   # dense fp16/bf16 tensor-core, fp32 accumulate
    l2_mb: float
    hbm_gbs: float
    clock_ghz: float
    compute_efficiency: float = _COMPUTE_EFFICIENCY

    def hardware(self, mac_bytes: int = 2, chained: bool = False) -> Hardware:
        """`chained=True` applies the depth-isolated chained-execution
        corrections (`backward_util_tiles`, `chain_overhead_*`) -- ONLY
        correct for a real multi-layer autograd graph, never for an
        isolated single-GEMM measurement (`profile_step.py`'s own anchor,
        which this must stay `chained=False` for). See docs/silicon_anchor.md.
        """
        peak_macs = self.peak_tflops * 1e12 / 2.0
        clock_hz = self.clock_ghz * 1e9
        side = max(1, round(math.sqrt(peak_macs / clock_hz)))
        native = frozenset({4}) if self.name in _PRE_AMPERE else frozenset({4, 2, 1})
        hw = Hardware(rows=side, cols=side, clock_hz=clock_hz,
                      mac_bytes=mac_bytes,
                      compute_efficiency=self.compute_efficiency,
                      kernel_launch_s=_KERNEL_LAUNCH_S,
                      util_tiles=_UTIL_TILES,
                      vector_flops_per_s=_VECTOR_FLOPS_PER_S,
                      native_low_precision_bytes=native,
                      emulation_penalty=_EMULATION_PENALTY)
        if chained:
            hw = dataclasses.replace(hw, backward_util_tiles=_BACKWARD_UTIL_TILES,
                                     chain_overhead_fwd_s=_CHAIN_OVERHEAD_FWD_S,
                                     chain_overhead_bwd_s=_CHAIN_OVERHEAD_BWD_S)
        return hw

    def memory(self, dram_efficiency: float = 0.75) -> Memory:
        return Memory(sram_bytes=int(self.l2_mb * (1 << 20)),
                      dram_bandwidth_bytes_s=self.hbm_gbs * 1e9,
                      dram_efficiency=dram_efficiency,
                      library_gemm=True)


# keyed by a lowercase substring matched against torch.cuda.get_device_name()
_SPECS = [
    GPUSpec("A100-SXM4-80GB", 312, 40, 2039, 1.41),
    GPUSpec("A100-SXM4-40GB", 312, 40, 1555, 1.41),
    GPUSpec("A100-PCIE",      312, 40, 1935, 1.41),
    GPUSpec("H100-SXM",       989, 50, 3352, 1.98),
    GPUSpec("H100-PCIe",      756, 50, 2039, 1.75),
    GPUSpec("H100-NVL",       835, 50, 3938, 1.98),
    GPUSpec("H200",           989, 50, 4800, 1.98),
    GPUSpec("V100-SXM2",      125,  6,  900, 1.53),
    GPUSpec("V100",           112,  6,  900, 1.38),
    GPUSpec("Tesla T4",        65,  4,  320, 1.59),
    GPUSpec("L4",             121, 48,  300, 2.04),
    GPUSpec("L40S",           362, 96,  864, 2.52),
    GPUSpec("L40",            181, 96,  864, 2.49),
    GPUSpec("A10G",           125,  6,  600, 1.71),
    GPUSpec("A10",            125,  6,  600, 1.70),
    GPUSpec("RTX 4090",       165, 72, 1008, 2.52),
    GPUSpec("RTX 3090",        71,  6,  936, 1.70),
    GPUSpec("RTX A6000",      155,  6,  768, 1.80),
    GPUSpec("RTX 6000 Ada",   182, 96,  960, 2.50),
    GPUSpec("A40",            150,  6,  696, 1.74),
]

_ALIASES = {
    "a100": "A100-SXM4-40GB",
    "h100": "H100-SXM",
    "t4": "Tesla T4",
    "v100": "V100-SXM2",
    "4090": "RTX 4090",
    "3090": "RTX 3090",
}


def lookup(device_name: str) -> GPUSpec | None:
    """Best-effort match of a torch device name to a spec."""
    if not device_name:
        return None
    d = device_name.lower()
    # exact-ish: longest spec name whose tokens all appear
    best = None
    for s in _SPECS:
        toks = [t for t in re.split(r"[\s-]+", s.name.lower()) if t]
        if all(t in d for t in toks):
            if best is None or len(s.name) > len(best.name):
                best = s
    if best:
        return best
    for key, target in _ALIASES.items():
        if key in d:
            return next(s for s in _SPECS if s.name == target)
    return None


def from_overrides(name, tflops, l2_mb, hbm_gbs, clock_ghz) -> GPUSpec:
    return GPUSpec(name or "custom", tflops, l2_mb, hbm_gbs, clock_ghz)


def catalog() -> list[GPUSpec]:
    return list(_SPECS)


if __name__ == "__main__":
    for s in _SPECS:
        hw, mem = s.hardware(), s.memory()
        print(f"{s.name:16s}  {s.peak_tflops:5.0f} TF  "
              f"array {hw.rows:4d}^2 @ {s.clock_ghz} GHz  "
              f"L2 {s.l2_mb:4.0f} MB  HBM {s.hbm_gbs:5.0f} GB/s")
