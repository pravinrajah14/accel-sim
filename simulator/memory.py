"""On-chip SRAM + off-chip DRAM data-movement model.

`gemm_traffic(M, N, K, hw, mem)` estimates the DRAM bytes a *well-scheduled*
GEMM moves.  Two terms:

  compulsory = (M·K + K·N + M·N) · b        -- every element crosses DRAM once
  penalty    = tiling overhead when even the best blocking does not fit

The scheduler's job is to keep the **smallest** of the three tensors resident
and stream the other two once each (tile the contraction if needed).  That
works -- penalty 0 -- whenever the smallest tensor plus a thin working strip
fits in SRAM.  Only when it doesn't do you pay a re-streaming penalty,
≈ 2·M·N·K·b / √(S/2b).

This was **validated against Timeloop**: the compulsory term is exact
(0 % error, 15 configs), including a weight-stationary `wgrad` with a 25 MB
stationary operand that still hits compulsory traffic at 8 MB SRAM by tiling
the token dimension -- which the earlier `ceil(held/sram)` model got badly
wrong.  Consequence: **DRAM traffic does not depend on the dataflow**
(weight- vs output-stationary); the dataflow only moves ~1-3 % of cycles.

The penalty term is the classic matmul I/O lower bound
(Hong & Kung 1981, `Omega(M*N*K / sqrt(S))`); its 2x constant is a
first-principles estimate, not a Timeloop-calibrated fit.  It assumes an
*optimal* schedule -- which is what Timeloop's mapper finds and what the
staircase validation confirms (penalty ~ 0 at 8 MB).

`Memory.library_gemm=True` switches the spill term to a *fixed-kernel* model
instead: a GEMM library (cuBLAS etc.) does not retile the contraction to the
Hong-Kung optimum, so when an operand does not fit it is re-streamed roughly
`ceil(bytes / (S/2))` times.  This is the regime the T4 silicon anchor sits
in (`docs/silicon_anchor.md`); it is *off* by default so the abstract model
and the Timeloop cross-check are unchanged.

Other assumptions:
* Double-buffered: layer latency = max(compute, memory).
* Achievable bandwidth = peak × efficiency (default 0.75).
"""

from dataclasses import dataclass
import math

from .compute import Hardware, matmul_seconds


@dataclass(frozen=True)
class Memory:
    sram_bytes: int = 2 * 1024 * 1024          # 2 MiB on-chip SRAM
    dram_bandwidth_bytes_s: float = 100e9      # 100 GB/s off-chip
    dram_efficiency: float = 0.75             # fraction of peak actually reached
    library_gemm: bool = False               # see gemm_traffic penalty term

    # A real training loop's peak memory is more than the sum of stored
    # activation tensors (`StepBreakdown.activation_bytes`): weight buffers,
    # autograd graph bookkeeping, and the CUDA caching allocator's own
    # overhead/fragmentation all add a roughly fixed floor that doesn't
    # shrink with a smaller microbatch. Default 0 = the abstract model's
    # existing behavior (activation_bytes alone). This is a step-level
    # constant, not a per-layer one -- add it once per model, not once per
    # `training_step` call (see docs/microbatching.md).
    framework_overhead_bytes: float = 0.0
    framework_overhead_scale: float = 1.0     # multiplies activation_bytes;
                                              # 1.0 = abstract model, unchanged

    @property
    def achievable_bw(self) -> float:
        return self.dram_bandwidth_bytes_s * self.dram_efficiency


@dataclass(frozen=True)
class Traffic:
    compulsory_bytes: float   # each tensor across DRAM once
    penalty_bytes: float      # re-streaming when blocking does not fit

    @property
    def total_bytes(self) -> float:
        return self.compulsory_bytes + self.penalty_bytes


def gemm_traffic(M: int, N: int, K: int, hw: Hardware, mem: Memory) -> Traffic:
    """DRAM bytes for one GEMM (M,K)x(K,N)->(M,N), best schedule at this SRAM."""
    b = hw.mac_bytes
    a, bb, z = M * K, K * N, M * N            # tensor sizes in elements
    compulsory = (a + bb + z) * b

    S = mem.sram_bytes
    # A good schedule keeps the smallest tensor resident and streams the other
    # two once (tiling the contraction if needed) -- penalty 0 -- as long as
    # that tensor plus a thin working strip (~ M+N+K elements) fits.
    resident = min(a, bb, z) * b
    strip = (M + N + K) * b
    if S <= 0 or resident + strip > S:
        # even that spills: the Hong-Kung I/O bound says re-reads are then
        # Omega(M*N*K / sqrt(S_elements)) -- use the classic 2x constant.
        s_elem = S / b if S > 0 else 1.0
        penalty = 2.0 * M * N * K * b / math.sqrt(max(1.0, s_elem))
        # never worse than re-reading each input once per resident-sized block
        cap = (a + bb) * b * max(1.0, resident / S) if S > 0 else float("inf")
        penalty = min(penalty, cap)

        if mem.library_gemm and S > 0 and K > 0:
            # fixed-kernel libraries sweep the full contraction K per output
            # tile instead of retiling K to the Hong-Kung optimum.  An output
            # tile is at most T x T where the working set of both input
            # K-strips plus the accumulator and its double-buffer fits:
            # ~4*K*T <= S.  If T < min(M, N) the inputs are re-swept, and
            # total input reads ~ 2*M*N*K / T.
            t = (S / b) / (4.0 * K)
            if t < min(M, N):
                t = max(1.0, t)
                input_reads = 2.0 * M * N * K / t
                lib_penalty = max(0.0, input_reads * b - (a + bb) * b)
                lib_penalty = min(lib_penalty, 32.0 * (a + bb) * b)  # bound
                penalty = max(penalty, lib_penalty)
    else:
        penalty = 0.0
    return Traffic(float(compulsory), float(penalty))


def memory_seconds(M: int, N: int, K: int, hw: Hardware, mem: Memory) -> float:
    """Seconds to move the GEMM traffic across DRAM."""
    return gemm_traffic(M, N, K, hw, mem).total_bytes / mem.achievable_bw


@dataclass(frozen=True)
class RooflineResult:
    compute_s: float
    memory_s: float

    @property
    def latency_s(self) -> float:
        return max(self.compute_s, self.memory_s)

    @property
    def bound(self) -> str:
        return "compute" if self.compute_s >= self.memory_s else "memory"

    @property
    def arithmetic_intensity_ratio(self) -> float:
        """compute_s / memory_s ; >1 compute-bound, <1 memory-bound."""
        return self.compute_s / self.memory_s if self.memory_s > 0 else math.inf


def roofline(M: int, N: int, K: int, hw: Hardware, mem: Memory,
             dataflow: str = "ws") -> RooflineResult:
    return RooflineResult(
        compute_s=matmul_seconds(M, N, K, hw, dataflow),
        memory_s=memory_seconds(M, N, K, hw, mem),
    )


if __name__ == "__main__":
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    for mb in (2, 8, 32, 128):
        mem = Memory(sram_bytes=mb << 20, dram_bandwidth_bytes_s=100e9)
        t = gemm_traffic(8192, 3072, 768, hw, mem)
        print(f"{mb:3d} MiB  compulsory {t.compulsory_bytes/1e6:6.1f} MB  "
              f"penalty {t.penalty_bytes/1e6:7.1f} MB  -> {t.total_bytes/1e6:.1f} MB")
