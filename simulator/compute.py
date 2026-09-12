"""Systolic-array cycle model.

`matmul_cycles(M, N, K, hw, dataflow)` returns the array clock cycles for a
dense GEMM of shape (M, K) x (K, N) -> (M, N).

Two dataflows:

* **"ws"** weight-stationary -- the K x N right operand is held in the array
  (K -> rows, N -> cols); the left operand streams through.  Overhead is the
  K-cycle shift-in of the stationary operand.

* **"os"** output-stationary -- an R x C tile of the M x N output is held in
  the accumulators; both inputs stream.  Overhead is the pipeline drain per
  output tile, i.e. ceil(M/rows) * ceil(N/cols) * (rows + cols).

Both share the MAC term ceil(M*N*K / (rows*cols)).  The dataflow choice
matters most for the backward weight-gradient GEMM -- see docs.

KNOWN LIMITATION (v1)
--------------------
The cycle model still assumes the streamed operands are perfectly
double-buffered against DRAM (Timeloop agrees: its pinned-mapping cycle
count for these shapes is exactly the MAC term).  The cost of a bad dataflow
therefore shows up in `simulator.memory` (DRAM traffic), not here.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Hardware:
    """Accelerator configuration.

    `compute_efficiency` is the fraction of the peak MAC rate a real GEMM
    sustains -- the compute-side analogue of `Memory.dram_efficiency`.
    Default 1.0 = the idealised array the Timeloop cross-check validates
    (its pinned mappings hit the pure MAC rate).  A real GPU sustains far
    less (~0.5 on cuBLAS for these shapes); `validate/silicon/gpu_specs.py`
    sets it per device.  It never changes the abstract-model results.
    """

    rows: int = 128          # systolic array height (PE rows)
    cols: int = 128          # systolic array width  (PE cols)
    clock_hz: float = 1.0e9  # array clock frequency
    mac_bytes: int = 2       # bytes per operand element (fp16 activations/weights)
    compute_efficiency: float = 1.0   # sustained / peak MAC rate (GPU: ~0.5)
    kernel_launch_s: float = 0.0      # per-kernel launch latency (GPU: ~6e-6)
    util_tiles: int = 1      # output tiles needed to fill the machine; a GEMM
                             # with fewer runs at reduced occupancy (GPU: ~48).
                             # 1 = the ideal array (no occupancy penalty).

    # Elementwise/vector-ALU compute (Adam's mul/addcmul/sqrt/div chain,
    # softmax backward's dS from dP,P) -- separate from tensor-core
    # `peak_macs_per_s`, and priced at zero by default (0.0 = "not modeled",
    # the abstract model's existing behavior). `gpu_specs.py` sets a real
    # per-GPU rate, calibrated against the isolated Adam-only silicon anchor
    # (`docs/optimizer_state.md`).
    vector_flops_per_s: float = 0.0
    # Which `Optimizer.state_bytes` values this hardware runs at native ALU
    # throughput. fp32 (4) is native everywhere; Turing/Volta (T4, V100)
    # have no native bf16 or fp8 arithmetic, so PyTorch promotes/emulates --
    # `emulation_penalty` prices that. Default = every precision native
    # (the abstract model's existing behavior: state_bytes only ever helps).
    native_low_precision_bytes: frozenset = frozenset({4, 2, 1})
    emulation_penalty: float = 1.0    # multiplier on elementwise cost when
                                       # state_bytes isn't in the set above

    # `_occupancy`'s `util_tiles` penalty was calibrated from isolated
    # per-GEMM timing, where wgrad's small output (skinny GEMM: small M/N,
    # huge K) genuinely ran at reduced occupancy. A depth-isolated chained
    # anchor showed real chained backward runs far closer to
    # NO occupancy penalty at all (util_tiles=1) than to the isolated
    # calibration's util_tiles=128 -- once relieved, backward's residual
    # error matches forward's own (~11%), rather than the ~2.6x gap the
    # isolated-calibrated occupancy penalty implied. None (default) = use
    # `util_tiles` for backward too, the abstract model's existing
    # behavior. `gpu_specs.py` sets a real override per device, calibrated
    # against this chained anchor (`docs/silicon_anchor.md`).
    backward_util_tiles: int | None = None

    # A real chained model pays a roughly fixed, one-time overhead per
    # forward/backward pass (kernel-launch dispatch amortized across one
    # autograd graph) -- NOT per layer, unlike `_occupancy`/`kernel_launch_s`.
    # `profile_step.py`'s isolated per-GEMM calibration bakes this into an
    # apparent per-layer cost (each of its 6 shapes pays its own one-time
    # setup), which is why summing it over N layers overstates a real
    # N-layer chain. Both default to 0.0 (abstract model unchanged) --
    # these are NOT applied inside `training_step`/`model_step` (which stay
    # pure per-layer sums, reusable standalone); each anchor-scoring script
    # adds them once per chain at its own aggregation site, the same
    # pattern `Memory.framework_overhead_bytes` uses. Calibrated via a
    # depth-isolated OLS fit (`profile_chain_depth.py`, R^2 = 0.96-0.97,
    # one shape/one GPU -- see docs/silicon_anchor.md).
    chain_overhead_fwd_s: float = 0.0
    chain_overhead_bwd_s: float = 0.0

    @property
    def num_pes(self) -> int:
        return self.rows * self.cols

    @property
    def peak_macs_per_s(self) -> float:
        return self.num_pes * self.clock_hz

    @property
    def effective_macs_per_s(self) -> float:
        return self.num_pes * self.clock_hz * self.compute_efficiency


def matmul_cycles(M: int, N: int, K: int, hw: Hardware,
                  dataflow: str = "ws") -> int:
    """Cycles for a single (M,K)x(K,N) GEMM on the array.

      mac_cycles = ceil(M * N * K / (rows * cols))    -- shared by both dataflows
      ws overhead = K + rows + cols   -- shift the K x N stationary operand in
      os overhead = rows + cols       -- no shift-in; both inputs stream

    The cost of choosing the wrong dataflow for a phase is a *traffic* cost
    (`simulator.memory`), not a cycle cost -- see docs/dataflow_choice.md.
    """
    if min(M, N, K) <= 0:
        return 0
    mac_cycles = math.ceil((M * N * K) / hw.num_pes)
    fill_drain = hw.rows + hw.cols
    if dataflow == "ws":
        overhead = K + fill_drain
    elif dataflow == "os":
        overhead = fill_drain
    else:
        raise ValueError(f"unknown dataflow {dataflow!r}")
    return int(mac_cycles + overhead)


def _occupancy(M: int, N: int, hw: Hardware) -> float:
    """Fraction of the machine a GEMM with this output shape can fill.

    1.0 for the ideal array (`util_tiles == 1`).  For a GPU, a GEMM whose
    output has fewer than `util_tiles` array-sized tiles leaves SMs idle --
    this is what makes a skinny `wgrad` (small M or N, huge K) far slower
    than a forward GEMM of the same FLOP count.
    """
    if hw.util_tiles <= 1:
        return 1.0
    tiles = math.ceil(M / hw.rows) * math.ceil(N / hw.cols)
    return min(1.0, tiles / hw.util_tiles)


def matmul_seconds(M: int, N: int, K: int, hw: Hardware,
                   dataflow: str = "ws") -> float:
    """Wall-clock seconds for the GEMM (compute only), including the
    sustained-vs-peak `compute_efficiency` and output-shape `_occupancy`
    factors (both 1.0 for the ideal array)."""
    eff = hw.clock_hz * hw.compute_efficiency * _occupancy(M, N, hw)
    return matmul_cycles(M, N, K, hw, dataflow) / eff


def elementwise_seconds(n_elements: int, hw: Hardware, *, native: bool = True) -> float:
    """Wall-clock seconds for one elementwise/reduction pass over
    `n_elements` at `hw.vector_flops_per_s`.

    Zero for the abstract model (`vector_flops_per_s` defaults to 0.0 --
    "not modeled"). On real hardware this prices GPU compute that used to be
    priced at zero everywhere: Adam's elementwise update chain
    (`simulator.dataflow.optimizer_seconds`) and softmax backward's
    elementwise/reduction work (`simulator.attention.attention_step`'s
    fused backward). `native=False` applies `hw.emulation_penalty` -- the
    real cost of PyTorch promoting/emulating a dtype the hardware has no
    native ALU throughput for (e.g. bf16 on Turing/Volta).
    """
    if hw.vector_flops_per_s <= 0:
        return 0.0
    penalty = 1.0 if native else hw.emulation_penalty
    return (n_elements / hw.vector_flops_per_s) * penalty


def ideal_mac_cycles(M: int, N: int, K: int, hw: Hardware) -> int:
    """Lower bound: perfectly packed MACs with zero load/fill overhead."""
    return math.ceil((M * N * K) / hw.num_pes)


if __name__ == "__main__":
    # Sanity check: for a large GEMM the total should converge to the simple
    # MACs / array_size formula (overhead terms become negligible).
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    M, N, K = 8192, 3072, 768
    total = matmul_cycles(M, N, K, hw)
    ideal = ideal_mac_cycles(M, N, K, hw)
    print(f"total cycles : {total:,}")
    print(f"ideal cycles : {ideal:,}")
    print(f"overhead     : {100 * (total - ideal) / ideal:.3f}%")
    assert total >= ideal
    assert (total - ideal) / ideal < 0.02, "overhead should be <2% for a big GEMM"
    print("OK")
