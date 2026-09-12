"""Multi-head self-attention: QK^T -> softmax -> (softmax)@V.

Extends accel-sim from "6 linear layers" to a full transformer layer. Batch
and sequence are explicit here -- unlike the linear layers in `dataflow.py`,
where batch and sequence collapse into one token count `M` -- because
attention's cost is `B*H` *independent* (S,Dh)x(Dh,S) and (S,S)x(S,Dh) GEMMs
with **no operand reuse across instances**: every (batch, head) slice of Q,
K, V is unique, unlike a weight-stationary linear layer where one weight is
reused across the whole token batch. That is the interesting modeling
difference from the rest of the codebase -- attention pays the systolic
array's fill/drain overhead `B*H` times over.

Two simplifications, stated plainly (v1 scope -- see docs/attention.md):

  * Softmax is priced as a **memory-only** pass over the (B,H,S,S) score
    tensor when not fused (one extra read + write beyond the QK^T write /
    scoreV read already counted). No separate compute-cycle model for
    exp/max, and no compute cost for the backward-through-softmax Jacobian
    (`dS` from `dP`, `P`) -- both fused and naive.
  * Backward's *GEMM* work (`dV`, `dP`, `dQ`, `dK`) is priced at 2x forward
    compute, the same convention `_backward_seconds` uses for a linear
    layer's dgrad+wgrad. **Naive backward stops there.** `fused` backward
    adds one more term -- see below -- which the T4 silicon anchor
    (`docs/attention.md`) showed matters a lot.

Attention fusion (`fused=True`)
--------------------------------
A flash-attention-style kernel never writes the full S x S score tensor to
DRAM: it tiles over K/V, keeps a running (row-max, row-sum) on chip, and
accumulates the output in place. That removes two things from the traffic
total, not just the softmax pass:

  1. the softmax read+write pass (as above), and
  2. the QK^T-output / scoreV-input round trip itself -- `gemm_traffic`'s
     compulsory term for those two GEMMs includes writing and re-reading the
     S x S score tensor; fused, it never leaves the array.

What crosses DRAM is then just Q, K, V (read once each) and the output O
(written once) -- O(S), not O(S^2). Unlike `Optimizer.state_resident`, this
isn't SRAM-capacity-gated the way that is: flash attention's on-chip
working set is one tile (`O(block_size * d_head)`),
independent of S by construction, so the benefit is modeled as unconditional
once fused -- the SRAM sizes this codebase considers (>=1 MiB) are far above
that tile's real footprint (a few KB).

Backward pays for this -- for real, not just in `activation_bytes`. Because
the score tensor `P = softmax(QK^T)` was never stored, computing `dV = P^T@dO`
and `dP = dO@V^T` needs `P` back, so a fused/flash-style backward
**recomputes** `S = QK^T` (and its softmax) from `Q`/`K` before it can do
anything else -- exactly the `recompute` lever (`dataflow.py`) applied to
attention instead of a GEMM's input activation, except here it isn't
optional: it's how the algorithm has to work once it didn't store `P`.

    fused backward compute = 2x forward compute   (dV, dP, dQ, dK -- same as naive)
                             + 1x QK^T compute     (recompute S, since P wasn't stored)

`activation_bytes` also drops from the full score tensor to ~0 (a small
per-row softmax-statistics footprint), because the score tensor is never
stored -- that part was already modeled; the extra compute term above was
not, until the T4 silicon anchor exposed the gap it left.

**This is a real, derived improvement, and it is still not enough.** It
moves the model's own backward:forward ratio from a flat 2.0x to ~2.4x
(GPT-2 shapes) -- correct *direction*, still short of the T4's measured
4.5x. Two candidate causes for the remainder, neither patched in from this
one data point: (a) the softmax backward itself (`dS` from `dP`, `P`) has
real elementwise + reduction compute that nothing in this codebase prices
(there is no "vector op" cost primitive anywhere, only GEMMs and DRAM
bytes); (b) the T4 is Turing (sm75) running PyTorch's memory-efficient
(not literal Flash) attention backend, which may simply run this backward
pattern less efficiently than a newer GPU's kernel would. See
docs/attention.md's Open section.
"""

from dataclasses import dataclass

from .compute import Hardware, matmul_cycles
from .memory import Memory, gemm_traffic


@dataclass(frozen=True)
class Attention:
    name: str
    batch: int      # B
    seq: int        # S -- tokens per sequence (NOT batch*seq; attention needs both)
    n_heads: int    # H
    d_head: int     # Dh

    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_head

    @property
    def score_elems(self) -> int:
        """Size of the materialized (B, H, S, S) attention-score tensor."""
        return self.batch * self.n_heads * self.seq * self.seq


def _forward_costs(attn: Attention, hw: Hardware, mem: Memory, fused: bool = False):
    """(compute_s, bytes_moved, score_bytes, qk_compute_s) for the forward pass.

    `score_bytes` is the stored-activation footprint for backward -- the
    full score tensor when not fused, ~0 (never materialized) when fused.
    `qk_compute_s` is just the QK^T instances' compute time (out of the
    total `compute_s`) -- fused backward needs it separately to price
    recomputing `S`/`P` (see module docstring).
    """
    B, S, H, Dh = attn.batch, attn.seq, attn.n_heads, attn.d_head
    b = hw.mac_bytes
    n = B * H  # independent GEMM instances -- no cross-instance reuse

    qk_cycles = matmul_cycles(S, S, Dh, hw)              # (S,Dh)x(Dh,S) -> (S,S)
    av_cycles = matmul_cycles(S, Dh, S, hw)              # (S,S)x(S,Dh)  -> (S,Dh)
    qk_compute_s = n * qk_cycles / hw.clock_hz
    compute_s = qk_compute_s + n * av_cycles / hw.clock_hz  # same MACs either way

    if fused:
        # Only Q, K, V (read once each) and O (written once) cross DRAM --
        # the S x S score tensor and its softmax stay in on-chip tiles.
        qkvo_bytes = 4 * B * S * attn.d_model * b
        bytes_moved = float(qkvo_bytes)
        stats_bytes = 2 * n * S * 4          # running (max, sum) per row, fp32
        return compute_s, bytes_moved, float(stats_bytes), qk_compute_s

    qk_traffic = gemm_traffic(S, S, Dh, hw, mem)
    av_traffic = gemm_traffic(S, Dh, S, hw, mem)
    gemm_bytes = n * (qk_traffic.total_bytes + av_traffic.total_bytes)

    score_bytes = n * S * S * b
    softmax_bytes = 2 * score_bytes           # one extra read + write, naive

    return compute_s, gemm_bytes + softmax_bytes, score_bytes, qk_compute_s


def attention_step(attn: Attention, hw: Hardware, mem: Memory, *, fused: bool = False):
    """Forward + backward for one self-attention block.

    `fused=True` models a flash-attention-style kernel: same forward compute,
    far less DRAM traffic, near-zero stored activations -- but a real extra
    backward compute cost (recomputing `S`/`P`, since they weren't stored;
    see module docstring).

    Returns a `dataflow.StepBreakdown` so it drops into `model_step`
    alongside the linear layers. `optimizer_s` is 0 -- attention (as
    modeled here, no learned QKV/output projections folded in) has no
    parameters of its own; those projections are the 4 linear layers
    (`q_proj`/`k_proj`/`v_proj`/`attn_out`) already in `workloads.py`.
    """
    from .dataflow import StepBreakdown   # local import: dataflow imports us

    compute_s, bytes_moved, score_bytes, qk_compute_s = _forward_costs(attn, hw, mem, fused)
    memory_s = bytes_moved / mem.achievable_bw
    fwd = max(compute_s, memory_s)
    fwd_bound = "compute" if compute_s >= memory_s else "memory"

    # backward GEMM work (dV, dP, dQ, dK) -- 2x forward compute, same
    # convention as a linear layer's dgrad+wgrad -- plus, fused only, the
    # cost of recomputing S/P from Q/K since they were never stored.
    #
    # TRIED AND REJECTED: applying `elementwise_seconds` (the same
    # native-elementwise rate calibrated against the optimizer anchor) to
    # softmax backward's compute over the (B,H,S,S) score tensor, to close
    # more of the fused-backward gap. That rate was fit against 0.5-2.4M
    # element tensors (Optimizer.state_bytes' m/v arrays); attention's score
    # tensor is 100M+ elements at GPT-2's own shapes (B=8, H=12, S=1024) --
    # 40-200x outside the fitted range. Applied there it predicts a ~22ms
    # overhead by itself, dwarfing the entire measured fused step -- a
    # physically implausible extrapolation, not a real fix. Not adopted.
    # The softmax-backward compute gap stays open, same as before
    # (docs/attention.md's Open section) -- it needs its own dedicated
    # attention-scale calibration, not a borrowed constant.
    bwd_compute = 2.0 * compute_s + (qk_compute_s if fused else 0.0)
    bwd_memory = 2.0 * memory_s
    bwd = max(bwd_compute, bwd_memory)
    bwd_bound = "compute" if bwd_compute >= bwd_memory else "memory"

    return StepBreakdown(
        layer=attn.name, forward_s=fwd, backward_s=bwd, optimizer_s=0.0,
        forward_bound=fwd_bound, backward_bound=bwd_bound,
        backward_bytes=2.0 * bytes_moved, activation_bytes=float(score_bytes),
    )
