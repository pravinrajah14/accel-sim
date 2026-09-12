"""Closed-form systems autotuner.

Given a model + hardware, exhaustively search the systems-lever space for the
minimum-latency training step (optionally under a stored-activation budget).

The whole point: every evaluation is a closed-form `model_step` (~µs), so the
*entire* lever space -- optimizer precision/residency x recomputation x
microbatch size -- is brute-forced in under a millisecond. A genetic
algorithm (MONET) explores one lever, slowly; this covers three, exhaustively,
instantly.
"""

from dataclasses import dataclass
import time

from .compute import Hardware
from .memory import Memory
from .dataflow import model_step, Optimizer


# the discrete lever grid
_OPT = [("fp32 streamed", Optimizer(4, False)),
        ("fp32 resident", Optimizer(4, True)),
        ("bf16 resident", Optimizer(2, True)),
        ("fp8 resident", Optimizer(1, True))]
_RECOMPUTE = [False, True]
_MICRO_FRACTIONS = [1, 2, 4, 8, 16]   # microbatch = tokens // fraction


@dataclass(frozen=True)
class Plan:
    opt_name: str
    recompute: bool
    microbatch: int | None
    step_ms: float
    activation_mb: float
    n_evaluated: int = 0
    search_ms: float = 0.0

    def label(self) -> str:
        mb = "no accum" if self.microbatch is None else f"microbatch {self.microbatch}"
        rc = "recompute" if self.recompute else "store acts"
        return f"{self.opt_name} | {rc} | {mb}"


def _candidates(layers, hw, mem):
    tokens = layers[0].M
    for on, opt in _OPT:
        for rc in _RECOMPUTE:
            for frac in _MICRO_FRACTIONS:
                mb = None if frac == 1 else max(1, tokens // frac)
                total, parts = model_step(layers, hw, mem, opt,
                                          recompute=rc, microbatch=mb)
                act = sum(p.activation_bytes for p in parts) / 1e6
                yield Plan(on, rc, mb, total * 1e3, act)


def autotune(layers, hw: Hardware, mem: Memory, *,
             memory_budget_mb: float | None = None) -> Plan:
    """Best (min-latency) plan, optionally under a stored-activation budget."""
    t0 = time.perf_counter()
    plans = [p for p in _candidates(layers, hw, mem)
             if memory_budget_mb is None or p.activation_mb <= memory_budget_mb]
    dt = (time.perf_counter() - t0) * 1e3
    if not plans:
        raise ValueError(f"no plan fits under {memory_budget_mb} MB activations")
    best = min(plans, key=lambda p: p.step_ms)
    n = len(_OPT) * len(_RECOMPUTE) * len(_MICRO_FRACTIONS)
    return Plan(best.opt_name, best.recompute, best.microbatch,
                best.step_ms, best.activation_mb, n_evaluated=n, search_ms=dt)


def pareto(layers, hw: Hardware, mem: Memory) -> list[Plan]:
    """Latency vs stored-activation-memory frontier over the lever space."""
    plans = sorted(_candidates(layers, hw, mem), key=lambda p: p.activation_mb)
    front, best_lat = [], float("inf")
    for p in plans:
        if p.step_ms < best_lat:
            front.append(p)
            best_lat = p.step_ms
    return front


if __name__ == "__main__":
    from .workloads import GPT2_BLOCK

    for edge, sram, bw in [(128, 4, 100), (256, 16, 100), (256, 64, 400)]:
        hw = Hardware(edge, edge, 1e9)
        mem = Memory(sram << 20, bw * 1e9)
        p = autotune(GPT2_BLOCK, hw, mem)
        print(f"{edge}² / {sram} MiB / {bw} GB/s  ->  {p.step_ms:6.2f} ms  "
              f"[{p.label()}]   ({p.n_evaluated} configs in {p.search_ms:.1f} ms)")
