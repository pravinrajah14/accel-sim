# Architecture

Two things to hold separate: the **hardware the model represents** and the
**software that computes the estimate**.

## 1. The modelled accelerator

```
        ┌──────────────────────────────────────────────────────┐
        │                  DRAM  (off-chip)                     │
        │   knob: bandwidth (GB/s)   ·   achievable = 0.75×peak │
        └───────────────────────────┬──────────────────────────┘
                                    │
                     data movement  │  bytes ÷ achievable_bw = memory_s
                                    │
        ┌───────────────────────────┴──────────────────────────┐
        │                  on-chip SRAM                         │
        │   knob: capacity (bytes)                              │
        │   holds the stationary weight + a slab of activations │
        │   too small ⇒ weight tiled ⇒ activations re-streamed  │
        └───────────────────────────┬──────────────────────────┘
                                    │  operands stream in
                                    │
        ┌───────────────────────────┴──────────────────────────┐
        │            systolic array   rows × cols @ clock        │
        │                                                       │
        │   cycles = ⌈M·N·K / (rows·cols)⌉  +  K + rows + cols   │
        │            └─ MAC term ──────┘    └─ fixed overhead ─┘  │
        └──────────────────────────────────────────────────────┘
```

* **Three knobs.** Array size (`rows × cols`), SRAM capacity, DRAM
  bandwidth. One dataflow family (weight-stationary; `matmul_cycles` also
  models output-stationary but it moves only ~1 % of cycles). One clock.
* **The array** is one closed-form equation, not a cycle loop. The MAC term
  dominates; the overhead is a fixed constant.
* **SRAM** is a single flat level. Its job in the model: whether the
  *smallest* of the GEMM's three tensors stays resident (stream the other
  two, penalty 0) or spills (re-streaming penalty).
* **DRAM** is a bandwidth number. No latency, no banks, no burst modelling.
* **What's abstracted away:** NoC, PE clusters, the accumulator hierarchy,
  DVFS, all dataflows but weight-stationary (traffic is dataflow-independent
  anyway — see `dataflow_choice.md`).

## 2. The software

Strict layering, no import cycles — each file only depends on the ones above it.

```
   compute.py     Hardware · matmul_cycles(M,N,K)              [pure compute]
       ▲
       │
   memory.py      Memory · gemm_traffic() · roofline()         [+ data movement]
       ▲
       │
   dataflow.py    Layer · training_step() · model_step()       [fwd+bwd+optimizer]
       ▲          Optimizer · recompute= · microbatch=           (the levers)
       │
   workloads.py   GPT2_BLOCK = [Layer, ...]                     [just data]
       ▲
       │
   ┌───┴───────────────┬───────────────────┐
experiments/        validate/            tests/
  sweep_*.py          gen_timeloop_*        test_sanity.py
  optimizer_cost.py   run_timeloop.sh
  recomputation.py    compare.py  ──►  matmul_cycles vs Timeloop
                            (vendored bench/ = real Timeloop example design)
       │
       ▼
   results/  plots/*.png · validation/validation.csv · RESULTS.md
```

### One query, end to end

```
inputs:  Layer(M,K,N) ,  Hardware(rows,cols,clock) ,  Memory(sram,bw)
         [ optional: Optimizer , recompute , microbatch ]
                              │
                              ▼
                       training_step()
          ┌───────────────────┼─────────────────────┐
          ▼                   ▼                     ▼
   _forward_seconds    _backward_seconds      optimizer_seconds
   matmul_cycles       2× matmul_cycles       Adam bytes
   + gemm_traffic      + backward traffic     ÷ bandwidth
   → max(compute,mem)  (+ recompute GEMM)
                       → max(compute,mem)
          └───────────────────┼─────────────────────┘
                              ▼
              StepBreakdown(forward_s, backward_s,
                            optimizer_s, bounds)
                              │
                   model_step() sums over the 6 layers
                              ▼
              total step latency  +  per-layer / per-phase breakdown
```

* **Closed-form, no search.** Every number is arithmetic on the inputs —
  microseconds per query. Contrast Timeloop (mapping search, seconds) and
  ASTRA-Sim (event sim, minutes).
* **`max(compute, memory)` per phase.** Assumes perfect double-buffering:
  whichever of the array or the DRAM pipe is slower sets the phase time.
* **Levers default OFF.** The no-lever path is a deliberate pessimistic
  bound; turning on `state_resident`, `recompute`, `microbatch`, … always
  produces an explicit, measurable delta rather than silently changing the
  baseline.
* **Only the compute model is externally validated.** `matmul_cycles` is
  checked against Timeloop (1.2 % MAE). `gemm_traffic` and the backward
  traffic model are first-order and self-consistent but not yet
  cross-checked — see `RESULTS.md` and the limitations in the README.
* **`workloads.py` is data, not logic.** Swapping in a different model =
  a new list of `Layer` tuples, nothing else changes.
