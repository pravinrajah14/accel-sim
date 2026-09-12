# Silicon anchor

Everything else in `validate/` checks accel-sim against **Timeloop** — which
is itself a model. This checks it against a **real GPU**.

The model is only validated top-to-bottom once its predictions have been put
next to a wall-clock measurement from actual silicon, with the error reported
honestly. That is what this directory does.

## Nine anchors

- **`profile_step.py` / `compare_silicon.py`** — the six weight GEMMs of a
  GPT-2-small block (forward, backward, Adam), each **in isolation**.
  Calibrated on one Tesla T4: forward 0.3%, backward 14%, optimizer 59%
  (see below), **step 6.7%** (`docs/silicon_anchor.md`). Its isolated-per-
  GEMM-sum methodology is itself the subject of `profile_chain_grid.py`
  below.
- **`profile_attention.py` / `compare_attention.py`** — naive
  (materializing) vs fused (PyTorch's `scaled_dot_product_attention`)
  multi-head attention, checking `simulator/attention.py`'s two modes
  against real kernels. Run on a T4: naive ~1.75× optimistic (GEMM-like),
  fused ~5.8× optimistic even after modeling the real backward-recompute
  cost (`docs/attention.md`).
- **`profile_recompute.py` / `compare_recompute.py`** — a real 4-layer chain,
  **stored** (normal autograd) vs **recompute** (`torch.utils.checkpoint`),
  checking the `recompute=` lever. Run on a T4: forward transfers (~13-18%
  error, GEMM-like), backward does **not** — model *over*-predicts by ~2x,
  the opposite direction from every other backward result here. Root cause
  identified: `profile_step.py`'s isolated-per-GEMM timing overstates real
  chained backward by ~2x on this GPU, not a shape-transfer problem per se
  (`docs/recomputation.md`).
- **`profile_gradaccum.py` / `compare_gradaccum.py`** — the same 4-layer
  chain run at `microbatch` fractions 1/2/4/8/16 of the token count,
  measuring step latency *and* peak activation memory
  (`torch.cuda.max_memory_allocated`) against the `microbatch=` lever. Run
  on a T4: latency shows the same isolated-vs-chained mismatch as the
  recompute anchor; **memory was fixed** with a 2-parameter floor+scale fit
  (`Memory.framework_overhead_bytes`/`framework_overhead_scale`) —
  measured/predicted ratio now ~1.0 at every fraction, was 3-12x low
  (`docs/microbatching.md`).
- **`profile_optimizer.py` / `compare_optimizer.py`** — a manual Adam update
  at fp32 vs bf16 state, no forward/backward, isolating `Optimizer.state_bytes`
  directly. Run on a T4: bf16 measured **33% slower** than fp32 while the
  model predicted +20% faster (Turing has no native bf16 arithmetic).
  **Fixed** with a new `elementwise_seconds` compute primitive + a native-
  dtype-support gate (`Hardware.native_low_precision_bytes`,
  `emulation_penalty`) — model now correctly predicts bf16 is slower on
  this GPU (`docs/optimizer_state.md`).
- **`profile_chain_grid.py` / `compare_chain_grid.py`** — real **chained**
  forward+backward+Adam (never isolated) across 4 configs, including the
  original 6-GEMM `GPT2_BLOCK` shapes end-to-end. Run on a T4: confirms
  real chained backward is 1.4-3.5× faster than the isolated-sum
  calibration predicts, but a 2-parameter `compute_efficiency`/`util_tiles`
  refit against the 4 configs doesn't generalize — depth, shape, and token
  count are confounded across them (`docs/silicon_anchor.md`).
- **`profile_chain_depth.py` / `compare_chain_depth.py`** — a depth-isolated
  follow-up: fixed 768×768 shape, fixed 8192 tokens, only chain depth
  varies (1/2/4/6/8 identical layers). Run on a T4: a clean fit
  (`measured_backward_ms ≈ 0.49·depth + 1.36`, R²=0.974), traced to
  `_occupancy`'s `util_tiles` penalty overstating wgrad's real chained
  cost. Added `Hardware.backward_util_tiles` + `chain_overhead_fwd_s`/
  `bwd_s`, **opt-in only** (`GPUSpec.hardware(chained=True)`) — tried
  against the other chain-grid configs and the real recompute/gradaccum
  anchors and it did not generalize (2/4 configs got worse; recompute's
  backward error barely moved). Real, narrow, honestly-scoped fix, not a
  general one (`docs/silicon_anchor.md`).
- **`profile_chain_generalize.py` / `compare_chain_generalize.py`** — a
  depth sweep + token-count sweep on a 2048×2048 shape verified
  occupancy-safe throughout (plus a repeated-config noise check). Run on a
  T4: depth sweep gave a perfect fit (R²=1.000) but shape-size-specific
  constants (nothing like 768×768's); a single unchained GEMM landed
  within 4.4% of `compute_efficiency=1.0`, pointing at a third entangled
  effect — efficiency rising with GEMM size — not isolated by this design
  (`docs/silicon_anchor.md`).
- **`profile_gemm_size.py` / `compare_gemm_size.py`** — isolates GEMM size
  alone: fixed depth=1 (no chaining), fixed 8192 tokens, only a square
  layer's size sweeps 256→4096 across the occupancy threshold. Run on a
  T4: backward's discount plateaued cleanly at large sizes (implied
  efficiency ~1.0) — looked adoptable, but **reverted** after checking the
  compute/memory bound: those GEMMs are memory-bound (weight matrix
  exceeds this GPU's L2), so `compute_efficiency` never reaches the
  result; matching the measured time as memory-bound would need ~451 GB/s
  on a 320 GB/s GPU, physically impossible. Points at `gemm_traffic`'s
  unvalidated penalty-term heuristic as the real remaining gap for large
  GEMMs — a different model component, not yet investigated
  (`docs/silicon_anchor.md`).

No LayerNorm / GELU / dropout / residual in either — that is
`simulator/`'s v1 scope throughout.

**Timing methodology note:** both scripts bracket the loss reduction
(`.float().square().mean()`) into the *forward* timing window with a
dedicated CUDA event, so it's never silently counted as part of *backward*.
`validate/silicon/profiles/*.json` were collected before this fix (the
3-event/4-event version); the imprecision it corrects is on the order of a
few hundredths of a millisecond — well below the multi-× gaps discussed in
`docs/silicon_anchor.md` and `docs/attention.md`, so it doesn't change any
conclusion there, but a fresh Colab run will report marginally cleaner
numbers than the archived ones.

## How to run — GEMM anchor

**1. On Colab** (or any GPU box): upload `validate/silicon/gemm/profile_step_colab.ipynb` to
[colab.research.google.com](https://colab.research.google.com) (File > Upload
notebook), `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run
all`. It prints a per-layer table, writes `silicon_profile.json`, and
auto-downloads it — no shell, no upload of any other file.

Editing the config cell before running lets you change `DTYPE`
(`bf16`/`fp16`/`fp32`, default `fp16`), `TOKENS` (default 8192), `ITERS` /
`WARMUP`.

**On a Colab T4 or a V100, keep `DTYPE = "fp16"`.** Those GPUs (Turing /
Volta) have no bf16 tensor cores; bf16 silently falls back to a ~20x-slower
SIMT GEMM and the comparison is meaningless. `compare_silicon.py` detects
this (magma/sgemm kernels in `top_ops`) and says so. bf16 is fine on
A100 / L4 / H100.

Prefer the command line? `profile_step.py` is the same script as a plain
CLI (`python validate/silicon/gemm/profile_step.py --out silicon_profile.json --dtype fp16
--tokens 8192 --iters 50 --warmup 15`) for a GPU box you can SSH into
instead of Colab.

**2. Back on your laptop** (no GPU, no torch — just `matplotlib`):

```bash
python validate/silicon/gemm/compare_silicon.py silicon_profile.json
```

It maps the GPU to an accel-sim `Hardware`/`Memory` via `gpu_specs.py`, runs
the closed-form `model_step`, and writes:

- a phase-by-phase measured-vs-predicted table + MAPE to stdout
- `results/validation/silicon.csv`
- `results/plots/silicon_anchor.png`

If the GPU isn't in `gpu_specs.py`, pass it explicitly:

```bash
python validate/silicon/gemm/compare_silicon.py p.json --gpu H100-SXM
# or fully custom:
python validate/silicon/gemm/compare_silicon.py p.json \
    --tflops 312 --l2-mb 40 --hbm-gbs 2039 --clock-ghz 1.41
```

`sample_profile.json` is a hand-written **synthetic** file (flagged as such)
so `compare_silicon.py` can be exercised with no GPU. Its numbers are made
up — do not cite them.

## How to run — attention anchor

Same two steps: upload `validate/silicon/attention/profile_attention_colab.ipynb` to Colab (T4 GPU,
Run all) for `attention_profile.json`, or run `validate/silicon/attention/profile_attention.py
--out attention_profile.json` on the CLI on any GPU box. Then, on your
laptop:

```bash
python validate/silicon/attention/compare_attention.py attention_profile.json
```

Writes `results/validation/attention_silicon.csv` and
`results/plots/attention_silicon.png`. `sample_attention_profile.json` is
the synthetic placeholder, same convention as the GEMM anchor.

The script checks its own fused path: `fused_top_ops` in the JSON should
show a `flash`/`efficient_attention` kernel, not a generic math fallback
(`compare_attention.py` warns if it doesn't) — the same lesson as bf16-on-T4
for the GEMM anchor, just for attention this time.

## How to run — recompute anchor

Same two steps: upload `validate/silicon/recompute/profile_recompute_colab.ipynb` to Colab (T4 GPU,
Run all) for `recompute_profile.json`, or run `validate/silicon/recompute/profile_recompute.py --out
recompute_profile.json` on the CLI. Then, on your laptop:

```bash
python validate/silicon/recompute/compare_recompute.py recompute_profile.json
```

Writes `results/validation/recompute_silicon.csv` and
`results/plots/recompute_silicon.png`. `sample_recompute_profile.json` is
the synthetic placeholder; the real T4 run is archived at
`profiles/t4_fp16_recompute.json` (and is what `run_all.sh` uses by
default). This one uses `torch.utils.checkpoint` directly — PyTorch's real
implementation of the exact idea `recompute=` models — so there's no
kernel-fallback trap to check for, unlike the other two anchors.

## How to run — gradient-accumulation anchor

Same two steps: upload `validate/silicon/gradaccum/profile_gradaccum_colab.ipynb` to Colab (T4 GPU, Run
all) for `gradaccum_profile.json`, or run `validate/silicon/gradaccum/profile_gradaccum.py --out
gradaccum_profile.json` on the CLI. Then, on your laptop:

```bash
python validate/silicon/gradaccum/compare_gradaccum.py gradaccum_profile.json
```

Writes `results/validation/gradaccum_silicon.csv` and
`results/plots/gradaccum_silicon.png` (latency vs fraction, activation
memory vs fraction). `sample_gradaccum_profile.json` is the synthetic
placeholder; the real T4 run is archived at
`profiles/t4_fp16_gradaccum.json` (and is what `run_all.sh` uses by
default).

## How to run — optimizer-state-precision anchor

Same two steps: upload `validate/silicon/optimizer/profile_optimizer_colab.ipynb` to Colab (T4 GPU, Run
all) for `optimizer_profile.json`, or run `validate/silicon/optimizer/profile_optimizer.py --out
optimizer_profile.json` on the CLI. Then, on your laptop:

```bash
python validate/silicon/optimizer/compare_optimizer.py optimizer_profile.json
```

Writes `results/validation/optimizer_silicon.csv` and
`results/plots/optimizer_silicon.png`. `sample_optimizer_profile.json` is
the synthetic placeholder; the real T4 run is archived at
`profiles/t4_fp16_optimizer.json` (and is what `run_all.sh` uses by
default). No forward/backward runs here, so there's no kernel-fallback trap
— but Turing (T4) has no native bf16 arithmetic, so a
real T4 run is itself informative about PyTorch's emulation path.

## How to run — chain-grid recalibration anchor

Same two steps: upload `validate/silicon/chain_grid/profile_chain_grid_colab.ipynb` to Colab (T4 GPU,
Run all) for `chain_grid_profile.json`, or run `validate/silicon/chain_grid/profile_chain_grid.py --out
chain_grid_profile.json` on the CLI. Then, on your laptop:

```bash
python validate/silicon/chain_grid/compare_chain_grid.py chain_grid_profile.json
```

Prints an old-constants-vs-refit-constants before/after table, writes
`results/validation/chain_grid_silicon.csv` and
`results/plots/chain_grid_silicon.png`. `sample_chain_grid_profile.json` is
the synthetic placeholder; the real T4 run is archived at
`profiles/t4_fp16_chain_grid.json` (and is what `run_all.sh` uses by
default). Unlike every other anchor here, this one's own constants
(`compute_efficiency`, `util_tiles`) are **not** applied automatically, and
the real run showed why that caution matters: the refit only helped 2 of
the 4 configs (depth/shape/tokens are confounded across them) — review the
before/after table yourself before ever updating `gpu_specs.py`'s
`_COMPUTE_EFFICIENCY`/`_UTIL_TILES` by hand.

## How to run — chain-depth isolation anchor

Same two steps: upload `validate/silicon/chain_depth/profile_chain_depth_colab.ipynb` to Colab (T4 GPU,
Run all) for `chain_depth_profile.json`, or run `validate/silicon/chain_depth/profile_chain_depth.py
--out chain_depth_profile.json` on the CLI. Then, on your laptop:

```bash
python validate/silicon/chain_depth/compare_chain_depth.py chain_depth_profile.json
```

Fits `measured_backward_ms ~= A*depth + B` by OLS and reports R² — high R²
(>0.9) means a fixed-per-chain-overhead + per-layer-marginal-cost model
explains the isolated-vs-chained gap once depth is isolated from shape and
token count (the chain-grid anchor's confound). The real T4 run got
R²=0.974 and the resulting `backward_util_tiles`/`chain_overhead_*` fit is
wired into `gpu_specs.py`, but **opt-in only** (`GPUSpec.hardware(...,
chained=True)`) — it does not generalize past this one (shape, token
count) pair, confirmed against the other chain-grid configs and the real
recompute/gradaccum anchors. Writes `results/validation/chain_depth_silicon.csv`
and `results/plots/chain_depth_silicon.png`. `sample_chain_depth_profile.json`
is the synthetic placeholder (an exact line by construction, R²=1.0, to
sanity-check the fit code — not a real result); the real T4 run is archived
at `profiles/t4_fp16_chain_depth.json` (and is what `run_all.sh` uses by
default).

## How to run — chain generalization test

Same two steps: upload `validate/silicon/chain_generalize/profile_chain_generalize_colab.ipynb` to Colab (T4
GPU, Run all) for `chain_generalize_profile.json`, or run
`validate/silicon/chain_generalize/profile_chain_generalize.py --out chain_generalize_profile.json` on the
CLI. Then, on your laptop:

```bash
python validate/silicon/chain_generalize/compare_chain_generalize.py chain_generalize_profile.json
```

Reports a fresh depth-sweep OLS fit for the occupancy-safe shape side by
side with 768×768's own fit, a residual-vs-token-count table (does
overhead scale with tokens?), and a repeated-config drift check. Writes
`results/validation/chain_generalize_silicon.csv` and
`results/plots/chain_generalize_silicon.png`. The real T4 run is archived
at `profiles/t4_fp16_chain_generalize.json` (used by `run_all.sh` by
default); `sample_chain_generalize_profile.json` is the synthetic
placeholder.

## How to run — GEMM-size isolation anchor

Same two steps: upload `validate/silicon/gemm_size/profile_gemm_size_colab.ipynb` to Colab (T4 GPU,
Run all) for `gemm_size_profile.json`, or run `validate/silicon/gemm_size/profile_gemm_size.py --out
gemm_size_profile.json` on the CLI. Then, on your laptop:

```bash
python validate/silicon/gemm_size/compare_gemm_size.py gemm_size_profile.json
```

Reports measured vs old-model-predicted forward/backward and the
"discount" ratio per size, alongside the theoretical wgrad occupancy —
**check the compute/memory bound before trusting any trend in this data**
(`simulator.dataflow._backward_seconds` picks `max(compute, memory)`; a
smooth discount curve can be an artifact of the memory-bound penalty term,
not compute efficiency — this is exactly what happened here, see
`docs/silicon_anchor.md`). Writes `results/validation/gemm_size_silicon.csv`
and `results/plots/gemm_size_silicon.png`. The real T4 run is archived at
`profiles/t4_fp16_gemm_size.json`; `sample_gemm_size_profile.json` is the
synthetic placeholder.

## The GPU → accel-sim mapping (`gpu_specs.py`)

| accel-sim knob | GPU quantity |
|---|---|
| `rows × cols × clock_hz` | dense fp16/bf16 tensor-core FLOP/s ÷ 2 |
| array shape (`rows`, `cols`) | square, `round(sqrt(peak_macs / clock))` |
| `sram_bytes` | L2 cache size |
| `dram_bandwidth_bytes_s` | HBM / GDDR bandwidth |
| `dram_efficiency` | 0.75 (pre-registered) |
| `compute_efficiency` | 0.55 (calibrated to one T4, see docs/silicon_anchor.md) |
| `util_tiles` | 128 (calibrated to one T4) |
| `library_gemm` | on — fixed-kernel contraction re-stream |
| `kernel_launch_s` | 8 µs (calibrated to one T4) |
| `mac_bytes` | 2 (bf16/fp16) or 4 (fp32) |
| `vector_flops_per_s` | 4.571e9 elements/s (calibrated to one T4's isolated Adam anchor, see docs/optimizer_state.md) |
| `emulation_penalty` | 1.94x (bf16-on-Turing promote/emulate cost, same anchor) |
| `native_low_precision_bytes` | `{4}` for Turing/Volta (T4, V100) — no native bf16/fp8 ALU; `{4,2,1}` (all) elsewhere |
| `backward_util_tiles` | 1 (T4) — **opt-in only**, `GPUSpec.hardware(..., chained=True)`; not applied by default |
| `chain_overhead_fwd_s` / `chain_overhead_bwd_s` | 0.785ms / 1.361ms (T4) — same opt-in, same caveat |

`compute_efficiency` through `native_low_precision_bytes` are the
GPU-realism knobs, added across the GEMM anchor's first run and the later
optimizer anchor's elementwise-compute fix — see `docs/silicon_anchor.md`
and `docs/optimizer_state.md` for what each fixes and why they're
calibrated to one GPU, not derived. They apply to every anchor by default,
since all of them build a `Hardware`/`Memory` through `gpu_specs.py`. The
last two rows are different: they only apply when a caller explicitly
passes `chained=True` to `GPUSpec.hardware()` (currently only
`compare_chain_depth.py` does), because they were calibrated on one shape/
token-count pair and do not generalize — see `docs/silicon_anchor.md`.
