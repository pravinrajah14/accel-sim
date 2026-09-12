#!/usr/bin/env bash
# Live demo / full reproduction of every result in the writeup.
# Regenerates every plot + validation number and prints a headline summary.
set -euo pipefail
cd "$(dirname "$0")"

# ---- narration helpers ------------------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; CYAN=""; GREEN=""; YELLOW=""; RESET=""
fi
step() { printf "\n${BOLD}${CYAN}== %s ==${RESET}\n" "$1"; }
ok()   { printf "${GREEN}%s${RESET}\n" "$1"; }
note() { printf "${DIM}%s${RESET}\n" "$1"; }
t0=$(date +%s)

printf "${BOLD}accel-sim${RESET} -- closed-form training-step model for a systolic-array accelerator\n"
note "forward + backward + Adam optimizer, validated against Timeloop and a real GPU"

python3 -m pip install -q -r requirements.txt

step "1. unit sanity checks"
python3 -m pytest -q tests/ 2>/dev/null || python3 tests/test_sanity.py

step "2. bottleneck-shift sweeps (array size / SRAM / bandwidth) -> results/plots/"
python3 experiments/sweep_array_size.py
python3 experiments/sweep_sram.py
python3 experiments/sweep_bandwidth.py
python3 experiments/dataflow_choice.py

step "3. systems levers -> results/plots/"
python3 experiments/optimizer_cost.py
python3 experiments/recomputation.py
python3 experiments/microbatching.py

step "4. closed-form autotuner (40-config search, ~1 ms)"
python3 experiments/autotune.py

step "4b. attention: GEMMs (linear) vs attention (quadratic) in sequence length"
python3 experiments/attention_scaling.py

step "4b2. attention fusion: naive vs flash-attention-style, swept over sequence length"
python3 experiments/attention_fusion.py

step "4c. torch.fx model import (arbitrary shapes, not hand-typed)"
python3 experiments/import_model_demo.py

step "5. Timeloop cross-check (cycles + DRAM traffic vs a mapping-search tool)"
if [ "${1:-}" = "--full" ]; then
  python3 validate/gen_timeloop_configs.py --arrays 64 128
  if docker info >/dev/null 2>&1; then
    bash validate/run_timeloop.sh
    python3 validate/compare.py
  else
    note "  skipped: Docker daemon not running."
  fi
else
  note "  skipped by default -- needs Docker + ~5 min (15 configs, each its own container)."
  note "  last recorded result (results/validation/validation.csv):"
  note "    ws-fwd cycles 1.2% MAPE | os-wgrad cycles 0.2% MAPE | DRAM traffic 0.0% (exact), n=15"
  note "  to reproduce live: ./run_all.sh --full   (needs Docker Desktop running)"
fi

step "6. silicon anchor -- real Tesla T4 GPU (fp16), not a synthetic sample"
python3 validate/silicon/gemm/compare_silicon.py validate/silicon/profiles/t4_fp16.json
note "  (re-profile any GPU: upload validate/silicon/gemm/profile_step_colab.ipynb to Colab)"

step "7. attention silicon anchor -- real Tesla T4 GPU (fp16), not a synthetic sample"
python3 validate/silicon/attention/compare_attention.py validate/silicon/profiles/t4_fp16_attention.json
note "  (re-profile any GPU: upload validate/silicon/attention/profile_attention_colab.ipynb to Colab)"

step "8. recompute silicon anchor -- real Tesla T4 GPU (fp16), not a synthetic sample"
python3 validate/silicon/recompute/compare_recompute.py validate/silicon/profiles/t4_fp16_recompute.json
note "  (re-profile any GPU: upload validate/silicon/recompute/profile_recompute_colab.ipynb to Colab)"

step "9. gradient-accumulation silicon anchor -- real Tesla T4 GPU (fp16), not a synthetic sample"
python3 validate/silicon/gradaccum/compare_gradaccum.py validate/silicon/profiles/t4_fp16_gradaccum.json
note "  (re-profile any GPU: upload validate/silicon/gradaccum/profile_gradaccum_colab.ipynb to Colab)"

step "10. optimizer-state-precision silicon anchor -- real Tesla T4 GPU (fp16), not a synthetic sample"
python3 validate/silicon/optimizer/compare_optimizer.py validate/silicon/profiles/t4_fp16_optimizer.json
note "  (re-profile any GPU: upload validate/silicon/optimizer/profile_optimizer_colab.ipynb to Colab)"

step "11. chained-training-step recalibration anchor -- real Tesla T4 GPU (fp16), not synthetic"
python3 validate/silicon/chain_grid/compare_chain_grid.py validate/silicon/profiles/t4_fp16_chain_grid.json
note "  (re-profile any GPU: upload validate/silicon/chain_grid/profile_chain_grid_colab.ipynb to Colab)"
note "  4 configs confirm isolated-vs-chained is real, but the discount factor does not move"
note "  monotonically with depth across them -- shape/depth/token-count are confounded here."

step "12. chain-depth isolation anchor -- real Tesla T4 GPU (fp16), not synthetic"
python3 validate/silicon/chain_depth/compare_chain_depth.py validate/silicon/profiles/t4_fp16_chain_depth.json
note "  (re-profile any GPU: upload validate/silicon/chain_depth/profile_chain_depth_colab.ipynb to Colab)"
note "  R^2=0.97 backward, 0.96 forward -- a fixed-per-chain-overhead + per-layer-marginal-cost"
note "  model DOES explain this one shape/token-count cleanly (backward_util_tiles=1 relieves"
note "  the isolated-calibrated occupancy penalty). But it does NOT generalize: tried against"
note "  the other 3 chain-grid configs and the real recompute/gradaccum anchors' own shape mix,"
note "  it barely moved their error (or made it worse) -- opt-in via chained=True, not default."

step "13. chain generalization test -- real Tesla T4 GPU (fp16), not synthetic"
python3 validate/silicon/chain_generalize/compare_chain_generalize.py validate/silicon/profiles/t4_fp16_chain_generalize.json
note "  (re-profile any GPU: upload validate/silicon/chain_generalize/profile_chain_generalize_colab.ipynb to Colab)"
note "  R^2=1.000 on an occupancy-safe shape confirms the fixed-overhead structural form again,"
note "  but the fitted constants are shape-size-specific (3.64/layer + 2.62ms vs 0.49/layer +"
note "  1.36ms) -- and depth=1 alone is within 4% of the compute_efficiency=1.0 ideal, revealing"
note "  a THIRD entangled effect: raw efficiency likely rises with GEMM size, never modeled."

step "14. GEMM-size isolation anchor -- real Tesla T4 GPU (fp16), not synthetic"
python3 validate/silicon/gemm_size/compare_gemm_size.py validate/silicon/profiles/t4_fp16_gemm_size.json
note "  (re-profile any GPU: upload validate/silicon/gemm_size/profile_gemm_size_colab.ipynb to Colab)"
note "  backward's discount plateaus cleanly at large sizes (implied ce~1.0) but REVERTED after"
note "  checking the bound: those GEMMs are memory-bound, so compute_efficiency never reaches"
note "  the result -- matching it as memory-bound would need 451 GB/s on a 320 GB/s GPU. Real"
note "  gap: gemm_traffic's unvalidated penalty-term heuristic, a different model component."

t1=$(date +%s)
printf "\n${BOLD}${YELLOW}== summary (%ds) ==${RESET}\n" "$((t1 - t0))"
cat <<'EOF'
  Timeloop cross-check   cycles 1.2% MAPE / 0.2% MAPE | DRAM traffic 0.0% (exact)   n=15
  Silicon anchor (T4)    step latency within 7%  (forward 0.3%, backward 14%, opt 59% --
                         opt got worse in isolation after the elementwise-compute fix below,
                         but partly cancels backward's error in the step-level sum)
  Bottleneck shift       array 128->192 PEs | BW 50->100 GB/s | SRAM 4->8 MiB
  Systems levers         fp8 optimizer state 3x | recompute crossover ~250 GB/s
  Autotuner              exhaustive 40-config search in ~1 ms
  Attention              overtakes GEMMs at 256->512 tokens; 74% of the step at seq=1024
  Attention fusion       saves 45% (seq 128) -> 98% (seq 8192); activations cut up to 2048x
  Attention anchor (T4)  naive off by 1.7x (like the GEMMs); fused backward off by 6.8x even
                         after modeling the real recompute cost (was 8.3x with the flat 2x rule)
  Recompute anchor (T4)  forward off by 1.2x (GEMM-like); backward off by 2.1x the OTHER
                         direction -- model over-predicts here, calibration doesn't transfer
  Gradaccum anchor (T4)  step latency still off by up to 1.6x (same calibration-transfer gap
                         as recompute); activation memory FIXED with a fitted floor + scale term
                         -- ratio now ~1.0 at every fraction, was 3-12x low
  Optimizer anchor (T4)  FIXED: added an elementwise-compute term + a native-dtype-support
                         gate, calibrated on this anchor -- model now predicts bf16 state is
                         44.9% slower (measured: 33.3% slower), sign now correct, not +20% faster
  Chain-grid anchor (T4) confirms chained IS faster than the isolated sum (was: hypothesis);
                         but discount does not move monotonically with depth across 4 configs --
                         shape/depth/tokens confounded, a 2-constant refit does not generalize
  Chain-depth anchor (T4) R^2=0.97 backward: depth cleanly explained ONCE shape+tokens are
                         fixed (backward_util_tiles=1 relieves an over-harsh occupancy penalty).
                         Does NOT generalize to other shapes/token-counts -- opt-in, not default
  Chain-generalize (T4)  R^2=1.000 on an occupancy-safe shape -- structural form holds again,
                         but constants are shape-size-specific. depth=1 alone lands within 4%
                         of the ce=1.0 ideal -- points to a THIRD effect: efficiency-vs-GEMM-size
  GEMM-size anchor (T4)  backward's discount plateaus cleanly at large sizes (implied ce~1.0)
                         but REVERTED -- those GEMMs are memory-bound, so compute_efficiency
                         never reaches the result (would need 451 GB/s on a 320 GB/s GPU).
                         Real gap: gemm_traffic's unvalidated penalty heuristic, not efficiency.
                         5-round investigation ends here; docs/silicon_anchor.md has the trail
EOF
note "full numbers + model-revision history: results/RESULTS.md"
