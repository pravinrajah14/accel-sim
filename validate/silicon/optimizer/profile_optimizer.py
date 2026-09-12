"""Silicon anchor for optimizer-state precision.

The model's claim (`docs/optimizer_state.md`): storing Adam's running
averages m, v in a lower-precision dtype (bf16 instead of fp32) halves the
DRAM traffic of the optimizer update. This isolates exactly that update --
no forward/backward, just the Adam arithmetic -- for the same four layer
sizes the other anchors use (GPT-2's own shapes), at fp32 vs bf16 state.

    python validate/silicon/profile_optimizer.py --out optimizer_profile.json

Then locally:

    python validate/silicon/compare_optimizer.py optimizer_profile.json

Note: Turing (T4, sm75) has no native bf16 arithmetic (that arrived with
Ampere) -- PyTorch may promote/emulate bf16 elementwise ops there. That is
itself useful silicon-anchor information, not a reason to skip the run.
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
CHAIN_DIMS = [(768, 768), (768, 768), (768, 3072), (3072, 768)]


def _bench_adam_step(torch, device, n_params, state_dtype, iters, warmup):
    param = torch.randn(n_params, device=device, dtype=torch.float32)
    grad = torch.randn(n_params, device=device, dtype=torch.float32)
    m = torch.zeros(n_params, device=device, dtype=state_dtype)
    v = torch.zeros(n_params, device=device, dtype=state_dtype)
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 1e-4

    def step():
        nonlocal m, v, param
        m = (beta1 * m.float() + (1 - beta1) * grad).to(state_dtype)
        v = (beta2 * v.float() + (1 - beta2) * grad * grad).to(state_dtype)
        param = param - lr * m.float() / (v.float().sqrt() + eps)

    times = []
    for i in range(warmup + iters):
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        step()
        ev1.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(ev0.elapsed_time(ev1))

    return {"mean_ms": statistics.fmean(times),
            "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="optimizer_profile.json")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required: pip install torch")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on a GPU box (Colab T4 is fine)")

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    layer_sizes = [k * n for k, n in CHAIN_DIMS]
    print(f"GPU: {gpu}   layer param counts={layer_sizes}   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    results = {}
    for state_name in ("fp32", "bf16"):
        state_dtype = getattr(torch, _DTYPES[state_name])
        per_layer = []
        for n_params in layer_sizes:
            r = _bench_adam_step(torch, device, n_params, state_dtype,
                                 args.iters, args.warmup)
            per_layer.append(r)
        total_ms = sum(r["mean_ms"] for r in per_layer)
        results[state_name] = {"per_layer": per_layer, "total_ms": total_ms}
        print(f"  {state_name} state   total {total_ms:8.4f} ms   "
              f"per-layer {[round(r['mean_ms'], 4) for r in per_layer]}")

    saved = (results["fp32"]["total_ms"] - results["bf16"]["total_ms"]) / results["fp32"]["total_ms"] * 100
    print(f"\n  measured saving from bf16 state: {saved:+.1f}%")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "layer_sizes": layer_sizes, "chain_dims": CHAIN_DIMS,
        "iters": args.iters, "warmup": args.warmup,
        "results": results, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_optimizer.py {args.out}")


if __name__ == "__main__":
    main()
