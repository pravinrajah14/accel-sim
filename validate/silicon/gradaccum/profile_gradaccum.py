"""Silicon anchor for gradient accumulation (microbatching).

The model's claim (`docs/microbatching.md`): splitting M tokens into n
chunks, accumulating dW, and running the optimizer once trades latency for
peak stored-activation memory -- each halving of the microbatch roughly
halves activation memory for a modest latency cost. This measures both
sides of that trade for real: wall-clock step time via CUDA events, and
peak activation memory via `torch.cuda.max_memory_allocated()` -- on the
same 4-layer chain (GPT-2's own shapes) the recompute anchor uses.

    python validate/silicon/profile_gradaccum.py --out gradaccum_profile.json

Then locally:

    python validate/silicon/compare_gradaccum.py gradaccum_profile.json
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
CHAIN_DIMS = [(768, 768), (768, 768), (768, 3072), (3072, 768)]
FRACTIONS = [1, 2, 4, 8, 16]


def _bench(torch, nn, device, dtype, M, frac, iters, warmup):
    layers = [nn.Linear(k, n, bias=False).to(device=device, dtype=dtype)
             for k, n in CHAIN_DIMS]
    params = [p for l in layers for p in l.parameters()]
    x_full = torch.randn(M, CHAIN_DIMS[0][0], device=device, dtype=dtype)

    mb = max(1, M // frac)
    n_chunks = -(-M // mb)  # ceil

    def one_step():
        for p in params:
            p.grad = None
        for i in range(n_chunks):
            chunk = x_full[i * mb:(i + 1) * mb]
            if chunk.shape[0] == 0:
                continue
            out = chunk
            for l in layers:
                out = l(out)
            loss = out.float().square().mean() / n_chunks
            loss.backward()
        # optimizer step omitted deliberately -- optimizer_seconds is
        # already anchored separately (profile_step.py); this isolates
        # exactly what the microbatch= lever changes: forward+backward
        # cost and peak activation memory.

    times = []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        one_step()
        ev1.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(ev0.elapsed_time(ev1))

    torch.cuda.reset_peak_memory_stats(device)
    one_step()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "peak_activation_mb": peak_mb,
        "microbatch_tokens": mb,
        "n_chunks": n_chunks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gradaccum_profile.json")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    ap.add_argument("--tokens", type=int, default=8 * 1024)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("torch is required: pip install torch")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on a GPU box (Colab T4 is fine)")

    device = torch.device("cuda")
    dtype = getattr(torch, _DTYPES[args.dtype])
    M = args.tokens
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   dtype={args.dtype}   tokens={M}   chain={CHAIN_DIMS}   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    configs = {}
    for frac in FRACTIONS:
        r = _bench(torch, nn, device, dtype, M, frac, args.iters, args.warmup)
        configs[str(frac)] = r
        print(f"  fraction {frac:3d}  (mb={r['microbatch_tokens']:5d}, n={r['n_chunks']:2d})  "
              f"step {r['mean_ms']:8.3f} ms   peak activations {r['peak_activation_mb']:8.1f} MB")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype], "tokens": M, "chain_dims": CHAIN_DIMS,
        "iters": args.iters, "warmup": args.warmup,
        "configs": configs, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_gradaccum.py {args.out}")


if __name__ == "__main__":
    main()
