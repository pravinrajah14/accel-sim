"""Silicon anchor: isolate GEMM size's effect on compute efficiency, at
fixed depth=1 (no chaining at all) and fixed token count (8192).

The chain-generalize anchor found that a single large, well-occupied
2048x2048 GEMM (depth=1) runs close to the model's `compute_efficiency=1.0`
ideal (within 4.4%), while GPT-2's own 768x768 shape needed the full 0.55
discount plus an occupancy fix -- suggesting `compute_efficiency` itself
should rise with GEMM size, a real GPU phenomenon (larger GEMMs sustain a
higher fraction of peak throughput) this model has never captured. But
that comparison mixed size with occupancy (768x768 IS occupancy-penalized
under util_tiles=128, 2048x2048 is not).

This script isolates SIZE alone: depth is fixed at 1 (no chain/depth
confound whatsoever -- a "chain" of 1 layer is just a single GEMM,
directly comparable to `profile_step.py`'s own isolated methodology) and
M is fixed at 8192 (GPT-2's own token count), while the layer's square
shape (K=N) sweeps from heavily occupancy-penalized (256) through the
occupancy threshold (util_tiles=128, crossed around 1536-2048) up to well
past it (4096) -- so the resulting curve shows both the size effect AND
where it interacts with the existing occupancy penalty, at a single
GPU and a single fixed depth/M.

    python validate/silicon/profile_gemm_size.py --out gemm_size_profile.json

Then locally:

    python validate/silicon/compare_gemm_size.py gemm_size_profile.json
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

TOKENS = 8 * 1024
SIZES = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]   # square (K=N) layers


def _bench(torch, nn, device, dtype, size, M, iters, warmup):
    lin = nn.Linear(size, size, bias=False).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-4)
    x = torch.randn(M, size, device=device, dtype=dtype, requires_grad=True)

    fwd, bwd, optt = [], [], []
    for i in range(warmup + iters):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        ev[0].record()
        y = lin(x)
        ev[1].record()
        loss = y.float().square().mean()
        ev[2].record()
        loss.backward()
        ev[3].record()
        opt.step()
        opt.zero_grad(set_to_none=True)
        x.grad = None
        ev[4].record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(ev[0].elapsed_time(ev[2]))
            bwd.append(ev[2].elapsed_time(ev[3]))
            optt.append(ev[3].elapsed_time(ev[4]))

    def stat(v):
        return {"mean_ms": statistics.fmean(v),
                "std_ms": statistics.pstdev(v) if len(v) > 1 else 0.0}
    return {"forward": stat(fwd), "backward": stat(bwd), "optimizer": stat(optt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gemm_size_profile.json")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
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
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   dtype={args.dtype}   tokens={TOKENS}   depth=1   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    results = {}
    for size in SIZES:
        r = _bench(torch, nn, device, dtype, size, TOKENS, args.iters, args.warmup)
        results[str(size)] = r
        print(f"  size {size:5d}  fwd {r['forward']['mean_ms']:8.3f}  "
              f"bwd {r['backward']['mean_ms']:8.3f}  "
              f"opt {r['optimizer']['mean_ms']:6.3f} ms")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype], "tokens": TOKENS, "sizes": SIZES,
        "iters": args.iters, "warmup": args.warmup,
        "results": results, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_gemm_size.py {args.out}")


if __name__ == "__main__":
    main()
