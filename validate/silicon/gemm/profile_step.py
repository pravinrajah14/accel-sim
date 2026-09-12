"""Silicon anchor -- measure a real per-layer training step on a GPU.

Run this on any CUDA box (a free Colab T4 works).  It times the forward,
backward and Adam-update phases of each GPT-2-block weight GEMM in isolation
-- the exact quantities `simulator.dataflow.training_step` predicts -- and
writes `silicon_profile.json`.  Copy that file back to your laptop and run
`compare_silicon.py` (no GPU needed) to score the model against it.

Only the six weight GEMMs are measured (no attention score matmuls, no
LayerNorm / GELU / dropout / residual) -- that is exactly accel-sim's v1
scope, so this is an apples-to-apples test of the GEMM + traffic + optimizer
model, not of transformer coverage.

    python validate/silicon/profile_step.py --out silicon_profile.json

Requires: torch (with CUDA).  No other repo code is imported, so you can
scp / upload just this one file.
"""

import argparse
import json
import platform
import statistics
import sys

# --- GPT-2 small block, must match simulator/workloads.py -------------------
D_MODEL = 768
D_FF = 3072
DEFAULT_TOKENS = 8 * 1024        # batch 8 x seq 1024

SHAPES = [                        # (name, K in-features, N out-features)
    ("q_proj",   D_MODEL, D_MODEL),
    ("k_proj",   D_MODEL, D_MODEL),
    ("v_proj",   D_MODEL, D_MODEL),
    ("attn_out", D_MODEL, D_MODEL),
    ("mlp_up",   D_MODEL, D_FF),
    ("mlp_down", D_FF,    D_MODEL),
]

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _bench_layer(torch, nn, device, dtype, M, K, N, iters, warmup):
    lin = nn.Linear(K, N, bias=False).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-4)          # fp32 m,v state
    # x requires grad: a real hidden layer's input is upstream, so backward
    # computes BOTH the input gradient (dgrad) and the weight gradient (wgrad).
    x = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)

    fwd, bwd, optt = [], [], []
    for i in range(warmup + iters):
        # 5 events, not 4: the loss reduction (.float().square().mean()) is
        # itself a few small kernels -- bracket it into "forward" so it does
        # not get silently counted as part of "backward" (a real, if small,
        # measurement imprecision this fixes).
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


def _top_ops(torch, device, dtype, M):
    """One profiled step over all six layers, for the kernel-level table."""
    import torch.nn as nn
    from torch.profiler import profile, ProfilerActivity

    lins = [nn.Linear(K, N, bias=False).to(device=device, dtype=dtype)
            for _, K, N in SHAPES]
    xs = [torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
          for _, K, _ in SHAPES]
    opt = torch.optim.Adam([p for l in lins for p in l.parameters()], lr=1e-4)

    for _ in range(5):
        loss = sum(l(x).float().square().mean() for l, x in zip(lins, xs))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        loss = sum(l(x).float().square().mean() for l, x in zip(lins, xs))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    rows = []
    for e in prof.key_averages():
        cuda_us = getattr(e, "self_device_time_total", 0) or \
            getattr(e, "self_cuda_time_total", 0)
        if cuda_us > 0:
            rows.append({"op": e.key, "cuda_ms": cuda_us / 1e3,
                         "count": e.count})
    rows.sort(key=lambda r: -r["cuda_ms"])
    return rows[:15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="silicon_profile.json")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--no-trace", action="store_true",
                    help="skip the torch.profiler kernel table")
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
    print(f"GPU: {gpu}   dtype={args.dtype}   tokens={M}   "
          f"iters={args.iters} (+{args.warmup} warmup)")
    if args.dtype == "bf16" and any(k in gpu for k in ("T4", "V100", "P100",
                                                       "1080", "2080")):
        print(f"  WARNING: {gpu} has no bf16 tensor cores -- bf16 will fall back "
              f"to a ~20x-slower kernel.\n  Re-run with --dtype fp16.")

    per_layer = {}
    agg = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    for name, K, N in SHAPES:
        r = _bench_layer(torch, nn, device, dtype, M, K, N,
                         args.iters, args.warmup)
        per_layer[name] = {"K": K, "N": N, **r}
        for ph in agg:
            agg[ph] += r[ph]["mean_ms"]
        print(f"  {name:9s}  fwd {r['forward']['mean_ms']:7.3f}  "
              f"bwd {r['backward']['mean_ms']:7.3f}  "
              f"opt {r['optimizer']['mean_ms']:7.3f} ms")

    agg["step"] = sum(agg.values())
    print(f"  {'TOTAL':9s}  fwd {agg['forward']:7.3f}  bwd {agg['backward']:7.3f}"
          f"  opt {agg['optimizer']:7.3f}  step {agg['step']:7.3f} ms")

    out = {
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype],
        "mac_bytes": 4 if args.dtype == "fp32" else 2,
        "tokens": M, "d_model": D_MODEL, "d_ff": D_FF,
        "iters": args.iters, "warmup": args.warmup,
        "phase_ms": agg,
        "per_layer": per_layer,
        "host": platform.platform(),
    }
    if not args.no_trace:
        try:
            out["top_ops"] = _top_ops(torch, device, dtype, M)
        except Exception as e:                       # noqa: BLE001
            print(f"  (kernel trace skipped: {e})")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_silicon.py {args.out}")


if __name__ == "__main__":
    main()
