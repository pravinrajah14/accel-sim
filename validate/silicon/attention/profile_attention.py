"""Silicon anchor for attention -- naive vs fused, on a real GPU.

`simulator/attention.py` models two attention kernels: `fused=False` (naive
-- explicit QK^T -> softmax -> (softmax)@V, materializing the full
(B,H,S,S) score tensor) and `fused=True` (flash-attention-style -- never
materializes it). This measures both, for real, on CUDA:

  naive : hand-written QK^T -> softmax -> (softmax)@V (matches fused=False)
  fused : torch.nn.functional.scaled_dot_product_attention, which PyTorch
          dispatches to a flash-attention / memory-efficient CUDA kernel
          that never materializes the full score tensor (matches fused=True)

Neither the GEMM-only silicon anchor (`profile_step.py`) nor the abstract
model's attention arithmetic (`docs/attention.md`) has a hardware anchor
yet -- this is that anchor.

    python validate/silicon/profile_attention.py --out attention_profile.json

Then locally, no GPU:

    python validate/silicon/compare_attention.py attention_profile.json

Requires torch (CUDA). On a Colab T4 or a V100 use --dtype fp16 (no bf16
tensor cores there -- same lesson as profile_step.py).
"""

import argparse
import json
import math
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _bench(torch, F, device, dtype, B, S, H, Dh, iters, warmup, fused):
    q = torch.randn(B, H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    scale = 1.0 / math.sqrt(Dh)

    def naive_forward():
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        return attn @ v

    def fused_forward():
        return F.scaled_dot_product_attention(q, k, v)

    forward_fn = fused_forward if fused else naive_forward

    fwd, bwd = [], []
    for i in range(warmup + iters):
        for t in (q, k, v):
            t.grad = None
        # 4 events, not 3: the loss reduction (.float().square().mean()) is
        # itself a few small kernels -- bracket it into "forward" so it does
        # not get silently counted as part of "backward" (a real, if small,
        # measurement imprecision this fixes).
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        out = forward_fn()
        ev[1].record()
        loss = out.float().square().mean()
        ev[2].record()
        loss.backward()
        ev[3].record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(ev[0].elapsed_time(ev[2]))
            bwd.append(ev[2].elapsed_time(ev[3]))

    def stat(v):
        return {"mean_ms": statistics.fmean(v),
                "std_ms": statistics.pstdev(v) if len(v) > 1 else 0.0}
    return {"forward": stat(fwd), "backward": stat(bwd)}


def _top_ops_fused(torch, F, device, dtype, B, S, H, Dh):
    """One profiled fused call, to confirm a real fused kernel ran (not a
    silent fallback to the math/generic backend)."""
    from torch.profiler import profile, ProfilerActivity
    q = torch.randn(B, H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    for _ in range(5):
        F.scaled_dot_product_attention(q, k, v).sum().backward()
        for t in (q, k, v):
            t.grad = None
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(q, k, v).sum().backward()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        cuda_us = getattr(e, "self_device_time_total", 0) or \
            getattr(e, "self_cuda_time_total", 0)
        if cuda_us > 0:
            rows.append({"op": e.key, "cuda_ms": cuda_us / 1e3, "count": e.count})
    rows.sort(key=lambda r: -r["cuda_ms"])
    return rows[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="attention_profile.json")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--d-head", type=int, default=64)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        sys.exit("torch is required: pip install torch")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on a GPU box (Colab T4 is fine)")

    device = torch.device("cuda")
    dtype = getattr(torch, _DTYPES[args.dtype])
    B, S, H, Dh = args.batch, args.seq, args.heads, args.d_head
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   dtype={args.dtype}   B={B} S={S} H={H} Dh={Dh}   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    naive = _bench(torch, F, device, dtype, B, S, H, Dh, args.iters, args.warmup, fused=False)
    print(f"  naive  fwd {naive['forward']['mean_ms']:8.3f}  bwd {naive['backward']['mean_ms']:8.3f} ms")
    fused = _bench(torch, F, device, dtype, B, S, H, Dh, args.iters, args.warmup, fused=True)
    print(f"  fused  fwd {fused['forward']['mean_ms']:8.3f}  bwd {fused['backward']['mean_ms']:8.3f} ms")
    saved = (naive["forward"]["mean_ms"] + naive["backward"]["mean_ms"] -
            fused["forward"]["mean_ms"] - fused["backward"]["mean_ms"])
    total_naive = naive["forward"]["mean_ms"] + naive["backward"]["mean_ms"]
    print(f"  measured fusion saving: {saved / total_naive * 100:.1f}%")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype],
        "batch": B, "seq": S, "n_heads": H, "d_head": Dh,
        "iters": args.iters, "warmup": args.warmup,
        "naive": naive, "fused": fused,
        "host": platform.platform(),
    }
    try:
        out["fused_top_ops"] = _top_ops_fused(torch, F, device, dtype, B, S, H, Dh)
    except Exception as e:                                        # noqa: BLE001
        print(f"  (fused kernel trace skipped: {e})")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_attention.py {args.out}")


if __name__ == "__main__":
    main()
