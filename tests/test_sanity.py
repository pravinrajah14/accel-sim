"""Phase-1/2 sanity checks. Runs under pytest or as a plain script."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.compute import Hardware, matmul_cycles, ideal_mac_cycles
from simulator.memory import Memory, roofline
from simulator.dataflow import (Layer, training_step, optimizer_state_bytes,
                                optimizer_seconds, model_step,
                                Optimizer, ADAM_FP32)
from simulator.workloads import GPT2_BLOCK, GPT2_BLOCK_FULL, ATTENTION
from simulator.attention import Attention, attention_step


def test_matmul_matches_mac_formula_when_compute_bound():
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    M, N, K = 8192, 3072, 768
    total = matmul_cycles(M, N, K, hw)
    ideal = ideal_mac_cycles(M, N, K, hw)
    assert total >= ideal
    assert (total - ideal) / ideal < 0.02  # <2% overhead for a big GEMM


def test_smaller_array_costs_more_cycles():
    M, N, K = 4096, 1024, 1024
    big = matmul_cycles(M, N, K, Hardware(256, 256, 1e9))
    small = matmul_cycles(M, N, K, Hardware(64, 64, 1e9))
    assert small > 4 * big * 0.9  # ~16x fewer PEs -> ~16x more cycles


def test_roofline_flips_with_bandwidth():
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    M, N, K = 8192, 768, 768
    fast = roofline(M, N, K, hw, Memory(2 << 20, 2000e9))
    slow = roofline(M, N, K, hw, Memory(2 << 20, 20e9))
    assert fast.bound == "compute"
    assert slow.bound == "memory"


def test_backward_is_about_2x_forward_flops():
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)   # compute-bound regime
    mem = Memory(64 << 20, 4000e9)
    layer = Layer("l", M=8192, K=1024, N=1024)
    s = training_step(layer, hw, mem)
    ratio = s.backward_s / s.forward_s
    assert 1.7 < ratio < 2.3


def test_adam_state_doubles_param_memory():
    layer = Layer("l", M=1, K=768, N=3072)
    assert optimizer_state_bytes(layer) == 2 * layer.param_count * 4
    assert optimizer_state_bytes(layer, Optimizer(state_bytes=1)) == \
        2 * layer.param_count


def test_traffic_compulsory_is_tensor_sum_when_it_fits():
    from simulator.memory import gemm_traffic
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    mem = Memory(64 << 20, 100e9)                     # everything fits
    M, N, K = 1024, 768, 768
    b = hw.mac_bytes
    t = gemm_traffic(M, N, K, hw, mem)
    assert t.penalty_bytes == 0
    assert t.total_bytes == (M * K + K * N + M * N) * b


def test_traffic_penalty_grows_as_sram_shrinks():
    from simulator.memory import gemm_traffic
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    M, N, K = 8192, 3072, 768
    big = gemm_traffic(M, N, K, hw, Memory(64 << 20, 100e9)).total_bytes
    small = gemm_traffic(M, N, K, hw, Memory(1 << 20, 100e9)).total_bytes
    assert small > big
    # a wgrad-shaped GEMM (small output) keeps compulsory traffic at modest SRAM
    wg = gemm_traffic(768, 3072, 8192, hw, Memory(8 << 20, 100e9))
    assert wg.penalty_bytes == 0


def test_microbatching_cuts_activations_linearly_costs_latency():
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(32 << 20, 100e9)
    full, pf = model_step(GPT2_BLOCK, hw, mem)
    half, ph = model_step(GPT2_BLOCK, hw, mem, microbatch=4096)  # n=2
    act_full = sum(p.activation_bytes for p in pf)
    act_half = sum(p.activation_bytes for p in ph)
    assert abs(act_half / act_full - 0.5) < 0.05        # ~half the activations
    assert full < half < 1.3 * full                     # small latency cost


def test_recompute_helps_memory_bound_hurts_compute_bound():
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    layer = Layer("mlp", M=8192, K=768, N=3072)
    slow = Memory(8 << 20, 25e9)     # memory-bound
    fast = Memory(8 << 20, 4000e9)   # compute-bound
    assert training_step(layer, hw, slow, recompute=True).total_s < \
        training_step(layer, hw, slow, recompute=False).total_s
    assert training_step(layer, hw, fast, recompute=True).total_s > \
        training_step(layer, hw, fast, recompute=False).total_s


def test_lower_precision_optimizer_is_cheaper():
    layer = Layer("l", M=1, K=1024, N=4096)
    mem = Memory(1 << 20, 100e9)                     # m,v don't fit -> streamed
    fp32 = optimizer_seconds(layer, mem, Optimizer(4))
    fp8 = optimizer_seconds(layer, mem, Optimizer(1))
    assert fp8 < fp32
    # resident + fits -> only master weights streamed, cheaper still
    big = Memory(64 << 20, 100e9)
    assert optimizer_seconds(layer, big, Optimizer(1, True)) < fp8


def test_efficiency_knobs_default_to_the_ideal_model():
    """compute_efficiency / util_tiles / kernel_launch_s must not move any
    abstract-model number at their defaults."""
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)
    assert hw.compute_efficiency == 1.0 and hw.util_tiles == 1
    from simulator.compute import matmul_seconds
    ideal = matmul_cycles(4096, 4096, 4096, hw) / hw.clock_hz
    assert abs(matmul_seconds(4096, 4096, 4096, hw) - ideal) < 1e-12
    gpu = Hardware(128, 128, 1e9, compute_efficiency=0.5, util_tiles=64,
                   kernel_launch_s=6e-6)
    assert matmul_seconds(512, 512, 4096, gpu) > matmul_seconds(512, 512, 4096,
                                                                Hardware(128, 128, 1e9))


def test_elementwise_seconds_zero_for_abstract_model():
    from simulator.compute import elementwise_seconds
    hw = Hardware(rows=128, cols=128, clock_hz=1e9)   # vector_flops_per_s defaults to 0.0
    assert elementwise_seconds(10_000_000, hw) == 0.0
    assert elementwise_seconds(10_000_000, hw, native=False) == 0.0


def test_elementwise_seconds_emulation_penalty():
    from simulator.compute import elementwise_seconds
    hw = Hardware(rows=128, cols=128, clock_hz=1e9,
                  vector_flops_per_s=1e9, emulation_penalty=2.0)
    native_s = elementwise_seconds(1_000_000, hw, native=True)
    emulated_s = elementwise_seconds(1_000_000, hw, native=False)
    assert native_s > 0
    assert abs(emulated_s - 2.0 * native_s) < 1e-12


def test_library_gemm_penalty_is_off_by_default_on_by_flag():
    from simulator.memory import gemm_traffic
    hw = Hardware(128, 128, 1e9)
    M, N, K = 768, 3072, 8192          # skinny wgrad shape, big contraction
    opt_sched = gemm_traffic(M, N, K, hw, Memory(4 << 20, 100e9))
    lib_sched = gemm_traffic(M, N, K, hw, Memory(4 << 20, 100e9, library_gemm=True))
    assert lib_sched.total_bytes > 3 * opt_sched.total_bytes
    # and it stays ~0 when everything fits
    big = gemm_traffic(768, 768, 768, hw, Memory(64 << 20, 100e9, library_gemm=True))
    assert big.penalty_bytes == 0


def test_import_model_recovers_shapes_from_a_traced_torch_module():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP test_import_model_recovers_shapes_from_a_traced_torch_module (no torch)")
        return
    from simulator.import_model import layers_from_torch

    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(768, 768, bias=False)
            self.mlp_up = nn.Linear(768, 3072, bias=False)
            self.mlp_down = nn.Linear(3072, 768, bias=False)

        def forward(self, x):
            q = self.q_proj(x)
            return self.mlp_down(torch.relu(self.mlp_up(q)))

    model = ToyBlock()
    layers = layers_from_torch(model, torch.randn(8, 1024, 768))  # (batch, seq, d_model)
    by_name = {l.name: l for l in layers}
    assert by_name["q_proj"].M == 8192
    assert (by_name["q_proj"].K, by_name["q_proj"].N) == (768, 768)
    assert (by_name["mlp_up"].K, by_name["mlp_up"].N) == (768, 3072)
    assert (by_name["mlp_down"].K, by_name["mlp_down"].N) == (3072, 768)

    only = layers_from_torch(model, torch.randn(8, 1024, 768), only=["mlp"])
    assert {l.name for l in only} == {"mlp_up", "mlp_down"}


def test_gpu_spec_maps_to_throughput_equivalent_array():
    from validate.silicon.gpu_specs import lookup
    s = lookup("NVIDIA A100-SXM4-40GB")
    assert s is not None and s.name == "A100-SXM4-40GB"
    hw = s.hardware()
    peak = hw.rows * hw.cols * hw.clock_hz          # MACs/s
    assert abs(2 * peak / 1e12 - s.peak_tflops) / s.peak_tflops < 0.01


def test_chained_flag_is_opt_in_and_isolated_stays_default():
    """The depth-isolated corrections (backward_util_tiles, chain_overhead_*)
    must never apply unless a caller explicitly asks for them -- isolated-
    context anchors (compare_silicon.py) must see the plain, unmodified
    calibration."""
    from validate.silicon.gpu_specs import lookup
    spec = lookup("Tesla T4")
    isolated = spec.hardware(mac_bytes=2)
    chained = spec.hardware(mac_bytes=2, chained=True)
    assert isolated.backward_util_tiles is None
    assert isolated.chain_overhead_fwd_s == 0.0 and isolated.chain_overhead_bwd_s == 0.0
    assert chained.backward_util_tiles is not None
    assert chained.chain_overhead_bwd_s > 0.0


def test_attention_compare_runs_on_sample_profile():
    import json
    from validate.silicon.gpu_specs import lookup
    from simulator.attention import Attention, attention_step
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "attention", "sample_attention_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    hw = spec.hardware(mac_bytes=2)
    mem = spec.memory()
    attn = Attention("a", batch=prof["batch"], seq=prof["seq"],
                     n_heads=prof["n_heads"], d_head=prof["d_head"])
    naive = attention_step(attn, hw, mem, fused=False)
    fused = attention_step(attn, hw, mem, fused=True)
    assert naive.total_s > 0 and fused.total_s > 0
    assert fused.total_s < naive.total_s


def test_recompute_compare_runs_on_sample_profile():
    import json
    from validate.silicon.recompute.compare_recompute import predict
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "recompute", "sample_recompute_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    hw, mem, pred = predict(prof, spec)
    assert pred["stored"]["total"] > 0 and pred["recompute"]["total"] > 0
    # forward is identical either way -- recompute only changes backward
    assert pred["stored"]["forward"] == pred["recompute"]["forward"]


def test_gradaccum_compare_runs_on_sample_profile():
    import json
    from validate.silicon.gradaccum.compare_gradaccum import predict
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "gradaccum", "sample_gradaccum_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    ms1, mb1 = predict(prof, spec, "1")
    ms2, mb2 = predict(prof, spec, "2")
    assert ms1 > 0 and mb1 > 0
    # x2 accumulation should roughly halve predicted peak activations
    assert abs(mb2 / mb1 - 0.5) < 0.05


def test_gradaccum_memory_floor_fit_closes_the_real_gap():
    import json
    from validate.silicon.gradaccum.compare_gradaccum import predict, fit_memory_floor
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "profiles", "t4_fp16_gradaccum.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    floor_mb, scale = fit_memory_floor(prof, spec)
    assert floor_mb > 0 and scale > 0
    import dataclasses
    fitted_mem = dataclasses.replace(spec.memory(), framework_overhead_bytes=floor_mb * 1e6,
                                     framework_overhead_scale=scale)
    for frac, d in prof["configs"].items():
        _, adj_mb = predict(prof, spec, frac, mem=fitted_mem)
        meas_mb = d["peak_activation_mb"]
        assert abs(adj_mb - meas_mb) / meas_mb < 0.05


def test_optimizer_compare_runs_on_sample_profile():
    import json
    from validate.silicon.optimizer.compare_optimizer import predict_total, _STATES
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "optimizer", "sample_optimizer_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])                     # Tesla T4 -- no native bf16
    fp32_pred = predict_total(prof, spec, _STATES[0][1])
    bf16_pred = predict_total(prof, spec, _STATES[1][1])
    assert fp32_pred > 0 and bf16_pred > 0
    # T4/Turing has no native bf16 ALU throughput -- the emulation penalty
    # makes bf16 state MORE expensive here, matching the real T4 anchor
    # (measured -33% "saving"), not the naive bytes-only "always cheaper".
    assert bf16_pred > fp32_pred


def test_native_bf16_gpu_still_finds_lower_precision_cheaper():
    """On a GPU with native bf16 (Ampere+), the emulation penalty never
    applies -- state_bytes should behave like the abstract model again."""
    from validate.silicon.gpu_specs import lookup
    from simulator.dataflow import Layer, optimizer_seconds, Optimizer
    spec = lookup("A100-SXM4-40GB")
    hw = spec.hardware(mac_bytes=2)
    mem = spec.memory()
    layer = Layer("l", M=1, K=768, N=3072)
    fp32 = optimizer_seconds(layer, mem, Optimizer(state_bytes=4), hw)
    bf16 = optimizer_seconds(layer, mem, Optimizer(state_bytes=2), hw)
    assert bf16 < fp32


def test_chain_grid_compare_runs_on_sample_profile():
    import json
    from validate.silicon.chain_grid.compare_chain_grid import _predict_chain, refit, _hardware_with
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "chain_grid", "sample_chain_grid_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    mem = spec.memory()
    configs = prof["configs"]
    ce, ut = refit(configs, spec, mem)
    assert 0.0 < ce <= 1.0
    assert ut >= 1
    hw = _hardware_with(spec, ce, ut)
    fwd_ms, bwd_ms = _predict_chain(configs["gpt2_block_full"]["dims"],
                                    configs["gpt2_block_full"]["tokens"], hw, mem)
    assert fwd_ms > 0 and bwd_ms > 0


def test_chain_depth_compare_runs_on_sample_profile():
    import json
    from validate.silicon.chain_depth.compare_chain_depth import ols_fit, _predict
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "chain_depth", "sample_chain_depth_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    hw = spec.hardware(mac_bytes=2)
    mem = spec.memory()
    depths = sorted(int(d) for d in prof["results"].keys())
    xs = depths
    ys = [prof["results"][str(d)]["backward"]["mean_ms"] for d in depths]
    a, b, r2 = ols_fit(xs, ys)
    assert r2 > 0.99   # synthetic sample is an exact line by construction
    assert a > 0
    p_fwd, p_bwd = _predict(tuple(prof["shape"]), prof["tokens"], depths[0], hw, mem)
    assert p_fwd > 0 and p_bwd > 0


def test_chain_generalize_compare_runs_on_sample_profile():
    import json
    from validate.silicon.chain_generalize.compare_chain_generalize import _predict
    from validate.silicon.chain_depth.compare_chain_depth import ols_fit
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "chain_generalize", "sample_chain_generalize_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    hw = spec.hardware(mac_bytes=2)
    mem = spec.memory()
    shape = tuple(prof["shape"])
    depths = ["depth1_m8192", "depth2_m8192", "depth4_m8192", "depth8_m8192"]
    xs = [prof["results"][n]["depth"] for n in depths]
    ys = [prof["results"][n]["backward"]["mean_ms"] for n in depths]
    a, b, r2 = ols_fit(xs, ys)
    assert r2 > 0.99   # synthetic sample is an exact line by construction
    fwd, bwd = _predict(shape, 8192, 4, hw, mem)
    assert fwd > 0 and bwd > 0


def test_gemm_size_compare_runs_on_sample_profile():
    import json
    from simulator.compute import _occupancy
    from simulator.dataflow import Layer, training_step
    from validate.silicon.gpu_specs import lookup
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "validate", "silicon", "gemm_size", "sample_gemm_size_profile.json")
    with open(path) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    hw = spec.hardware(mac_bytes=2)
    mem = spec.memory()
    size = prof["sizes"][0]
    layer = Layer("l", M=prof["tokens"], K=size, N=size)
    p = training_step(layer, hw, mem)
    assert p.forward_s > 0 and p.backward_s > 0
    assert 0.0 <= _occupancy(size, size, hw) <= 1.0


def test_silicon_compare_runs_on_sample_profile():
    import json
    from validate.silicon.gpu_specs import lookup
    from validate.silicon.gemm.compare_silicon import predict
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "validate", "silicon", "gemm", "sample_profile.json")) as f:
        prof = json.load(f)
    spec = lookup(prof["gpu"])
    hw, mem, agg, by_name = predict(prof, spec)
    assert set(agg) == {"forward", "backward", "optimizer", "step"}
    assert agg["step"] == agg["forward"] + agg["backward"] + agg["optimizer"]
    assert all(v > 0 for v in agg.values())
    assert len(by_name) == 6


def test_attention_backward_is_2x_forward():
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(8 << 20, 100e9)
    attn = Attention("a", batch=8, seq=1024, n_heads=12, d_head=64)
    part = attention_step(attn, hw, mem)
    assert part.backward_s == 2.0 * part.forward_s
    assert part.optimizer_s == 0.0
    assert part.activation_bytes == attn.score_elems * hw.mac_bytes


def test_attention_cost_grows_quadratically_with_sequence_length():
    # both points memory-bound (see experiments/attention_scaling.py) so the
    # O(S^2) score-tensor traffic should dominate: 4x the sequence -> ~16x
    # the cost, far more than the 4x a GEMM's token count would cost.
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(64 << 20, 400e9)
    short = attention_step(Attention("a", 8, 512, 12, 64), hw, mem)
    long = attention_step(Attention("a", 8, 2048, 12, 64), hw, mem)
    assert short.forward_bound == long.forward_bound == "memory"
    assert long.total_s > 10 * short.total_s


def test_attention_fusion_never_slower_and_cuts_activations():
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(8 << 20, 100e9)
    for seq in (128, 1024, 8192):
        attn = Attention("a", batch=8, seq=seq, n_heads=12, d_head=64)
        naive = attention_step(attn, hw, mem, fused=False)
        fused = attention_step(attn, hw, mem, fused=True)
        assert fused.total_s <= naive.total_s
        assert fused.activation_bytes < naive.activation_bytes
        # same MACs either way -- fusion saves traffic, not compute
        naive_compute = matmul_cycles(seq, seq, 64, hw) + matmul_cycles(seq, 64, seq, hw)
        assert naive_compute > 0  # sanity the shapes line up (no exception above)


def test_attention_fusion_saving_grows_with_sequence_length():
    # fusion removes an O(S^2) traffic term -- the *fraction* saved should
    # grow as sequence length grows, unlike a SRAM-capacity-gated lever.
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(8 << 20, 100e9)

    def saved_frac(seq):
        attn = Attention("a", batch=8, seq=seq, n_heads=12, d_head=64)
        n = attention_step(attn, hw, mem, fused=False).total_s
        f = attention_step(attn, hw, mem, fused=True).total_s
        return (n - f) / n

    assert saved_frac(8192) > saved_frac(128)


def test_model_step_dispatches_attention_and_layers():
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(8 << 20, 100e9)
    gemms_only, _ = model_step(GPT2_BLOCK, hw, mem)
    full, parts = model_step(GPT2_BLOCK_FULL, hw, mem)
    assert full > gemms_only  # attention only adds cost
    names = {p.layer for p in parts}
    assert "attention" in names and "mlp_up" in names
    attn_part = next(p for p in parts if p.layer == "attention")
    solo = attention_step(ATTENTION, hw, mem)
    assert abs(attn_part.total_s - solo.total_s) < 1e-12

    fused_total, fused_parts = model_step(GPT2_BLOCK_FULL, hw, mem, fused_attention=True)
    assert fused_total < full            # fusion only helps
    fused_attn = next(p for p in fused_parts if p.layer == "attention")
    gemm_parts_unchanged = [p for p in fused_parts if p.layer != "attention"]
    naive_gemm_parts = [p for p in parts if p.layer != "attention"]
    assert [p.total_s for p in gemm_parts_unchanged] == [p.total_s for p in naive_gemm_parts]
    assert fused_attn.total_s < attn_part.total_s   # fused_attention only touches Attention items


def test_autotune_picks_a_valid_fast_plan():
    from simulator.autotune import autotune, _candidates
    hw = Hardware(rows=256, cols=256, clock_hz=1e9)
    mem = Memory(16 << 20, 100e9)
    best = autotune(GPT2_BLOCK, hw, mem)
    all_lat = [p.step_ms for p in _candidates(GPT2_BLOCK, hw, mem)]
    assert best.step_ms == min(all_lat)             # it is the minimum
    assert best.n_evaluated == 40
    tight = autotune(GPT2_BLOCK, hw, mem, memory_budget_mb=20)
    assert tight.activation_mb <= 20
    assert tight.step_ms >= best.step_ms            # budget costs something (or ties)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
