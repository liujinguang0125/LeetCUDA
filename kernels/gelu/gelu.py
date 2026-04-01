import time
from math import prod
from typing import Optional, Sequence

import torch
import torch.nn
import torch.utils
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

GPU_NAME = torch.cuda.get_device_name(0)
GPU_FP32_PEAK_TFLOPS = {
    "Tesla T4": 8.1,
    "Tesla V100-SXM2-16GB": 15.7,
    "Tesla V100-SXM2-32GB": 15.7,
    "Tesla A100-SXM4-40GB": 19.5,
    "Tesla A100-SXM4-80GB": 19.5,
    "NVIDIA A100-SXM4-80GB": 19.5,
    "NVIDIA A100-PCIE-40GB": 19.5,
    "NVIDIA A800-SXM4-80GB": 19.5,
    "NVIDIA H100": 67.0,
    "NVIDIA H800": 67.0,
    "NVIDIA GeForce RTX 3090": 35.6,
    "NVIDIA GeForce RTX 4090": 82.6,
}.get(GPU_NAME, 0)
GPU_FP16_PEAK_TFLOPS = GPU_FP32_PEAK_TFLOPS * 2
GPU_BW_GB_S = {
    "Tesla T4": 320,
    "Tesla V100-SXM2-16GB": 900,
    "Tesla V100-SXM2-32GB": 900,
    "Tesla A100-SXM4-40GB": 2039,
    "Tesla A100-SXM4-80GB": 2039,
    "NVIDIA A100-SXM4-80GB": 2039,
    "NVIDIA A100-PCIE-40GB": 1555,
    "NVIDIA A800-SXM4-80GB": 2039,
    "NVIDIA H100": 3350,
    "NVIDIA H800": 3350,
    "NVIDIA GeForce RTX 3090": 936,
    "NVIDIA GeForce RTX 4090": 1008,
}.get(GPU_NAME, 0)

GELU_FLOPS_PER_ELEMENT = 14

# Load the CUDA kernel as a python module
lib = load(
    name="gelu_lib",
    sources=["gelu.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

torch_gelu = torch.nn.GELU("tanh")


def torch_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    return torch_gelu(x)


def torch_gelu_copy_(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_(torch_gelu(x))
    return out


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"

    numel = x.numel()
    dtype_bytes = x.element_size()
    is_f16 = dtype_bytes == 2

    total_bytes = numel * dtype_bytes * 2  # read x + write y
    bw_gb_s = total_bytes / (mean_time * 1e-3) / 1e9
    bw_util = bw_gb_s / GPU_BW_GB_S * 100 if GPU_BW_GB_S > 0 else 0

    total_flops = numel * GELU_FLOPS_PER_ELEMENT
    tflops = total_flops / (mean_time * 1e-3) / 1e12
    peak = GPU_FP16_PEAK_TFLOPS if is_f16 else GPU_FP32_PEAK_TFLOPS
    compute_util = tflops / peak * 100 if peak > 0 else 0

    print(
        f"{out_info:>18}: {mean_time:.6f}ms, "
        f"BW: {bw_gb_s:7.1f} GB/s ({bw_util:5.1f}%), "
        f"TFLOPS: {tflops:6.3f} ({compute_util:5.1f}%)"
    )
    if show_all:
        print(out)
    return out, mean_time


def validate_against_torch(
    tag: str,
    out: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
    shape_desc: str = "",
) -> bool:
    o = out.detach().float().flatten()
    r = ref.detach().float().flatten()
    abs_diff = (o - r).abs()
    max_abs = abs_diff.max().item()
    denom = r.abs().clamp(min=1e-6)
    max_rel = (abs_diff / denom).max().item()
    ok = torch.allclose(o, r, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    prefix = f"[{shape_desc}] " if shape_desc else ""
    print(
        f"{prefix}{'check_' + tag:>18}: {status}, max_abs:{max_abs:.8e}, "
        f"max_rel:{max_rel:.8e}, atol:{atol:.1e}, rtol:{rtol:.1e}"
    )
    return ok


PRECISION_SHAPES: Sequence[tuple[int, ...]] = (
    (1,),
    (3,),
    (7,),
    (8,),
    (15,),
    (255,),
    (256,),
    (1023,),
    (1025,),
    (2, 3, 5),
    (4, 1023),
    (17, 2049),
    (32, 8000),
)


def run_multi_shape_precision_suite(seed: int = 12345) -> None:
    torch.manual_seed(seed)
    print("=" * 85)
    print("精度校验: 自定义 gelu vs torch.nn.GELU('tanh')（多 shape）")
    print("=" * 85)
    failed = 0
    for shape in PRECISION_SHAPES:
        shape_desc = str(shape)
        print("-" * 85)
        print(f"shape={shape_desc}, numel={prod(shape)}")

        x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
        ref_f32 = torch_gelu_ref(x)
        y = torch.zeros_like(x)

        lib.gelu_f32(x, y)
        failed += not validate_against_torch(
            "f32", y, ref_f32, atol=1e-5, rtol=1e-5, shape_desc=shape_desc
        )
        lib.gelu_f32x4(x, y)
        failed += not validate_against_torch(
            "f32x4", y, ref_f32, atol=1e-5, rtol=1e-5, shape_desc=shape_desc
        )
        torch_gelu_copy_(x, y)
        failed += not validate_against_torch(
            "f32_th", y, ref_f32, atol=1e-5, rtol=1e-5, shape_desc=shape_desc
        )

        x16 = x.half().contiguous()
        ref_f16 = torch_gelu_ref(x16)
        y16 = torch.zeros_like(x16)

        lib.gelu_f16(x16, y16)
        failed += not validate_against_torch(
            "f16", y16, ref_f16, atol=5e-3, rtol=5e-3, shape_desc=shape_desc
        )
        lib.gelu_f16x2(x16, y16)
        failed += not validate_against_torch(
            "f16x2", y16, ref_f16, atol=5e-3, rtol=5e-3, shape_desc=shape_desc
        )
        lib.gelu_f16x8(x16, y16)
        failed += not validate_against_torch(
            "f16x8", y16, ref_f16, atol=5e-3, rtol=5e-3, shape_desc=shape_desc
        )
        lib.gelu_f16x8_pack(x16, y16)
        failed += not validate_against_torch(
            "f16x8_pack", y16, ref_f16, atol=5e-3, rtol=5e-3, shape_desc=shape_desc
        )
        torch_gelu_copy_(x16, y16)
        failed += not validate_against_torch(
            "f16_th", y16, ref_f16, atol=5e-3, rtol=5e-3, shape_desc=shape_desc
        )

    print("=" * 85)
    if failed:
        print(f"多 shape 精度: 共 {failed} 项未通过 allclose 阈值")
    else:
        print("多 shape 精度: 全部通过")
    print("=" * 85)


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

print(f"GPU: {GPU_NAME}")
print(f"FP32 Peak: {GPU_FP32_PEAK_TFLOPS} TFLOPS, "
      f"FP16 Peak: {GPU_FP16_PEAK_TFLOPS} TFLOPS, "
      f"BW Peak: {GPU_BW_GB_S} GB/s")

run_multi_shape_precision_suite()

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    ref_f32 = torch_gelu_ref(x)
    _, _ = run_benchmark(lib.gelu_f32, x, "f32", y)
    validate_against_torch("f32", y, ref_f32, atol=1e-5, rtol=1e-5)
    _, _ = run_benchmark(lib.gelu_f32x4, x, "f32x4", y)
    validate_against_torch("f32x4", y, ref_f32, atol=1e-5, rtol=1e-5)
    _, _ = run_benchmark(torch_gelu_copy_, x, "f32_th", y)
    validate_against_torch("f32_th", y, ref_f32, atol=1e-5, rtol=1e-5)

    print("-" * 85)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    ref_f16 = torch_gelu_ref(x_f16)
    _, _ = run_benchmark(lib.gelu_f16, x_f16, "f16", y_f16)
    validate_against_torch("f16", y_f16, ref_f16, atol=5e-3, rtol=5e-3)
    _, _ = run_benchmark(lib.gelu_f16x2, x_f16, "f16x2", y_f16)
    validate_against_torch("f16x2", y_f16, ref_f16, atol=5e-3, rtol=5e-3)
    _, _ = run_benchmark(lib.gelu_f16x8, x_f16, "f16x8", y_f16)
    validate_against_torch("f16x8", y_f16, ref_f16, atol=5e-3, rtol=5e-3)
    _, _ = run_benchmark(lib.gelu_f16x8_pack, x_f16, "f16x8pack", y_f16)
    validate_against_torch("f16x8_pack", y_f16, ref_f16, atol=5e-3, rtol=5e-3)
    _, _ = run_benchmark(torch_gelu_copy_, x_f16, "f16_th", y_f16)
    validate_against_torch("f16_th", y_f16, ref_f16, atol=5e-3, rtol=5e-3)
    print("-" * 85)
