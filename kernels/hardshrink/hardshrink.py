import time
from math import prod
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a Python module
lib = load(
    name="hardshrink_lib",
    sources=["hardshrink_opt.cu"],
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


def torch_hardshrink(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_(F.hardshrink(x, lambd=0.5))
    return out


def torch_hardshrink_ref(x: torch.Tensor) -> torch.Tensor:
    return F.hardshrink(x, lambd=0.5)


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
    for i in range(warmup):
        perf_func(x, out)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        perf_func(x, out)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, *** time: {mean_time:.8f}ms")
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
    print("精度校验: 自定义 hardshrink vs F.hardshrink（多 shape）")
    print("=" * 85)
    failed = 0
    for shape in PRECISION_SHAPES:
        shape_desc = str(shape)
        print("-" * 85)
        print(f"shape={shape_desc}, numel={prod(shape)}")

        x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
        ref_f32 = torch_hardshrink_ref(x)
        y = torch.zeros_like(x)

        lib.hardshrink_f32(x, y)
        failed += not validate_against_torch(
            "f32", y, ref_f32, atol=1e-6, rtol=1e-6, shape_desc=shape_desc
        )
        lib.hardshrink_f32x4(x, y)
        failed += not validate_against_torch(
            "f32x4", y, ref_f32, atol=1e-6, rtol=1e-6, shape_desc=shape_desc
        )
        torch_hardshrink(x, y)
        failed += not validate_against_torch(
            "f32_th", y, ref_f32, atol=1e-6, rtol=1e-6, shape_desc=shape_desc
        )

        x16 = x.half().contiguous()
        ref_f16 = torch_hardshrink_ref(x16)
        y16 = torch.zeros_like(x16)

        lib.hardshrink_f16(x16, y16)
        failed += not validate_against_torch(
            "f16", y16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        lib.hardshrink_f16x2(x16, y16)
        failed += not validate_against_torch(
            "f16x2", y16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        lib.hardshrink_f16x8(x16, y16)
        failed += not validate_against_torch(
            "f16x8", y16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        lib.hardshrink_f16x8_pack(x16, y16)
        failed += not validate_against_torch(
            "f16x8_pack", y16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        torch_hardshrink(x16, y16)
        failed += not validate_against_torch(
            "f16_th", y16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
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

run_multi_shape_precision_suite()

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    ref_f32 = torch_hardshrink_ref(x)
    _, _ = run_benchmark(lib.hardshrink_f32, x, "f32", y)
    validate_against_torch("f32", y, ref_f32, atol=1e-6, rtol=1e-6)
    _, _ = run_benchmark(lib.hardshrink_f32x4, x, "f32x4", y)
    validate_against_torch("f32x4", y, ref_f32, atol=1e-6, rtol=1e-6)
    _, _ = run_benchmark(torch_hardshrink, x, "f32_th", y)
    validate_against_torch("f32_th", y, ref_f32, atol=1e-6, rtol=1e-6)

    print("-" * 85)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    ref_f16 = torch_hardshrink_ref(x_f16)
    _, _ = run_benchmark(lib.hardshrink_f16, x_f16, "f16", y_f16)
    validate_against_torch("f16", y_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(lib.hardshrink_f16x2, x_f16, "f16x2", y_f16)
    validate_against_torch("f16x2", y_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(lib.hardshrink_f16x8, x_f16, "f16x8", y_f16)
    validate_against_torch("f16x8", y_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(lib.hardshrink_f16x8_pack, x_f16, "f16x8pack", y_f16)
    validate_against_torch("f16x8_pack", y_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(torch_hardshrink, x_f16, "f16_th", y_f16)
    validate_against_torch("f16_th", y_f16, ref_f16, atol=1e-3, rtol=1e-3)
    print("-" * 85)
