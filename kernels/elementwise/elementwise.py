import time
from math import prod
from typing import Optional, Sequence

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)


def torch_add_copy_(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """将 torch.add(a, b) 写入 out，兼容不支持 torch.add(..., out=) 的环境。"""
    out.copy_(torch.add(a, b))


# Load the CUDA kernel as a python module
lib = load(
    name="elementwise_lib",
    sources=["elementwise.cu"],
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


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


def validate_add_against_torch(
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
    print("精度校验: 自定义 elementwise_add vs torch.add（多 shape）")
    print("=" * 85)
    failed = 0
    for shape in PRECISION_SHAPES:
        shape_desc = str(shape)
        print("-" * 85)
        print(f"shape={shape_desc}, numel={prod(shape)}")

        a = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
        b = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
        ref_f32 = torch.add(a, b)
        c = torch.zeros_like(a)

        lib.elementwise_add_f32(a, b, c)
        failed += not validate_add_against_torch(
            "f32", c, ref_f32, atol=1e-6, rtol=1e-6, shape_desc=shape_desc
        )
        lib.elementwise_add_f32x4(a, b, c)
        failed += not validate_add_against_torch(
            "f32x4", c, ref_f32, atol=1e-6, rtol=1e-6, shape_desc=shape_desc
        )
        torch_add_copy_(a, b, c)
        failed += not validate_add_against_torch(
            "f32_th", c, ref_f32, atol=1e-6, rtol=1e-6, shape_desc=shape_desc
        )

        a16 = a.half().contiguous()
        b16 = b.half().contiguous()
        ref_f16 = torch.add(a16, b16)
        c16 = torch.zeros_like(a16)

        lib.elementwise_add_f16(a16, b16, c16)
        failed += not validate_add_against_torch(
            "f16", c16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        lib.elementwise_add_f16x2(a16, b16, c16)
        failed += not validate_add_against_torch(
            "f16x2", c16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        lib.elementwise_add_f16x8(a16, b16, c16)
        failed += not validate_add_against_torch(
            "f16x8", c16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        lib.elementwise_add_f16x8_pack(a16, b16, c16)
        failed += not validate_add_against_torch(
            "f16x8_pack", c16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
        )
        torch_add_copy_(a16, b16, c16)
        failed += not validate_add_against_torch(
            "f16_th", c16, ref_f16, atol=1e-3, rtol=1e-3, shape_desc=shape_desc
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
    a = torch.randn((S, K)).cuda().float().contiguous()
    b = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()
    ref_f32 = torch.add(a, b)
    _, _ = run_benchmark(lib.elementwise_add_f32, a, b, "f32", c)
    validate_add_against_torch("f32", c, ref_f32, atol=1e-6, rtol=1e-6)
    _, _ = run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4", c)
    validate_add_against_torch("f32x4", c, ref_f32, atol=1e-6, rtol=1e-6)
    _, _ = run_benchmark(torch_add_copy_, a, b, "f32_th", c)
    validate_add_against_torch("f32_th", c, ref_f32, atol=1e-6, rtol=1e-6)

    print("-" * 85)
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    c_f16 = c.half().contiguous()
    ref_f16 = torch.add(a_f16, b_f16)
    _, _ = run_benchmark(lib.elementwise_add_f16, a_f16, b_f16, "f16", c_f16)
    validate_add_against_torch("f16", c_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", c_f16)
    validate_add_against_torch("f16x2", c_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", c_f16)
    validate_add_against_torch("f16x8", c_f16, ref_f16, atol=1e-3, rtol=1e-3)
    _, _ = run_benchmark(
        lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", c_f16
    )
    validate_add_against_torch(
        "f16x8_pack", c_f16, ref_f16, atol=1e-3, rtol=1e-3
    )
    _, _ = run_benchmark(torch_add_copy_, a_f16, b_f16, "f16_th", c_f16)
    validate_add_against_torch("f16_th", c_f16, ref_f16, atol=1e-3, rtol=1e-3)
    print("-" * 85)
