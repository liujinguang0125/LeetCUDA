import time
from functools import partial
from typing import Optional, Sequence

import torch
from torch.nn.functional import embedding
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="embedding",
    sources=["embedding.cu"],
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
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start = time.time()
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
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>23}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


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
        f"{prefix}{'check_' + tag:>23}: {status}, max_abs:{max_abs:.8e}, "
        f"max_rel:{max_rel:.8e}, atol:{atol:.1e}, rtol:{rtol:.1e}"
    )
    return ok


PRECISION_CONFIGS: Sequence[dict] = (
    {"vocab": 64, "seqlen": 1, "emb_size": 8},
    {"vocab": 64, "seqlen": 3, "emb_size": 8},
    {"vocab": 128, "seqlen": 16, "emb_size": 32},
    {"vocab": 256, "seqlen": 7, "emb_size": 64},
    {"vocab": 512, "seqlen": 32, "emb_size": 128},
    {"vocab": 1024, "seqlen": 64, "emb_size": 256},
    {"vocab": 1024, "seqlen": 128, "emb_size": 512},
    {"vocab": 2048, "seqlen": 256, "emb_size": 1024},
    {"vocab": 4096, "seqlen": 512, "emb_size": 768},
    {"vocab": 8192, "seqlen": 1024, "emb_size": 512},
    {"vocab": 32000, "seqlen": 128, "emb_size": 1024},
)


def run_multi_shape_precision_suite(seed: int = 42) -> None:
    torch.manual_seed(seed)
    print("=" * 110)
    print("精度校验: 自定义 embedding vs F.embedding（多 embedding size）")
    print("=" * 110)
    failed = 0

    for cfg in PRECISION_CONFIGS:
        vocab = cfg["vocab"]
        seqlen = cfg["seqlen"]
        emb_size = cfg["emb_size"]
        shape_desc = f"V={vocab}, S={seqlen}, E={emb_size}"
        print("-" * 110)
        print(f"{shape_desc}")

        ids = torch.randint(0, vocab, (seqlen,), device="cuda", dtype=torch.int32).contiguous()
        weight_f32 = torch.randn(vocab, emb_size, device="cuda", dtype=torch.float32).contiguous()
        ref_f32 = embedding(ids.long(), weight_f32)
        o_f32 = torch.zeros(seqlen, emb_size, device="cuda", dtype=torch.float32).contiguous()

        lib.embedding_f32(ids, weight_f32, o_f32)
        failed += not validate_against_torch(
            "f32", o_f32, ref_f32, atol=0, rtol=0, shape_desc=shape_desc
        )

        if emb_size % 4 == 0:
            o_f32.fill_(0)
            lib.embedding_f32x4(ids, weight_f32, o_f32)
            failed += not validate_against_torch(
                "f32x4", o_f32, ref_f32, atol=0, rtol=0, shape_desc=shape_desc
            )
            o_f32.fill_(0)
            lib.embedding_f32x4_pack(ids, weight_f32, o_f32)
            failed += not validate_against_torch(
                "f32x4_pack", o_f32, ref_f32, atol=0, rtol=0, shape_desc=shape_desc
            )

        weight_f16 = weight_f32.half().contiguous()
        ref_f16 = embedding(ids.long(), weight_f16)
        o_f16 = torch.zeros(seqlen, emb_size, device="cuda", dtype=torch.float16).contiguous()

        lib.embedding_f16(ids, weight_f16, o_f16)
        failed += not validate_against_torch(
            "f16", o_f16, ref_f16, atol=0, rtol=0, shape_desc=shape_desc
        )

        if emb_size % 8 == 0:
            o_f16.fill_(0)
            lib.embedding_f16x8(ids, weight_f16, o_f16)
            failed += not validate_against_torch(
                "f16x8", o_f16, ref_f16, atol=0, rtol=0, shape_desc=shape_desc
            )
            o_f16.fill_(0)
            lib.embedding_f16x8_pack(ids, weight_f16, o_f16)
            failed += not validate_against_torch(
                "f16x8_pack", o_f16, ref_f16, atol=0, rtol=0, shape_desc=shape_desc
            )

    print("=" * 110)
    if failed:
        print(f"多 shape 精度: 共 {failed} 项未通过")
    else:
        print("多 shape 精度: 全部通过")
    print("=" * 110)


Ms = [1024, 4096]
Ns = [2048, 4096]
Ks = [512, 1024]
MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]

run_multi_shape_precision_suite()

for M, N, K in MNKs:
    print("-" * 110)
    print(" " * 45 + f"MaxV={M}, SeqLen={N}, EmbSize={K}")
    i = torch.randint(0, M, size=(N,)).cuda().int().contiguous()
    weight = torch.randn((M, K)).float().cuda().contiguous()
    o = torch.zeros((N, K)).float().cuda().contiguous()
    ref_f32 = embedding(i.long(), weight)

    run_benchmark(lib.embedding_f32, i, weight, "f32", o)
    validate_against_torch("f32", o, ref_f32, atol=0, rtol=0)
    run_benchmark(lib.embedding_f32x4, i, weight, "f32x4", o)
    validate_against_torch("f32x4", o, ref_f32, atol=0, rtol=0)
    run_benchmark(lib.embedding_f32x4_pack, i, weight, "f32x4_pack", o)
    validate_against_torch("f32x4_pack", o, ref_f32, atol=0, rtol=0)
    run_benchmark(partial(embedding), i, weight, "f32_th")

    print("-" * 110)
    weight_f16 = torch.randn((M, K)).half().cuda().contiguous()
    o_f16 = torch.zeros((N, K)).half().cuda().contiguous()
    ref_f16 = embedding(i.long(), weight_f16)
    run_benchmark(lib.embedding_f16, i, weight_f16, "f16", o_f16)
    validate_against_torch("f16", o_f16, ref_f16, atol=0, rtol=0)
    run_benchmark(lib.embedding_f16x8, i, weight_f16, "f16x8", o_f16)
    validate_against_torch("f16x8", o_f16, ref_f16, atol=0, rtol=0)
    run_benchmark(lib.embedding_f16x8_pack, i, weight_f16, "f16x8_pack", o_f16)
    validate_against_torch("f16x8_pack", o_f16, ref_f16, atol=0, rtol=0)
    run_benchmark(partial(embedding), i, weight_f16, "f16_th")
    print("-" * 110)
