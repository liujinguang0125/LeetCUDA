#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)
// to clear the error among self defined gelu and pytorch gelu. Calculate
// $\sqrt{\frac{\pi}{2}}$ by $\sqrt{2 * \pi} / 2$
#define HALF_SQRT_2_PI                                                         \
  __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)

#define HALF_GELU_OPS gelu_tanh_approximate
#define HALF2_GELU_OPS gelu_tanh_approximate_half2
#define GELU_OPS gelu_tanh_approximate

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
// half 标量版本
__inline__ __device__ half gelu_tanh_approximate(half x) {
  half x_cube = __hmul(x, __hmul(x, x));
  half inner = __hmul(HALF_SQRT_2_PI, (__hadd(x, __hmul(HALF_V_APP, x_cube))));
  return __hmul(HALF_DIV2, __hmul(x, (__hadd(HALF_1, __float2half(tanhf(__half2float(inner)))))));
}

// half2 向量版本：一条指令同时处理 2 个 half，指令数约减半
// half2 向量版本：两路 half 升 float32 各自 tanhf 再 pack 回 half2
__inline__ __device__ half2 gelu_tanh_approximate_half2(half2 x) {
  const half2 half2_val = __float2half2_rn(0.5f);
  const half2 one2 = __float2half2_rn(1.0f);
  const half2 sqrt2pi2 = __float2half2_rn(0.7978845608f);
  const half2 vapp2 = __float2half2_rn(0.044715f);

  half2 x_cube = __hmul2(__hmul2(x, x), x);
  half2 inner = __hmul2(sqrt2pi2, __hadd2(x, __hmul2(vapp2, x_cube)));
  float inner_lo = __half2float(__low2half(inner));
  float inner_hi = __half2float(__high2half(inner));
  half2 tanh_val = __halves2half2(__float2half(tanhf(inner_lo)),
                                  __float2half(tanhf(inner_hi)));
  return __hmul2(__hmul2(half2_val, x), __hadd2(one2, tanh_val));
}

__inline__ __device__ float gelu_tanh_approximate(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

__inline__ __device__ float gelu_none_approximate(float x) {
  return x * 0.5 * (1 + erff(x * M_SQRT1_2));
}

// FP32
__global__ void gelu_f32_kernel(float *__restrict__ x, float *__restrict__ y,
                                int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = GELU_OPS(v);
  }
}

// FP32 Vec4
__global__ void gelu_f32x4_kernel(float *__restrict__ x, float *__restrict__ y,
                                  int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
    reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
    reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
    reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);
    reg_y.x = GELU_OPS(reg_x.x);
    reg_y.y = GELU_OPS(reg_x.y);
    reg_y.z = GELU_OPS(reg_x.z);
    reg_y.w = GELU_OPS(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  } else {
    for (int i = idx; i < N; ++i) {
      float v = fminf(fmaxf(x[i], MIN_EXP_F32), MAX_EXP_F32);
      y[i] = GELU_OPS(v);
    }
  }
}

// FP16 标量
__global__ void gelu_f16_kernel(half *__restrict__ x, half *__restrict__ y,
                                int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    half v = __hmin(__hmax(x[idx], MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = HALF_GELU_OPS(v);
  }
}

// FP16x2: 使用 half2 向量 intrinsic 并行计算
__global__ void gelu_f16x2_kernel(half *__restrict__ x, half *__restrict__ y,
                                  int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  const half2 min2 = __halves2half2(MIN_EXP_F16, MIN_EXP_F16);
  const half2 max2 = __halves2half2(MAX_EXP_F16, MAX_EXP_F16);

  if (idx + 1 < N) {
    half2 reg_x = HALF2(x[idx]);
    reg_x = __hmin2(__hmax2(reg_x, min2), max2);
    HALF2(y[idx]) = HALF2_GELU_OPS(reg_x);
  } else if (idx < N) {
    half v = __hmin(__hmax(x[idx], MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = HALF_GELU_OPS(v);
  }
}

// FP16x8 unpack: 4 个 half2 向量并行
__global__ void gelu_f16x8_kernel(half *__restrict__ x, half *__restrict__ y,
                                  int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  const half2 min2 = __halves2half2(MIN_EXP_F16, MIN_EXP_F16);
  const half2 max2 = __halves2half2(MAX_EXP_F16, MAX_EXP_F16);

  if (idx + 7 < N) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      half2 reg_x = __hmin2(__hmax2(HALF2(x[idx + i]), min2), max2);
      HALF2(y[idx + i]) = HALF2_GELU_OPS(reg_x);
    }
    // half2 reg_x_0 = __hmin2(__hmax2(HALF2(x[idx + 0]), min2), max2);
    // half2 reg_x_1 = __hmin2(__hmax2(HALF2(x[idx + 2]), min2), max2);
    // half2 reg_x_2 = __hmin2(__hmax2(HALF2(x[idx + 4]), min2), max2);
    // half2 reg_x_3 = __hmin2(__hmax2(HALF2(x[idx + 6]), min2), max2);
    // HALF2(y[idx + 0]) = HALF2_GELU_OPS(reg_x_0);
    // HALF2(y[idx + 2]) = HALF2_GELU_OPS(reg_x_1);
    // HALF2(y[idx + 4]) = HALF2_GELU_OPS(reg_x_2);
    // HALF2(y[idx + 6]) = HALF2_GELU_OPS(reg_x_3);
  } else if (idx < N) {
    for (int i = idx; i < N; ++i) {
      half v = __hmin(__hmax(x[i], MIN_EXP_F16), MAX_EXP_F16);
      y[i] = HALF_GELU_OPS(v);
    }
  }
}

// FP16x8 pack: 128-bit load/store + half2 向量计算
__global__ void gelu_f16x8_pack_kernel(half *__restrict__ x,
                                       half *__restrict__ y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx + 7 < N) {
    half2 pack_x[4], pack_y[4];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

    const half2 min2 = __halves2half2(MIN_EXP_F16, MIN_EXP_F16);
    const half2 max2 = __halves2half2(MAX_EXP_F16, MAX_EXP_F16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half2 v = __hmin2(__hmax2(pack_x[i], min2), max2);
      HALF2(pack_y[i]) = HALF2_GELU_OPS(v);
    }
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  } else if (idx < N) {
    for (int i = 0; idx + i < N; ++i) {
      half v = __hmin(__hmax(x[idx + i], MIN_EXP_F16), MAX_EXP_F16);
      y[idx + i] = HALF_GELU_OPS(v);
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)     \
  void gelu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      gelu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_GELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_GELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_GELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_GELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_GELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_GELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_pack)
}
