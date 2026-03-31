#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// 定义阈值
#define THRESHOLD_A 3.0
#define THRESHOLD_B -3.0

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("Tensor dtype must be " #th_type);                \
  }

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// FP32: 用乘法替代除法 (1/6 = 0.16666667f)
__device__ __forceinline__ float hardswish(float x) {
  float x_plus_3 = x + 3.0f;

  float relu6 = fmin(fmax(x_plus_3, 0.0f), 6.0f);

  return x * relu6 * 0.16666667f;
}

// FP16: 用 __hmul 替代 __hdiv，预算 1/6
__device__ __forceinline__ half hardswish_half(half x) {
  const half three = __float2half(3.f);
  const half inv6 = __float2half(0.16666667f);
  const half zero = __float2half(0.f);
  const half six = __float2half(6.f);

  half x_plus_3 = __hadd(x, three);
  half relu6 = __hmin(__hmax(x_plus_3, zero), six);

  return __hmul(__hmul(x, relu6), inv6);
}

__device__ __forceinline__ half2 hardswish_half2(half2 x) {
  const half2 three = __float2half2_rn(3.f);
  const half2 inv6 = __float2half2_rn(0.16666667f);
  const half2 zero = __float2half2_rn(0.f);
  const half2 six = __float2half2_rn(6.f);

  half2 x_plus_3 = __hadd2(x, three);
  half2 relu6 = __hmin2(__hmax2(x_plus_3, zero), six);

  return __hmul2(__hmul2(x, relu6), inv6);
}

// FP32
__global__ void hardswish_f32_kernel(float *__restrict__ x,
                                     float *__restrict__ y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = hardswish(x[idx]);
}

__global__ void hardswish_f32x4_kernel(float *__restrict__ x,
                                       float *__restrict__ y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = hardswish(reg_x.x);
    reg_y.y = hardswish(reg_x.y);
    reg_y.z = hardswish(reg_x.z);
    reg_y.w = hardswish(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  } else {
    for (int i = idx; i < N; ++i) {
      y[i] = hardswish(x[i]);
    }
  }
}

// FP16
__global__ void hardswish_f16_kernel(half *__restrict__ x, half *__restrict__ y,
                                     int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = hardswish_half(x[idx]);
}

__global__ void hardswish_f16x2_kernel(half *__restrict__ x,
                                       half *__restrict__ y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 1 < N) {
    half2 reg_x = HALF2(x[idx]);
    HALF2(y[idx]) = hardswish_half2(reg_x);
  } else if (idx < N) {
    y[idx] = hardswish_half(x[idx]);
  }
}

__global__ void hardswish_f16x8_kernel(half *__restrict__ x,
                                       half *__restrict__ y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 7 < N) {
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    HALF2(y[idx + 0]) = hardswish_half2(reg_x_0);
    HALF2(y[idx + 2]) = hardswish_half2(reg_x_1);
    HALF2(y[idx + 4]) = hardswish_half2(reg_x_2);
    HALF2(y[idx + 6]) = hardswish_half2(reg_x_3);
  } else if (idx < N) {
    for (int i = idx; i < N; ++i) {
      y[i] = hardswish_half(x[i]);
    }
  }
}

__global__ void hardswish_f16x8_pack_kernel(half *__restrict__ x,
                                            half *__restrict__ y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 7 < N) {
    half2 pack_x[4], pack_y[4];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pack_y[i] = hardswish_half2(pack_x[i]);
    }
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  } else if (idx < N) {
    for (int i = 0; idx + i < N; ++i) {
      y[idx + i] = hardswish_half(x[idx + i]);
    }
  }
}

#define TORCH_BINDING_HARDSWISH(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void hardswish_##packed_type(torch::Tensor x, torch::Tensor y) {             \
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
      hardswish_##packed_type##_kernel<<<grid, block>>>(                       \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        hardswish_##packed_type##_kernel<<<grid, block>>>(                     \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        hardswish_##packed_type##_kernel<<<grid, block>>>(                     \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_HARDSWISH(f32, torch::kFloat32, float, 1)
TORCH_BINDING_HARDSWISH(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_HARDSWISH(f16, torch::kHalf, half, 1)
TORCH_BINDING_HARDSWISH(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_HARDSWISH(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_HARDSWISH(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hardswish_f32)
  TORCH_BINDING_COMMON_EXTENSION(hardswish_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(hardswish_f16)
  TORCH_BINDING_COMMON_EXTENSION(hardswish_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(hardswish_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(hardswish_f16x8_pack)
}
