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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDGST128BITS(value) (reinterpret_cast<const float4 *>(&(value)))

__global__ void embedding_f32_kernel(const int *__restrict__ idx,
                                     const float *__restrict__ weight,
                                     float *__restrict__ output, int n,
                                     int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  // int tid = bx * blockDim.x + tx;
  if (tx < emb_size) {
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
  }
}

__global__ void embedding_f32x4_kernel(const int *__restrict__ idx,
                                       const float *__restrict__ weight,
                                       float *__restrict__ output, int n,
                                       int emb_size) {
  int tx = threadIdx.x * 4;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  int out_base = bx * emb_size;
  if (tx + 3 < emb_size) {
    float pack_x[4];
    pack_x[0] = weight[offset + tx];
    pack_x[1] = weight[offset + tx + 1];
    pack_x[2] = weight[offset + tx + 2];
    pack_x[3] = weight[offset + tx + 3];

    output[out_base + tx] = pack_x[0];
    output[out_base + tx + 1] = pack_x[1];
    output[out_base + tx + 2] = pack_x[2];
    output[out_base + tx + 3] = pack_x[3];
  } else {
    for (int i = tx; i < emb_size; ++i) {
      output[out_base + i] = weight[offset + i];
    }
  }
}

__global__ void embedding_f32x4_pack_kernel(const int *__restrict__ idx,
                                            const float *__restrict__ weight,
                                            float *__restrict__ output, int n,
                                            int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  int out_base = bx * emb_size;
  int ex = 4 * tx;
  if (ex + 3 < emb_size) {
    LDST128BITS(output[out_base + ex]) = __ldg(LDGST128BITS(weight[offset + ex]));
  } else {
    for (int i = ex; i < emb_size; ++i) {
      output[out_base + i] = weight[offset + i];
    }
  }
}

__global__ void embedding_f16_kernel(const int *__restrict__ idx,
                                     const half *__restrict__ weight,
                                     half *__restrict__ output, int n,
                                     int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  if (tx < emb_size) {
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
  }
}

__global__ void embedding_f16x8_kernel(const int *__restrict__ idx,
                                      half *__restrict__ weight,
                                       half *__restrict__ output, int n,
                                       int emb_size) {
  int tx = threadIdx.x * 8;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  int out_base = bx * emb_size;

  if (tx + 7 < emb_size) {
    half2 pack_x[4];
    pack_x[0] = HALF2(weight[offset + tx]);
    pack_x[1] = HALF2(weight[offset + tx + 2]);
    pack_x[2] = HALF2(weight[offset + tx + 4]);
    pack_x[3] = HALF2(weight[offset + tx + 6]);

    HALF2(output[out_base + tx]) = pack_x[0];
    HALF2(output[out_base + tx + 2]) = pack_x[1];
    HALF2(output[out_base + tx + 4]) = pack_x[2];
    HALF2(output[out_base + tx + 6]) = pack_x[3];

    // output[out_base + tx] = weight[offset + tx];
    // output[out_base + tx + 1] = weight[offset + tx + 1];
    // output[out_base + tx + 2] = weight[offset + tx + 2];
    // output[out_base + tx + 3] = weight[offset + tx + 3];
    // output[out_base + tx + 4] = weight[offset + tx + 4];
    // output[out_base + tx + 5] = weight[offset + tx + 5];
    // output[out_base + tx + 6] = weight[offset + tx + 6];
    // output[out_base + tx + 7] = weight[offset + tx + 7];
  } else {
    for (int i = tx; i < emb_size; ++i) {
      output[out_base + i] = weight[offset + i];
    }
  }
}

__global__ void embedding_f16x8_pack_kernel(const int *__restrict__ idx,
                                            const half *__restrict__ weight,
                                            half *__restrict__ output, int n,
                                            int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  int out_base = bx * emb_size;
  int ex = 8 * tx;
  if (ex + 7 < emb_size) {
    LDST128BITS(output[out_base + ex]) = __ldg(LDGST128BITS(weight[offset + ex]));
  } else {
    for (int i = ex; i < emb_size; ++i) {
      output[out_base + i] = weight[offset + i];
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
                               torch::Tensor o) {                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
    CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
    CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
                                                                               \
    const int N = a.size(0);                                                   \
    const int emb_size = weight.size(1);                                       \
    dim3 block(emb_size / n_elements);                                         \
    dim3 grid(N);                                                              \
    embedding_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<int *>(a.data_ptr()),                                 \
        reinterpret_cast<element_type *>(weight.data_ptr()),                   \
        reinterpret_cast<element_type *>(o.data_ptr()), N, emb_size);          \
  }

TORCH_BINDING_EMBEDDING(f32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f32x4_pack, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f16, torch::kHalf, half, 1)
TORCH_BINDING_EMBEDDING(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_EMBEDDING(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4_pack);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8_pack);
}
