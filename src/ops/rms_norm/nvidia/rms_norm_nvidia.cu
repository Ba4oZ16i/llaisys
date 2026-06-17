#include "../../../utils.hpp"
#include "rms_norm_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;

template <typename T>
__global__ void get_sum(float *temp_sum, const T *in, size_t m) { //  n*m
    size_t tid = threadIdx.x;
    size_t row = blockIdx.x;
    extern __shared__ char smem[];
    auto *shareMem = reinterpret_cast<float *>(smem);
    float sum = 0.0f;
    for (size_t i = tid; i < m; i += blockDim.x) {
        float val = static_cast<float>(in[row * m + i]);
        sum += val * val;
    }
    shareMem[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shareMem[tid] += shareMem[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        temp_sum[row] = shareMem[0];
    }
}

template <typename T>
__global__ void fin_rms(T *out, const T *in, const T *weights, float eps,
                        size_t n, size_t m, float *temp_sum) { // n *m
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < m) {
        out[row * m + col] = static_cast<T>(
            (static_cast<float>(in[row * m + col]) * static_cast<float>(weights[col])) /
            pow(temp_sum[row] / m + eps, 0.5));
    }
}

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, std::vector<size_t> shape) {
    size_t blockSize1 = 512;
    size_t gridSize1 = shape[0];
    size_t shared_bytes = blockSize1 * sizeof(float);
    float *temp_sum;
    cudaMalloc(&temp_sum, gridSize1 * sizeof(float));
    get_sum<T><<<gridSize1, blockSize1, shared_bytes>>>(temp_sum, in, shape[1]);
    dim3 blockSize2(32, 16);
    dim3 gridSize2((shape[1] + blockSize2.x - 1) / blockSize2.x,
                   (shape[0] + blockSize2.y - 1) / blockSize2.y);
    fin_rms<T><<<gridSize2, blockSize2>>>(out, in, weight, eps, shape[0], shape[1], temp_sum);
    cudaFree(temp_sum);
}

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, std::vector<size_t> shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, shape);
        break;
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const half *>(in),
            reinterpret_cast<const half *>(weight),
            eps, shape);
        break;
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<bf16 *>(out),
            reinterpret_cast<const bf16 *>(in),
            reinterpret_cast<const bf16 *>(weight),
            eps, shape);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia