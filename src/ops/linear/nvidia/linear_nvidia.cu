#include "../../../utils.hpp"
#include "linear_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;
template <typename T>
__global__ void linear_nvidia_(T *re, const T *input, const T *weights, const T *bias,
                               size_t M, size_t N, size_t K,
                               bool is_bias) { // M*K  N*k  M*N  M*n
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum += static_cast<float>(input[row * K + k] * weights[col * K + k]);
        }
        if (is_bias)
            sum += static_cast<float>(bias[col]);
        re[row * N + col] = static_cast<T>(sum);
    }
}

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, std::vector<size_t> in_shape, std::vector<size_t> weight_shape,
            bool has_bias) {
    dim3 blockSize(32, 16);
    dim3 gridSize((weight_shape[0] + blockSize.x - 1) / blockSize.x,
                  (in_shape[0] + blockSize.y - 1) / blockSize.y);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_nvidia_<float><<<gridSize, blockSize>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            in_shape[0], weight_shape[0], weight_shape[1], has_bias);
        break;
    case LLAISYS_DTYPE_F16:
        linear_nvidia_<half><<<gridSize, blockSize>>>(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const half *>(in),
            reinterpret_cast<const half *>(weight),
            reinterpret_cast<const half *>(bias),
            in_shape[0], weight_shape[0], weight_shape[1], has_bias);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_nvidia_<bf16><<<gridSize, blockSize>>>(
            reinterpret_cast<bf16 *>(out),
            reinterpret_cast<const bf16 *>(in),
            reinterpret_cast<const bf16 *>(weight),
            reinterpret_cast<const bf16 *>(bias),
            in_shape[0], weight_shape[0], weight_shape[1], has_bias);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia