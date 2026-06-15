#include "../../../utils.hpp"
#include "swiglu_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using bf16 = __nv_bfloat16;

template <typename T>
__global__ void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel)
        return;
    float temp = 1 + exp(-static_cast<float>(gate[idx]));
    out[idx] = static_cast<T>(
        static_cast<float>(up[idx]) * (static_cast<float>(gate[idx]) / temp));
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    size_t blockSize = 256;
    size_t gridSize = (numel + blockSize - 1) / blockSize;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float><<<gridSize, blockSize>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<half><<<gridSize, blockSize>>>(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const half *>(gate),
            reinterpret_cast<const half *>(up),
            numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<bf16><<<gridSize, blockSize>>>(
            reinterpret_cast<bf16 *>(out),
            reinterpret_cast<const bf16 *>(gate),
            reinterpret_cast<const bf16 *>(up),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia