#include "../../../utils.hpp"
#include "embedding_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;
template <typename T>
__global__ void embedding_nvidia_(T *out, const int64_t *index, const T *weight,
                                  size_t numel, size_t weight_shape) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel * weight_shape)
        return;
    size_t i = idx / weight_shape,
           j = idx % weight_shape;

    out[idx] = weight[index[i] * weight_shape + j];
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_numel, size_t weight_shape) {
    int threads = 256;
    int blocks = (index_numel*weight_shape + threads - 1) / threads;
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        embedding_nvidia_<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            index_numel,weight_shape);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        embedding_nvidia_<half><<<blocks, threads>>>(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const half *>(weight),
            index_numel,weight_shape);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        embedding_nvidia_<bf16><<<blocks, threads>>>(
            reinterpret_cast<bf16 *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const bf16 *>(weight),
            index_numel,weight_shape);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia