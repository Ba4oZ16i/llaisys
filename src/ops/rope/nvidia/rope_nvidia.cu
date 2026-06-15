#include "../../../utils.hpp"
#include "rope_nvidia.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;

template <typename T>
__global__ void rope_(T *out, const T *in, const int64_t *pos_ids,
                      size_t nhead, size_t dh, float theta) {
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x;
    float p = static_cast<float>(pos_ids[idx / nhead]);
    for (size_t i = tid; i < dh / 2; i += blockDim.x) { // 感觉这个地方不对，如果blockDim远超dh,很多线程其实是无用的
        float agri = p / pow(theta, 2.0 * i / dh);
        float a = static_cast<float>(in[idx * dh + i]);
        float b = static_cast<float>(in[idx * dh + i + dh / 2]);
        out[idx * dh + i] = static_cast<T>(
            a * cos(agri) - b * sin(agri));
        out[idx * dh + i + dh / 2] = static_cast<T>(
            b * cos(agri) + a * sin(agri));
    }
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, std::vector<size_t> shape, float theta) {
    size_t blockSize = 128;
    size_t gridSize = shape[0] * shape[1];
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float><<<gridSize, blockSize>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            shape[1], shape[2], theta);
    case LLAISYS_DTYPE_F16:
        return rope_<half><<<gridSize, blockSize>>>(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const half *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            shape[1], shape[2], theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<bf16><<<gridSize, blockSize>>>(
            reinterpret_cast<bf16 *>(out),
            reinterpret_cast<const bf16 *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            shape[1], shape[2], theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
