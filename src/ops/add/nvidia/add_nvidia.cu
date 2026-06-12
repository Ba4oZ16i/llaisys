#include "add_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

__global__ void add_kernel_f32(float *c, const float *a, const float *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_kernel_f16(__half *c, const __half *a, const __half *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

__global__ void add_kernel_bf16(__nv_bfloat16 *c, const __nv_bfloat16 *a,
                                 const __nv_bfloat16 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel_f32<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel_f16<<<grid_size, block_size>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel_bf16<<<grid_size, block_size>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia