#include "../../../utils.hpp"
#include "add_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_nvidia(float *c, const float *a, const float *b,
                           size_t n) {
    constexpr bool is_half = std::is_same_v<T, half> || std::is_same_v<T, half2>;
    constexpr int pack = is_half ? sizeof(T) / sizeof(half) : sizeof(T) / sizeof(float);
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * pack;
    if (idx >= n)
        return;
    const T *ap = reinterpret_cast<const T *>(a);
    const T *bp = reinterpret_cast<const T *>(b);
    T *cp = reinterpret_cast<T *>(c);
    T av = ap[idx / pack];
    T bv = bp[idx / pack];
    if constexpr (is_half) {
        if constexpr (pack == 1) {
            cp[idx] = av + bv;
        } else {
            cp[idx / pack] = __hadd2(av, bv);
        }
    } else {
        if constexpr (pack == 1) {
            cp[idx] = av + bv;
        } else {
            cp[idx / pack] = make_float2(av.x + bv.x, av.y + bv.y);
        }
    }
}

__global__ void add_kernel_bf16(__nv_bfloat16 *c, const __nv_bfloat16 *a,
                                const __nv_bfloat16 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2bfloat16(__bfloat162float(a[idx]) +
                                  __bfloat162float(b[idx]));
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    int threads = 256;
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        constexpr int pack = 2;
        int blocks = (numel / pack + threads - 1) / threads;
        add_nvidia<float2><<<blocks, threads>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        constexpr int pack = 2;
        int blocks = (numel / pack + threads - 1) / threads;
        add_nvidia<half2><<<blocks, threads>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        int blocks = ((int)numel + threads - 1) / threads;
        add_kernel_bf16<<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia