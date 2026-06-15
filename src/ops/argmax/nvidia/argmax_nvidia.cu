#include "../../../utils.hpp"
#include "argmax_nvidia.cuh"
#include "llaisys.h"
#include <cstddef>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;

template <typename T>
struct val_idx {
    T val;
    long idx;
};

template <typename T>
__device__ T neg_inf() {
    if constexpr (std::is_same_v<T, float>)
        return __int_as_float(0xFF800000);
    else if constexpr (std::is_same_v<T, half>)
        return __float2half(__int_as_float(0xFF800000));
    else
        return __float2bfloat16(__int_as_float(0xFF800000));
}

template <typename T>
__global__ void block_argmax(val_idx<T> *block_results, const T *vals, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    extern __shared__ char smem[];
    auto *shareMem = reinterpret_cast<val_idx<T> *>(smem);

    if (idx < numel) {
        shareMem[tid].val = vals[idx];
        shareMem[tid].idx = idx;
    } else {
        shareMem[tid].val = neg_inf<T>();
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shareMem[tid].val < shareMem[tid + offset].val) {
                shareMem[tid] = shareMem[tid + offset];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_results[blockIdx.x] = shareMem[0];
    }
}

template <typename T>
__global__ void final_argmax(long *max_idx, T *max_val,
                              const val_idx<T> *block_results, size_t num_blocks) {
    size_t tid = threadIdx.x;
    extern __shared__ char smem[];
    auto *shareMem = reinterpret_cast<val_idx<T> *>(smem);

    if (tid < num_blocks) {
        shareMem[tid] = block_results[tid];
    } else {
        shareMem[tid].val = neg_inf<T>();
        shareMem[tid].idx = -1;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shareMem[tid].val < shareMem[tid + offset].val) {
                shareMem[tid] = shareMem[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_val = shareMem[0].val;
        *max_idx = shareMem[0].idx;
    }
}

template <typename T>
void argmax_(long *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;
    size_t shared_bytes = block_size * sizeof(val_idx<T>);
    val_idx<T> *temp_block_results;
    cudaMalloc(&temp_block_results, grid_size * sizeof(val_idx<T>));
    block_argmax<<<grid_size, block_size, shared_bytes>>>(
        temp_block_results, vals, numel);
    size_t final_block_size = 1;
    while (final_block_size < grid_size) {
        final_block_size <<= 1;
    }
    size_t final_shared_bytes = final_block_size * sizeof(val_idx<T>);

    final_argmax<<<1, final_block_size, final_shared_bytes>>>(
        max_idx, max_val, temp_block_results, grid_size);
    cudaDeviceSynchronize();
    cudaFree(temp_block_results);
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<long *>(max_idx),
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<long *>(max_idx),
                       reinterpret_cast<half *>(max_val),
                       reinterpret_cast<const half *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<long *>(max_idx),
                       reinterpret_cast<bf16 *>(max_val),
                       reinterpret_cast<const bf16 *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia