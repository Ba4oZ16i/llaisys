#include "../../../utils.hpp"
#include "sampling_nvidia.cuh"
#include <cmath>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using bf16 = __nv_bfloat16;
struct val_idx {
    float val;
    size_t idx;
};
__device__ void heap_adjust(val_idx *heap, size_t top_k, size_t pos) {
    val_idx tmp = heap[pos];
    size_t child;
    while ((child = 2 * pos + 1) < top_k) {
        if (child + 1 < top_k && heap[child + 1].val < heap[child].val) {
            child++;
        }
        if (tmp.val <= heap[child].val)
            break;
        heap[pos] = heap[child];
        pos = child;
    }
    heap[pos] = tmp;
}
__device__ void heap_insert(val_idx *heap, size_t top_k, val_idx element) {
    if (element.val < heap[0].val)
        return;
    heap[0] = element;
    heap_adjust(heap, top_k, 0);
}

template <typename T>
__global__ void block_max(float *temp_block_max, const T *vals, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    extern __shared__ char smem[];
    float *shareMem = reinterpret_cast<float *>(smem);

    if (idx < numel) {
        shareMem[tid] = static_cast<float>(vals[idx]);
    } else {
        shareMem[tid] = -INFINITY;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shareMem[tid] < shareMem[tid + offset]) {
                shareMem[tid] = shareMem[tid + offset];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        temp_block_max[blockIdx.x] = shareMem[0];
    }
}
__global__ void final_max(float *max_val, const float *temp_block_max, size_t numel) {
    size_t tid = threadIdx.x;
    extern __shared__ char smem[];
    float *shareMem = reinterpret_cast<float *>(smem);

    if (tid < numel) {
        shareMem[tid] = temp_block_max[tid];
    } else {
        shareMem[tid] = -INFINITY;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shareMem[tid] < shareMem[tid + offset]) {
                shareMem[tid] = shareMem[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_val = shareMem[0];
    }
}

template <typename T>
__global__ void block_heap(val_idx *block_topk, const T *vals, size_t numel,
                           float *max_val, float temperature, size_t top_k) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ char smem[];
    val_idx *shareMem = reinterpret_cast<val_idx *>(smem);
    val_idx *block_heap = shareMem + blockDim.x * top_k;
    val_idx *thread_heap = &shareMem[tid * top_k];
    for (size_t i = 0; i < top_k; i++) {
        thread_heap[i].val = -INFINITY;
        thread_heap[i].idx = -1;
    }

    for (size_t i = idx; i < numel; i += gridDim.x * blockDim.x) {
        float val = (static_cast<float>(vals[i]) - *max_val) * temperature;
        heap_insert(thread_heap, top_k, val_idx{val, i});
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            block_heap[i] = thread_heap[i];
        }
        for (int i = 1; i < blockDim.x; i++) {
            for (int j = 0; j < top_k; j++) {
                heap_insert(block_heap, top_k, shareMem[i * top_k + j]);
            }
        }
        for (size_t i = 0; i < top_k; i++) {
            block_topk[blockIdx.x * top_k + i] = block_heap[i];
        }
    }
}

__global__ void fin_heap(val_idx *fin_topk, val_idx *block_topk, size_t numel, size_t top_k) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ char smem[];
    val_idx *shareMem = reinterpret_cast<val_idx *>(smem);
    val_idx *block_heap = shareMem + blockDim.x * top_k;
    val_idx *thread_heap = &shareMem[tid * top_k];
    for (size_t i = 0; i < top_k; i++) {
        thread_heap[i].val = -INFINITY;
        thread_heap[i].idx = -1;
    }

    for (size_t i = idx; i < numel; i += blockDim.x) {
        heap_insert(thread_heap, top_k, block_topk[i]);
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            block_heap[i] = thread_heap[i];
        }
        for (int i = 1; i < blockDim.x; i++) {
            for (int j = 0; j < top_k; j++) {
                heap_insert(block_heap, top_k, shareMem[i * top_k + j]);
            }
        }
        for (size_t i = 0; i < top_k; i++) {
            fin_topk[i] = block_heap[i];
        }
    }
}

__global__ void softmax(val_idx *ele, size_t top_k) {
    size_t tid = threadIdx.x;
    extern __shared__ char smem[];
    float *shareMem = reinterpret_cast<float *>(smem);
    if (tid >= top_k) {
        shareMem[tid] = 0.0f;
    } else {
        shareMem[tid] = expf(ele[tid].val);
    }
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shareMem[tid] += shareMem[tid + offset];
        }
        __syncthreads();
    }
    float sum = shareMem[0] + 1e-10f;
    if (tid < top_k) {
        ele[tid].val = expf(ele[tid].val) / sum;
    }
}
template <typename T>
__global__ void fin_top_p(int64_t *idx, T *val, const T *vals,
                          val_idx *ele, size_t top_k, float top_p) {
    for (size_t i = 0; i < top_k - 1; i++) {
        for (size_t j = 0; j < top_k - 1 - i; j++) {
            if (ele[j].val < ele[j + 1].val) {
                val_idx tmp = ele[j];
                ele[j] = ele[j + 1];
                ele[j + 1] = tmp;
            }
        }
    }
    float cumsum = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < top_k; i++) {
        cumsum += ele[i].val;
        cutoff = i + 1;
        if (cumsum >= top_p) {
            break;
        }
    }

    float new_sum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        new_sum += ele[i].val;
    }
    curandState state;
    curand_init(clock64(), threadIdx.x, 0, &state);
    float rand_val = curand_uniform(&state); // [0, 1)

    float running = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        running += ele[i].val / new_sum;
        if (rand_val <= running) {
            *idx = static_cast<int64_t>(ele[i].idx);
            *val = vals[ele[i].idx];
            return;
        }
    }
}

template <typename T>
void sampling_(int64_t *idx, T *val, const T *vals, size_t numel,
               float temperature, int64_t top_k, float top_p) {
    size_t blockSize1 = 256;
    size_t gridSize1 = (numel + blockSize1 - 1) / blockSize1;
    size_t shared_bytes = blockSize1 * sizeof(float);
    float *temp_block_max;
    cudaMalloc(&temp_block_max, gridSize1 * sizeof(float));
    block_max<T><<<gridSize1, blockSize1, shared_bytes>>>(
        temp_block_max, vals, numel);
    size_t blockSize2 = 1;
    while (blockSize2 < gridSize1) {
        blockSize2 <<= 1;
    }
    shared_bytes = blockSize2 * sizeof(float);
    float *max_val;
    cudaMalloc(&max_val, sizeof(float));
    final_max<<<1, blockSize2, shared_bytes>>>(
        max_val, temp_block_max, gridSize1);

    size_t blockSize3 = 32;
    size_t gridSize3 = (blockSize3 - 1 + numel) / blockSize3;
    shared_bytes = (blockSize3 + 1) * top_k * sizeof(val_idx);
    val_idx *block_topk;
    cudaMalloc(&block_topk, gridSize3 * top_k * sizeof(val_idx));
    block_heap<T><<<gridSize3, blockSize3, shared_bytes>>>(
        block_topk, vals, numel, max_val, 1 / temperature, top_k);
    val_idx *fin_topk;
    cudaMalloc(&fin_topk, top_k * sizeof(val_idx));
    size_t blockSize4 = 32;
    shared_bytes = (blockSize4 + 1) * top_k * sizeof(val_idx);
    fin_heap<<<1, blockSize4, shared_bytes>>>(
        fin_topk, block_topk, gridSize3 * top_k, top_k);
    shared_bytes = 256 * sizeof(float);
    softmax<<<1, 256, shared_bytes>>>(fin_topk, top_k);
    fin_top_p<T><<<1, 1>>>(idx, val, vals, fin_topk, top_k, top_p);
    cudaFree(temp_block_max);
    cudaFree(max_val);
    cudaFree(block_topk);
    cudaFree(fin_topk);
}

namespace llaisys::ops::nvidia {
void sampling(std::byte *idx, std::byte *val, const std::byte *vals,
              size_t numel, llaisysDataType_t type,
              float temperature, int64_t top_k, float top_p) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return sampling_(reinterpret_cast<int64_t *>(idx),
                         reinterpret_cast<float *>(val),
                         reinterpret_cast<const float *>(vals),
                         numel, temperature, top_k, top_p);
    case LLAISYS_DTYPE_F16:
        return sampling_(reinterpret_cast<int64_t *>(idx),
                         reinterpret_cast<half *>(val),
                         reinterpret_cast<const half *>(vals),
                         numel, temperature, top_k, top_p);
    case LLAISYS_DTYPE_BF16:
        return sampling_(reinterpret_cast<int64_t *>(idx),
                         reinterpret_cast<bf16 *>(val),
                         reinterpret_cast<const bf16 *>(vals),
                         numel, temperature, top_k, top_p);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
