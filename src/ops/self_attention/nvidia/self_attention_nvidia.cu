#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using bf16 = __nv_bfloat16;

template <typename T> // A: seqlen nhead total_len
__global__ void QK4A(float *temp_A, const T *Q, const T *K, float scale,
                     size_t total_len, size_t qlen, size_t nhead, size_t nkvhead, size_t dh) {
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t seq_idx = idx / nhead;
    size_t qhead = idx % nhead;
    size_t kvh = qhead / (nhead / nkvhead);
    for (int i = tid; i < total_len; i += blockDim.x) {
        float sum = 0.0f;
        if (i <= seq_idx + (total_len - qlen)) {
            for (int j = 0; j < dh; j++) {
                sum += static_cast<float>(Q[seq_idx * nhead * dh + qhead * dh + j]) *
                       static_cast<float>(K[i * nkvhead * dh + kvh * dh + j]);
            }
            temp_A[idx * total_len + i] = sum * scale;
        } else {
            temp_A[idx * total_len + i] = -INFINITY;
        }
    }
}

//            qlen nhead total_len
__global__ void casual_softmax(float *temp_A, size_t qlen, size_t nhead, size_t total_len) {
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t token_idx = idx / nhead;
    size_t head_idx = idx % nhead;
    size_t cache_len = total_len - qlen;
    size_t limit = token_idx + cache_len;
    extern __shared__ char smem[];
    float *shareMem = reinterpret_cast<float *>(smem);
    float max = -INFINITY;
    for (int i = tid; i <= limit; i += blockDim.x) {
        if (max < temp_A[token_idx * nhead * total_len + head_idx * total_len + i]) {
            max = temp_A[token_idx * nhead * total_len + head_idx * total_len + i];
        }
    }
    shareMem[tid] = max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shareMem[tid] < shareMem[tid + offset])
                shareMem[tid] = shareMem[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shareMem[blockDim.x] = shareMem[0];
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i <= limit; i += blockDim.x) {
        sum += expf(temp_A[token_idx * nhead * total_len + head_idx * total_len + i] - shareMem[blockDim.x]);
    }
    shareMem[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shareMem[tid] += shareMem[tid + offset];
        }
        __syncthreads();
    }

    for (int i = tid; i < total_len; i += blockDim.x) {
        if (i <= limit) {
            temp_A[token_idx * nhead * total_len + head_idx * total_len + i] = expf(temp_A[token_idx * nhead * total_len + head_idx * total_len + i] - shareMem[blockDim.x]) / shareMem[0];
        } else {
            temp_A[token_idx * nhead * total_len + head_idx * total_len + i] = 0;
        }
    }
}

template <typename T> //  attn_val:seqlen nhead dh    a: seqlen nhead totallen  v: totallen nkvhead dh
__global__ void AV4ATTN(T *attn_val, float *temp_A,const T *V,
                        size_t total_len, size_t nhead, size_t nkvhead, size_t dh) {
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t token_idx = idx / nhead;
    size_t head_idx = idx % nhead;
    size_t kvh = head_idx / (nhead / nkvhead);
    for (int i = tid; i < dh; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < total_len; j++) {
            sum += temp_A[token_idx * nhead * total_len + head_idx * total_len + j] *
                   static_cast<float>(V[j * nkvhead * dh + kvh * dh + i]);
        }
        attn_val[token_idx * nhead * dh + head_idx * dh + i] = static_cast<T>(sum);
    }
}

template <typename T>
void self_attention_(T *attn_val, const T *Q, const T *K, const T *V, float scale,
                     std::vector<size_t> q_shape, std::vector<size_t> kv_shape) {
    size_t blockSize1 = 128;
    size_t gridSize1 = q_shape[0] * q_shape[1];
    float *temp_A;
    cudaMalloc(&temp_A, q_shape[0] * q_shape[1] * kv_shape[0] * sizeof(float));
    QK4A<T><<<gridSize1, blockSize1>>>(temp_A, Q, K, scale, kv_shape[0], q_shape[0], q_shape[1], kv_shape[1], q_shape[2]);
    cudaDeviceSynchronize();
    size_t shareMem = (blockSize1 + 1) * sizeof(float);
    casual_softmax<<<gridSize1, blockSize1, shareMem>>>(temp_A, q_shape[0], q_shape[1], kv_shape[0]);
    cudaDeviceSynchronize();
    size_t blockSize2 = 256;
    AV4ATTN<T><<<gridSize1, blockSize2>>>(attn_val, temp_A, V, kv_shape[0], q_shape[1], kv_shape[1], kv_shape[2]);
    cudaDeviceSynchronize();
    cudaFree(temp_A);
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, std::vector<size_t> q_shape, std::vector<size_t> kv_shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               scale, q_shape, kv_shape);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<half *>(attn_val),
                               reinterpret_cast<const half *>(q),
                               reinterpret_cast<const half *>(k),
                               reinterpret_cast<const half *>(v),
                               scale, q_shape, kv_shape);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<bf16 *>(attn_val),
                               reinterpret_cast<const bf16 *>(q),
                               reinterpret_cast<const bf16 *>(k),
                               reinterpret_cast<const bf16 *>(v),
                               scale, q_shape, kv_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia