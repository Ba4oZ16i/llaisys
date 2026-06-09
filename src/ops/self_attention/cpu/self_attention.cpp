#include "self_attention.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>
#include <math.h>
#include <type_traits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *qq, const T *kk, const T *vv,
                     float scale, std::vector<size_t> q_shape, std::vector<size_t> kv_shape) {
    #pragma omp parallel for
    for (size_t i = 0; i < q_shape[1]; i++) {
        size_t kvh = i / (q_shape[1] / kv_shape[1]);
        std::vector<float> A;
        for (size_t j = 0; j < q_shape[0]; j++) {
            for (size_t k = 0; k < kv_shape[0]; k++) {
                float re = 0;
                for (size_t l = 0; l < kv_shape[2]; l++) {
                    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                        re += llaisys::utils::cast<float>(
                                  qq[j * q_shape[1] * q_shape[2] + i * q_shape[2] + l]) *
                              llaisys::utils::cast<float>(
                                  kk[k * kv_shape[1] * kv_shape[2] + kvh * kv_shape[2] + l]);
                    } else {
                        re += qq[j * q_shape[1] * q_shape[2] + i * q_shape[2] + l] *
                              kk[k * kv_shape[1] * kv_shape[2] + kvh * kv_shape[2] + l];
                    }
                }
                A.push_back(re * scale);
            }
        }
        //得到shape[ seqlen total_len]A,单头
        for (size_t j = 0; j < q_shape[0]; j++) {
            float maxnum = -INFINITY;
            float sum = 0;
            for (size_t k = 0; k < kv_shape[0]; k++) {
                if (k > j + (kv_shape[0] - q_shape[0])) {
                    A[j * kv_shape[0] + k] = -INFINITY;
                } else {
                    if (maxnum < A[j * kv_shape[0] + k]) {
                        maxnum = A[j * kv_shape[0] + k];
                    }
                }
            }
            for (size_t k = 0; k < kv_shape[0]; k++) {
                if (k <= j + (kv_shape[0] - q_shape[0])) {
                    sum += exp(A[j * kv_shape[0] + k] - maxnum);
                }
            }
            for (size_t k = 0; k < kv_shape[0]; k++) {
                if (k <= j + (kv_shape[0] - q_shape[0])) {
                    A[j * kv_shape[0] + k] = exp(A[j * kv_shape[0] + k] - maxnum) / sum;
                } else {
                    A[j * kv_shape[0] + k] = 0;
                }
            }
        }
        for (size_t j = 0; j < q_shape[0]; j++) {
            for (size_t k = 0; k < kv_shape[2]; k++) {
                float sum = 0;
                for (size_t l = 0; l < kv_shape[0]; l++) {
                    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                        sum +=
                            A[j * kv_shape[0] + l] *
                            llaisys::utils::cast<float>(
                                vv[l * kv_shape[1] * kv_shape[2] + kvh * kv_shape[2] + k]);
                    } else {
                        sum +=
                            A[j * kv_shape[0] + l] * vv[l * kv_shape[1] * kv_shape[2] + kvh * kv_shape[2] + k];
                    }
                }
                if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    attn_val[j * q_shape[1] * q_shape[2] + i * q_shape[2] + k] =
                        llaisys::utils::cast<T>(sum);
                } else {
                    attn_val[j * q_shape[1] * q_shape[2] + i * q_shape[2] + k] = sum;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
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
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               scale, q_shape, kv_shape);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               scale, q_shape, kv_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu