#include "../../../utils.hpp"
#include "llaisys.h"
#include "sampling_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

template <typename T>
void sampling_(int64_t *idx, T *val, const T *vals, size_t numel,
               float temperature, int64_t top_k, float top_p) {
    if ((top_k <= 0 || (size_t)top_k >= numel) && top_p >= 1.0f) {
        float inv_temp = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;
        float max_val = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max:max_val)
        for (size_t i = 0; i < numel; i++) {
            float v;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                v = llaisys::utils::cast<float>(vals[i]) * inv_temp;
            } else {
                v = (vals[i]) * inv_temp;
            }
            if (v > max_val)
                max_val = v;
        }
        float sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < numel; i++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                sum += std::exp(llaisys::utils::cast<float>(vals[i]) * inv_temp - max_val);
            } else {
                sum += std::exp(vals[i] * inv_temp - max_val);
            }
        }
        float rand_val = (float)std::rand() / (float)RAND_MAX;
        float cumsum = 0;
        int64_t chosen = numel - 1;
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < numel; i++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                cumsum += std::exp(llaisys::utils::cast<float>(vals[i]) * inv_temp - max_val) * inv_sum;
            } else {
                cumsum += std::exp(vals[i] * inv_temp - max_val) * inv_sum;
            }
            if (rand_val <= cumsum) {
                chosen = i;
                break;
            }
        }
        *val = vals[chosen];
        *idx = chosen;
        return;
    }

    std::vector<float> logits(numel);
    float inv_temp = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;
    float max_val = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max:max_val)
    for (size_t i = 0; i < numel; i++) {
        float v;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            v = llaisys::utils::cast<float>(vals[i]) * inv_temp;
        } else {
            v = vals[i] * inv_temp;
        }
        logits[i] = v;
        if (v > max_val)
            max_val = v;
    }
    float sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < numel; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
#pragma omp parallel for
    for (size_t i = 0; i < numel; i++) {
        logits[i] *= inv_sum;
    }





    
    std::vector<std::pair<float, size_t>> candidates;
    if (top_k > 0 && (size_t)top_k < numel) {
        std::vector<float> copy = logits;
        auto nth = copy.begin() + (top_k - 1);
        std::nth_element(copy.begin(), nth, copy.end(), std::greater<float>());
        float threshold = *nth;
        if (threshold < 0)
            threshold = 0;
        float new_sum = 0;
        for (size_t i = 0; i < numel; i++) {
            if (logits[i] >= threshold && logits[i] > 0) {
                candidates.push_back({logits[i], i});
                new_sum += logits[i];
            }
        }
        if (new_sum > 0) {
            float inv_new_sum = 1.0f / new_sum;
            for (auto &p : candidates) {
                p.first *= inv_new_sum;
            }
        }
    }
    if (top_p < 1.0f) {
        if (candidates.empty()) {
            for (size_t i = 0; i < numel; i++) {
                if (logits[i] > 0) {
                    candidates.push_back({logits[i], i});
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  std::greater<std::pair<float, size_t>>());
        float cum = 0;
        float new_sum = 0;
        for (auto &p : candidates) {
            cum += p.first;
            if (cum > top_p) {
                p.first = 0;
            } else {
                new_sum += p.first;
            }
        }
        if (new_sum > 0) {
            float inv_new_sum = 1.0f / new_sum;
            for (auto &p : candidates) {
                if (p.first > 0) {
                    p.first *= inv_new_sum;
                }
            }
        }
    }
    float rand_val = (float)std::rand() / (float)RAND_MAX;
    float cumsum = 0;
    int64_t chosen = numel - 1;
    if (!candidates.empty()) {
        for (auto &p : candidates) {
            cumsum += p.first;
            if (rand_val <= cumsum) {
                chosen = p.second;
                break;
            }
        }
    } else {
        for (size_t i = 0; i < numel; i++) {
            cumsum += logits[i];
            if (rand_val <= cumsum) {
                chosen = i;
                break;
            }
        }
    }
    *val = vals[chosen];
    *idx = chosen;
}

namespace llaisys::ops::cpu {
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
                         reinterpret_cast<llaisys::fp16_t *>(val),
                         reinterpret_cast<const llaisys::fp16_t *>(vals),
                         numel, temperature, top_k, top_p);
    case LLAISYS_DTYPE_BF16:
        return sampling_(reinterpret_cast<int64_t *>(idx),
                         reinterpret_cast<llaisys::bf16_t *>(val),
                         reinterpret_cast<const llaisys::bf16_t *>(vals),
                         numel, temperature, top_k, top_p);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
