#include "sampling_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>
template <typename T>
void sampling_(int64_t *idx, T *val, const T *vals, size_t numel,
               float temperature, int64_t top_k, float top_p) {
    std::vector<float> logits(numel);
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            logits[i] = temperature > 0 ? llaisys::utils::cast<float>(vals[i]) / temperature : llaisys::utils::cast<float>(vals[i]);
        } else {
            logits[i] = temperature > 0 ? vals[i] / temperature : vals[i];
        }
    }
    float max_val = logits[0];
    for (size_t i = 1; i < numel; i++) {
        if (logits[i] > max_val)
            max_val = logits[i];
    }
    float sum = 0;
    for (size_t i = 0; i < numel; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    for (size_t i = 0; i < numel; i++) {
        logits[i] /= sum;
    }

    if (top_k > 0 && (size_t)top_k < numel) {
        std::vector<float> sorted = logits;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        float threshold = sorted[top_k - 1];
        for (size_t i = 0; i < numel; i++) {
            if (logits[i] < threshold) {
                logits[i] = 0;
            }
        }
        float new_sum = 0;
        for (size_t i = 0; i < numel; i++)
            new_sum += logits[i];
        for (size_t i = 0; i < numel; i++)
            logits[i] /= new_sum;
    }
    if (top_p < 1) {
        std::vector<std::pair<float, size_t>> sorted_pairs;
        for (size_t i = 0; i < numel; i++) {
            sorted_pairs.push_back(std::make_pair(logits[i], i));
        }
        std::sort(sorted_pairs.begin(), sorted_pairs.end(),
                  std::greater<std::pair<float, size_t>>());
        float sum1 = 0;
        for (auto &p : sorted_pairs) {
            sum1 += p.first;
            if (sum1 > top_p) {
                p.first = 0;
            }
        }
        std::fill(logits.begin(), logits.end(), 0);
        float new_sum = 0;
        for (auto &p : sorted_pairs) {
            logits[p.second] = p.first;
            new_sum += p.first;
        }
        if (new_sum > 0) {
            for (size_t i = 0; i < numel; i++)
                logits[i] /= new_sum;
        }
    }

    float rand = (float)std::rand() / (float)RAND_MAX;
    float chosesum = 0;
    int64_t chosen = 0;
    for (size_t i = 0; i < numel; i++) {
        chosesum += logits[i];
        if (rand <= chosesum) {
            chosen = i;
            break;
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
