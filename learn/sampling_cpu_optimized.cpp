#include "sampling_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
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

    // ---- fast path: 无过滤，直接从 logits 采样，避免分配大数组 ----
    if ((top_k <= 0 || (size_t)top_k >= numel) && top_p >= 1.0f) {
        float inv_temp = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;

        float max_val = -std::numeric_limits<float>::infinity();
        #pragma omp parallel for reduction(max:max_val)
        for (size_t i = 0; i < numel; i++) {
            float v = llaisys::utils::cast<float>(vals[i]) * inv_temp;
            if (v > max_val) max_val = v;
        }

        float sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < numel; i++) {
            sum += std::exp(llaisys::utils::cast<float>(vals[i]) * inv_temp - max_val);
        }

        float rand_val = (float)std::rand() / (float)RAND_MAX;
        float cumsum = 0;
        int64_t chosen = numel - 1;
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < numel; i++) {
            cumsum += std::exp(llaisys::utils::cast<float>(vals[i]) * inv_temp - max_val) * inv_sum;
            if (rand_val <= cumsum) {
                chosen = i;
                break;
            }
        }
        *val = vals[chosen];
        *idx = chosen;
        return;
    }

    // ---- 通用路径：temperature → softmax → top-k → top-p → 随机采样 ----
    std::vector<float> logits(numel);
    float inv_temp = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;

    // Pass 1: temperature + 找最大值（合并为一个并行循环）
    float max_val = -std::numeric_limits<float>::infinity();
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 0; i < numel; i++) {
        float v = llaisys::utils::cast<float>(vals[i]) * inv_temp;
        logits[i] = v;
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp + sum（合并为一次并行循环）
    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < numel; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }

    // Pass 3: 归一化
    float inv_sum = 1.0f / sum;
    #pragma omp parallel for
    for (size_t i = 0; i < numel; i++) {
        logits[i] *= inv_sum;
    }

    // 用于存放最终候选 token 的 (概率, 索引) 列表
    std::vector<std::pair<float, size_t>> candidates;

    if (top_k > 0 && (size_t)top_k < numel) {
        // ---- top-k: nth_element O(n) 找阈值，不对全量排序 ----
        std::vector<float> copy = logits;
        auto nth = copy.begin() + (top_k - 1);
        std::nth_element(copy.begin(), nth, copy.end(), std::greater<float>());
        float threshold = *nth;
        if (threshold < 0) threshold = 0;  // 防止全0时阈值变负

        // 收集 ≥阈值的候选
        float new_sum = 0;
        for (size_t i = 0; i < numel; i++) {
            if (logits[i] >= threshold && logits[i] > 0) {
                candidates.push_back({logits[i], i});
                new_sum += logits[i];
            }
        }

        // 重新归一化
        if (new_sum > 0) {
            float inv_new_sum = 1.0f / new_sum;
            for (auto &p : candidates) {
                p.first *= inv_new_sum;
            }
        }
    }

    // ---- top-p: 只在 candidates 上排序（通常是 top_k 筛出的几十个，不再是 15 万） ----
    if (top_p < 1.0f) {
        if (candidates.empty()) {
            // 没有 top_k 预筛，需要从全部 logits 构建候选
            for (size_t i = 0; i < numel; i++) {
                if (logits[i] > 0) {
                    candidates.push_back({logits[i], i});
                }
            }
        }

        // 按概率降序排列（只排几十个候选，极快）
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

        // 重新归一化
        if (new_sum > 0) {
            float inv_new_sum = 1.0f / new_sum;
            for (auto &p : candidates) {
                if (p.first > 0) {
                    p.first *= inv_new_sum;
                }
            }
        }
    }

    // ---- 随机采样 ----
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
        // 没有候选（极端情况），从全量里退避采样
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
