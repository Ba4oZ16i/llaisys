#include "rms_norm.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>
#include <math.h>
#include <type_traits>
#include <vector>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, std::vector<size_t> shape) {
    #pragma omp parallel for
    for (size_t i = 0; i < shape[0]; i++) {
        float sum = 0;
        for (size_t j = 0; j < shape[1]; j++) {
            size_t idx = i * shape[1] + j;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float v = llaisys::utils::cast<float>(in[idx]);
                sum += v * v;
            } else {
                sum += in[idx] * in[idx];
            }
        }
        sum = std::sqrt(sum / shape[1] + eps);
        for (size_t j = 0; j < shape[1]; j++) {
            size_t idx = i * shape[1] + j;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[idx] = llaisys::utils::cast<T>(
                    llaisys::utils::cast<float>(weight[j]) * (llaisys::utils::cast<float>(in[idx]) / sum));
            } else {
                out[idx] = weight[j] * (in[idx] / sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, std::vector<size_t> shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         eps, shape);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         eps, shape);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         eps, shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu