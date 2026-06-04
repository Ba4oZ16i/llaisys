#include "linear.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             std::vector<size_t> in_shape, std::vector<size_t> weight_shape,
             bool has_bias) {
    size_t cmp = 0;
    for (size_t i = 0; i < in_shape[0]; i++) {
        for (size_t j = 0; j < weight_shape[0]; j++) {
            float sum = 0;
            for (size_t k = 0; k < in_shape[1]; k++) {
                if constexpr (
                    std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    sum += llaisys::utils::cast<float>(in[i * in_shape[1] + k]) * llaisys::utils::cast<float>(weight[j * in_shape[1] + k]);
                } else {
                    sum += in[i * in_shape[1] + k] * weight[j * in_shape[1] + k];
                }
            }
            if (has_bias) {
                if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    sum += llaisys::utils::cast<float>(bias[j]);
                    out[cmp++] = llaisys::utils::cast<T>(sum);
                } else {
                    sum += bias[j];
                    out[cmp++] = sum;
                }
            } else {
                if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    out[cmp++] = llaisys::utils::cast<T>(sum);
                } else {
                    out[cmp++] = sum;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, std::vector<size_t> in_shape, std::vector<size_t> weight_shape,
            bool has_bias) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       in_shape, weight_shape, has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       in_shape, weight_shape, has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       in_shape, weight_shape, has_bias);

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops::cpu
