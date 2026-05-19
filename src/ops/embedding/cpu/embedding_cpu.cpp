#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstddef>
#include <cstdint>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t numel, size_t weight_shape) {
    for (size_t i = 0; i < numel; i++) {
        size_t temp = i * weight_shape;
        for (size_t j = 0; j < weight_shape; j++) {
            out[temp + j] = weight[index[i] * weight_shape + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_numel, size_t weight_shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight),
                          index_numel,
                          weight_shape);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          index_numel,
                          weight_shape);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          index_numel,
                          weight_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu