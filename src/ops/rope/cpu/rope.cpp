#include "rope.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <type_traits>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           std::vector<size_t> shape, float theta) {
    for (size_t i = 0; i < shape[0]; i++) {
        float p = pos_ids[i];
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2] / 2; k++) {
                float agri = p / pow(theta, 2.0 * k / shape[2]);
                if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    float a = llaisys::utils::cast<float>(in[i * shape[1] * shape[2] + j * shape[2] + k]);
                    float b = llaisys::utils::cast<float>(in[i * shape[1] * shape[2] + j * shape[2] + k + shape[2] / 2]);
                    float out_a = a * cos(agri) - b * sin(agri);
                    float out_b = b * cos(agri) + a * sin(agri);
                    out[i * shape[1] * shape[2] + j * shape[2] + k] = llaisys::utils::cast<T>(out_a);
                    out[i * shape[1] * shape[2] + j * shape[2] + k + shape[2] / 2] = llaisys::utils::cast<T>(out_b);
                } else {
                    float a = in[i * shape[1] * shape[2] + j * shape[2] + k];
                    float b = in[i * shape[1] * shape[2] + j * shape[2] + k + shape[2] / 2];
                    float out_a = a * cos(agri) - b * sin(agri);
                    float out_b = b * cos(agri) + a * sin(agri);
                    out[i * shape[1] * shape[2] + j * shape[2] + k] = out_a;
                    out[i * shape[1] * shape[2] + j * shape[2] + k + shape[2] / 2] = out_b;
                }
            }
        }
    }
}
namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, std::vector<size_t> shape, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            shape, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            shape, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            shape, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}; // namespace llaisys::ops::cpu