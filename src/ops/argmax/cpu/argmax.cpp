#include "argmax.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>
#include <type_traits>

template <typename T>
void argmax_(long *max_idx, T *max_val, const T *vals, size_t numel) {
  size_t best_idx = 0;
  T best_val = vals[0];
  for (size_t i = 1; i < numel; i++) {
    float temp, best;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                  std::is_same_v<T, llaisys::fp16_t>) {
      temp = llaisys::utils::cast<float>(vals[i]);
      best = llaisys::utils::cast<float>(best_val);
    } else {
      temp = static_cast<float>(vals[i]);
      best = static_cast<float>(best_val);
    }
    if (temp > best) {
      best_idx = i;
      best_val = vals[i];
    }
  }
  *max_idx = best_idx;
  *max_val = best_val;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
  switch (type) {
  case LLAISYS_DTYPE_F32:
    return argmax_(reinterpret_cast<long *>(max_idx),
                reinterpret_cast<float *>(max_val),
                reinterpret_cast<const float *>(vals), numel);
  case LLAISYS_DTYPE_F16:
    return argmax_(reinterpret_cast<long *>(max_idx),
                reinterpret_cast<llaisys::fp16_t *>(max_val),
                reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
  case LLAISYS_DTYPE_BF16:
    return argmax_(reinterpret_cast<long *>(max_idx),
                reinterpret_cast<llaisys::bf16_t *>(max_val),
                reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
  default:
    EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }
}
} 