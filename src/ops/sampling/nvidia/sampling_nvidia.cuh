#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void sampling(std::byte *idx, std::byte *val, const std::byte *vals,
              size_t numel, llaisysDataType_t type,
              float temperature, int64_t top_k, float top_p);
}
