#pragma once
#include "llaisys.h"
#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type,  std::vector<size_t> shape,float theta);
}