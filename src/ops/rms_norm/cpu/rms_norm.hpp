#pragma once
#include "llaisys.h"
#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in,const std::byte *weight, float eps,
              llaisysDataType_t type, std::vector<size_t> shape);
}