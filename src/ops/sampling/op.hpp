#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void sampling(tensor_t idx, tensor_t val, tensor_t vals,
              tensor_t temperature, tensor_t top_k, tensor_t top_p);
}
