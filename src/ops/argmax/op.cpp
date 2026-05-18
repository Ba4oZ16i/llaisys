#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
  CHECK_SAME_DEVICE(max_idx, max_val, vals);
  ASSERT(max_idx->isContiguous() && max_val->isContiguous() &&
             vals->isContiguous(),
         "all tensors must be contiguous.");
  if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
    return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                       vals->dtype(), vals->numel());
  }
  llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
  switch (vals->deviceType()) {
  case LLAISYS_DEVICE_CPU:
    return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                       vals->dtype(), vals->numel());
  default:
    EXCEPTION_UNSUPPORTED_DEVICE;
  }
}
} // namespace llaisys::ops
