#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#include "llaisys.h"
#include <cstdint>
#include <type_traits>
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->dtype() == weight->dtype(), "dtype");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "INT64");
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
    default:
      EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
