#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/linear.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
           "Add: all tensors must be contiguous.");
    // ASSERT(in->shape()[0] == weight->shape()[1] && weight->shape()[1] == bias->shape()[0] , "shape");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape(), weight->shape());
    }
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape(), weight->shape());
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
