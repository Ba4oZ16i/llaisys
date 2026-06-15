#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/linear.hpp"
#include "llaisys.h"
#include "nvidia/linear_nvidia.cuh"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    bool has_bias = (bias != nullptr);
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "all tensors must be contiguous.");
    ASSERT(!has_bias || bias->isContiguous(), "bias must be contiguous");

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           has_bias ? bias->data() : nullptr,
                           weight->dtype(), in->shape(), weight->shape(), has_bias);
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              has_bias ? bias->data() : nullptr,
                              weight->dtype(), in->shape(), weight->shape(), has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
