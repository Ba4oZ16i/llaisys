#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu.hpp"
#include "llaisys.h"
#include "nvidia/swiglu_nvidia.cuh"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "isContiguous");

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops