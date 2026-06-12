#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"
#include "nvidia/add_nvidia.cuh"

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(),
           "Add: all tensors must be contiguous.");

    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        tensor_t gpu_a = a->to(LLAISYS_DEVICE_NVIDIA);
        tensor_t gpu_b = b->to(LLAISYS_DEVICE_NVIDIA);
        tensor_t gpu_c = Tensor::create(c->shape(), c->dtype(), LLAISYS_DEVICE_NVIDIA);
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        nvidia::add(gpu_c->data(), gpu_a->data(), gpu_b->data(),
                    c->dtype(), c->numel());
        llaisys::core::context().runtime().api()->memcpy_sync(
            c->data(), gpu_c->data(),
            c->numel() * c->elementSize(),
            LLAISYS_MEMCPY_D2H);
        return;
    }

    // Non-CPU path: dispatch to the correct device
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops