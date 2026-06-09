#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/sampling_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void sampling(tensor_t idx, tensor_t val, tensor_t vals,
              tensor_t temperature, tensor_t top_k, tensor_t top_p) {
    CHECK_SAME_DEVICE(idx, val, vals);
    ASSERT(idx->isContiguous() && val->isContiguous() && vals->isContiguous(),
           "all tensors must be contiguous.");
    float temp = 1.0;
    if (temperature != nullptr) {
        temp = *reinterpret_cast<const float *>(temperature->data());
    }
    int64_t k = 0;
    if (top_k != nullptr) {
        k = *reinterpret_cast<const int64_t *>(top_k->data());
    }
    float p = 1.0;
    if (top_p != nullptr) {
        p = *reinterpret_cast<const float *>(top_p->data());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sampling(idx->data(), val->data(), vals->data(),
                             vals->numel(), vals->dtype(), temp, k, p);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
