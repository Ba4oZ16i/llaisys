#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/sampling_cpu.hpp"
#include "llaisys.h"
#include "nvidia/sampling_nvidia.cuh"

namespace llaisys::ops {
void sampling(tensor_t idx, tensor_t val, tensor_t vals,
              tensor_t temperature, tensor_t top_k, tensor_t top_p) {
    ASSERT(idx->isContiguous() && val->isContiguous() && vals->isContiguous(),
           "all tensors must be contiguous.");
    // 读取采样参数（可能来自 GPU，统一搬到 CPU 读）
    float temp = 0.8;
    if (temperature != nullptr) {
        auto t = temperature->to(LLAISYS_DEVICE_CPU);
        temp = *reinterpret_cast<const float *>(t->data());
    }
    int64_t k = 50;
    if (top_k != nullptr) {
        auto tk = top_k->to(LLAISYS_DEVICE_CPU);
        k = *reinterpret_cast<const int64_t *>(tk->data());
    }
    float p = 0.8;
    if (top_p != nullptr) {
        auto tp = top_p->to(LLAISYS_DEVICE_CPU);
        p = *reinterpret_cast<const float *>(tp->data());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sampling(idx->data(), val->data(), vals->data(),
                             vals->numel(), vals->dtype(), temp, k, p);
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::sampling(idx->data(), val->data(), vals->data(),
                                 vals->numel(), vals->dtype(), temp, k, p);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
