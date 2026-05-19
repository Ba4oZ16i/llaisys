#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Add: all tensors must be contiguous.");

    if(in->deviceType()==LLAISYS_DEVICE_CPU){
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(),in->shape(),theta);
    }
    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(),in->shape(),theta);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
