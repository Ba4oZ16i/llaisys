#pragma once

#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"

#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device,
               int *device_ids, int ndevice);
    ~Qwen2Model();
    void fillWeights(LlaisysQwen2Weights *weights);
    int64_t infer(int64_t *token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta meta_;
    llaisysDeviceType_t device_;
    int device_id_;

    LlaisysTensor in_embed_;    // [voc, hs]
    LlaisysTensor out_embed_;   // [voc, hs]
    LlaisysTensor out_norm_w_;  // [hs]
    std::vector<llaisysTensor_t> attn_norm_w_;  // [hs]
    std::vector<llaisysTensor_t> attn_q_w_;     // [nh*dh, hs]
    std::vector<llaisysTensor_t> attn_q_b_;     // [nh*dh]
    std::vector<llaisysTensor_t> attn_k_w_;     // [nkvh*dh, hs]
    std::vector<llaisysTensor_t> attn_k_b_;     // [nkvh*dh]
    std::vector<llaisysTensor_t> attn_v_w_;     // [nkvh*dh, hs]
    std::vector<llaisysTensor_t> attn_v_b_;     // [nkvh*dh]
    std::vector<llaisysTensor_t> attn_o_w_;     // [hs, nh*dh]
    std::vector<llaisysTensor_t> mlp_norm_w_;   // [hs]
    std::vector<llaisysTensor_t> mlp_gate_w_;   // [di, hs]
    std::vector<llaisysTensor_t> mlp_up_w_;     // [di, hs]
    std::vector<llaisysTensor_t> mlp_down_w_;   // [hs, di]

    std::vector<tensor_t> cache_k_;
    std::vector<tensor_t> cache_v_;
    size_t cache_len_ = 0;
};

} // namespace llaisys::models
