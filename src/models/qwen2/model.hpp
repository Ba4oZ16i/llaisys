#pragma once
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::models {

class Qwen2Model {
private:
    LlaisysQwen2Meta meta_;
    llaisysDeviceType_t device_;
    int device_id_;
    int ndevice_;

    LlaisysTensor in_embed_;                  // voc,hs
    LlaisysTensor out_embed_;                 //输出预测头权重voc,hs
    LlaisysTensor out_norm_w_;                //最终层RMS权重hs
    std::vector<llaisysTensor_t> attn_norm_w_; //做归一化训练的那一个 hs
    std::vector<llaisysTensor_t> attn_q_w_;   // q头乘,nh*dh hs
    std::vector<llaisysTensor_t> attn_q_b_;   // bias  nh*dh
    std::vector<llaisysTensor_t> attn_k_w_;
    std::vector<llaisysTensor_t> attn_k_b_;
    std::vector<llaisysTensor_t> attn_v_w_;
    std::vector<llaisysTensor_t> attn_v_b_;
    std::vector<llaisysTensor_t> attn_o_w_;   //输出，拼注意力头 hs nh*dh
    std::vector<llaisysTensor_t> mlp_norm_w_; //归一化  hs
    std::vector<llaisysTensor_t> mlp_gate_w_; //sw里面要训练的两个 di, hs
    std::vector<llaisysTensor_t> mlp_up_w_;   //di, hs
    std::vector<llaisysTensor_t> mlp_down_w_; //降维的， hs, di，这个转置一下就降成hs了

    std::vector<tensor_t> cache_k_;
    std::vector<tensor_t> cache_v_;
    size_t cache_len_;
public:
    Qwen2Model(LlaisysQwen2Meta meta, llaisysDeviceType_t device, int *device_id, int ndevice);
    ~Qwen2Model();
    void fillWeights(LlaisysQwen2Weights *weights);
    int64_t infer(int64_t *token_ids, size_t ntoken, float temperature, int64_t top_k, float top_p);
};


} // namespace llaisys::models