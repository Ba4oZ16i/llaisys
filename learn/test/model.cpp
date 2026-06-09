#include "model.hpp"

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       int *device_ids,
                       int ndevice)
    : meta_(meta), device_(device), device_id_(device_ids[0]) {
    in_embed_.tensor = Tensor::create({meta_.voc, meta_.hs},
                                      meta_.dtype, device_, device_id_);
    out_embed_.tensor = Tensor::create({meta_.voc, meta_.hs},
                                       meta_.dtype, device_, device_id_);
    out_norm_w_.tensor = Tensor::create({meta_.hs},
                                        meta_.dtype, device_, device_id_);
    for (size_t i = 0; i < meta_.nlayer; i++) {

        attn_norm_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        attn_q_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.nh * meta_.dh, meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        attn_q_b_.push_back(
            new LlaisysTensor{Tensor::create({meta_.nh * meta_.dh},
                                             meta_.dtype, device_, device_id_)});
        attn_k_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh, meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        attn_k_b_.push_back(
            new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh},
                                             meta_.dtype, device_, device_id_)});
        attn_v_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh, meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        attn_v_b_.push_back(
            new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh},
                                             meta_.dtype, device_, device_id_)});
        attn_o_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.hs, meta_.nh * meta_.dh},
                                             meta_.dtype, device_, device_id_)});
        mlp_norm_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        mlp_gate_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.di, meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        mlp_up_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.di, meta_.hs},
                                             meta_.dtype, device_, device_id_)});
        mlp_down_w_.push_back(
            new LlaisysTensor{Tensor::create({meta_.hs, meta_.di},
                                             meta_.dtype, device_, device_id_)});
        cache_k_.push_back(
            Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh},
                           meta_.dtype, device_, device_id_));
        cache_v_.push_back(
            Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh},
                           meta_.dtype, device_, device_id_));
    }
}

Qwen2Model::~Qwen2Model() {
    for (auto *p : attn_norm_w_)
        delete p;
    for (auto *p : attn_q_w_)
        delete p;
    for (auto *p : attn_q_b_)
        delete p;
    for (auto *p : attn_k_w_)
        delete p;
    for (auto *p : attn_k_b_)
        delete p;
    for (auto *p : attn_v_w_)
        delete p;
    for (auto *p : attn_v_b_)
        delete p;
    for (auto *p : attn_o_w_)
        delete p;
    for (auto *p : mlp_norm_w_)
        delete p;
    for (auto *p : mlp_gate_w_)
        delete p;
    for (auto *p : mlp_up_w_)
        delete p;
    for (auto *p : mlp_down_w_)
        delete p;
}

void Qwen2Model::fillWeights(LlaisysQwen2Weights *weights) {
    weights->in_embed = &in_embed_;
    weights->out_embed = &out_embed_;
    weights->out_norm_w = &out_norm_w_;
    weights->attn_norm_w = attn_norm_w_.data();
    weights->attn_q_w = attn_q_w_.data();
    weights->attn_q_b = attn_q_b_.data();
    weights->attn_k_w = attn_k_w_.data();
    weights->attn_k_b = attn_k_b_.data();
    weights->attn_v_w = attn_v_w_.data();
    weights->attn_v_b = attn_v_b_.data();
    weights->attn_o_w = attn_o_w_.data();
    weights->mlp_norm_w = mlp_norm_w_.data();
    weights->mlp_gate_w = mlp_gate_w_.data();
    weights->mlp_up_w = mlp_up_w_.data();
    weights->mlp_down_w = mlp_down_w_.data();
}

} // namespace llaisys::models
