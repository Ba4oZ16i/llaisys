#include "model.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "llaisys/tensor.h"

namespace llaisys::models {
Qwen2Model::Qwen2Model(LlaisysQwen2Meta meta, llaisysDeviceType_t device, int *device_id, int ndevice)
    : meta_(meta), device_(device), device_id_(device_id[0]), ndevice_(ndevice) {
    in_embed_.tensor = Tensor::create({meta.voc, meta.hs}, meta.dtype, device, device_id[0]);
    out_embed_.tensor = Tensor::create({meta.voc, meta.hs}, meta.dtype, device, device_id[0]);
    out_norm_w_.tensor = Tensor::create({meta.hs}, meta.dtype, device, device_id[0]);
    cache_len_ = 0;
    for (size_t i = 0; i < meta.nlayer; i++) {
        attn_norm_w_.push_back(new LlaisysTensor{Tensor::create({meta_.hs}, meta_.dtype, device_, device_id[0])});
        attn_q_w_.push_back(new LlaisysTensor{Tensor::create({meta_.nh * meta_.dh, meta_.hs}, meta_.dtype, device_, device_id[0])});
        attn_q_b_.push_back(new LlaisysTensor{Tensor::create({meta_.nh * meta_.dh}, meta_.dtype, device_, device_id[0])});
        attn_k_w_.push_back(new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh, meta_.hs}, meta_.dtype, device_, device_id[0])});
        attn_k_b_.push_back(new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id[0])});
        attn_v_w_.push_back(new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh, meta_.hs}, meta_.dtype, device_, device_id[0])});
        attn_v_b_.push_back(new LlaisysTensor{Tensor::create({meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id[0])});
        cache_k_.push_back(Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id[0]));
        cache_v_.push_back(Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id[0]));
        attn_o_w_.push_back(new LlaisysTensor{Tensor::create({meta_.hs, meta_.nh * meta_.dh}, meta_.dtype, device_, device_id[0])});
        mlp_norm_w_.push_back(new LlaisysTensor{Tensor::create({meta_.hs}, meta_.dtype, device_, device_id[0])});
        mlp_gate_w_.push_back(new LlaisysTensor{Tensor::create({meta_.di, meta_.hs}, meta_.dtype, device_, device_id[0])});
        mlp_up_w_.push_back(new LlaisysTensor{Tensor::create({meta_.di, meta_.hs}, meta_.dtype, device_, device_id[0])});
        mlp_down_w_.push_back(new LlaisysTensor{Tensor::create({meta_.hs, meta_.di}, meta_.dtype, device_, device_id[0])});
    }
}
Qwen2Model::~Qwen2Model() {
    for (LlaisysTensor *p : attn_norm_w_)
        delete p;
    for (LlaisysTensor *p : attn_q_w_)
        delete p;
    for (LlaisysTensor *p : attn_q_b_)
        delete p;
    for (LlaisysTensor *p : attn_k_w_)
        delete p;
    for (LlaisysTensor *p : attn_k_b_)
        delete p;
    for (LlaisysTensor *p : attn_v_w_)
        delete p;
    for (LlaisysTensor *p : attn_v_b_)
        delete p;
    for (LlaisysTensor *p : attn_o_w_)
        delete p;
    for (LlaisysTensor *p : mlp_norm_w_)
        delete p;
    for (LlaisysTensor *p : mlp_gate_w_)
        delete p;
    for (LlaisysTensor *p : mlp_up_w_)
        delete p;
    for (LlaisysTensor *p : mlp_down_w_)
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

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    if (ntoken <= cache_len_) {
        cache_len_ = 0;  // 新对话，清空 KV Cache
    }
    size_t new_token_num = ntoken - cache_len_;
    tensor_t new_token_ids = Tensor::create({new_token_num}, LLAISYS_DTYPE_I64, device_, device_id_);
    std::memcpy(new_token_ids->data(), token_ids + cache_len_, new_token_num * sizeof(int64_t));
    tensor_t new_token_input = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
    llaisys::ops::embedding(new_token_input, new_token_ids, in_embed_.tensor);

    for (size_t i = 0; i < meta_.nlayer; i++) {
        tensor_t normed = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::rms_norm(normed, new_token_input, attn_norm_w_[i]->tensor, meta_.epsilon);

        tensor_t q_temp = Tensor::create({new_token_num, meta_.nh * meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(q_temp, normed, attn_q_w_[i]->tensor, attn_q_b_[i]->tensor);
        tensor_t q = q_temp->view({new_token_num, meta_.nh, meta_.dh});

        tensor_t k_temp = Tensor::create({new_token_num, meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(k_temp, normed, attn_k_w_[i]->tensor, attn_k_b_[i]->tensor);
        tensor_t k = k_temp->view({new_token_num, meta_.nkvh, meta_.dh});

        tensor_t v_temp = Tensor::create({new_token_num, meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(v_temp, normed, attn_v_w_[i]->tensor, attn_v_b_[i]->tensor);
        tensor_t v = v_temp->view({new_token_num, meta_.nkvh, meta_.dh});

        tensor_t q_rope = Tensor::create({new_token_num, meta_.nh, meta_.dh}, meta_.dtype, device_, device_id_);
        tensor_t k_rope = Tensor::create({new_token_num, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id_);
        tensor_t pos_ids = Tensor::create({new_token_num}, LLAISYS_DTYPE_I64, device_, device_id_);
        int64_t *p = reinterpret_cast<int64_t *>(pos_ids->data());
        for (size_t j = 0; j < new_token_num; j++) {
            p[j] = static_cast<int64_t>(cache_len_ + j);
        }
        llaisys::ops::rope(q_rope, q, pos_ids, meta_.theta);
        llaisys::ops::rope(k_rope, k, pos_ids, meta_.theta);

        size_t elem_size = k_rope->elementSize();
        std::memcpy(cache_k_[i]->data() + cache_len_ * meta_.nkvh * meta_.dh * elem_size,
                    k_rope->data(),
                    new_token_num * meta_.nkvh * meta_.dh * elem_size);
        std::memcpy(cache_v_[i]->data() + cache_len_ * meta_.nkvh * meta_.dh * elem_size,
                    v->data(),
                    new_token_num * meta_.nkvh * meta_.dh * elem_size);

        tensor_t cache_k = cache_k_[i]->slice(0, 0, ntoken);
        tensor_t cache_v = cache_v_[i]->slice(0, 0, ntoken);
        tensor_t attn_out = Tensor::create({new_token_num, meta_.nh, meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::self_attention(attn_out, q_rope, cache_k, cache_v, 1.0 / std::pow(meta_.dh, 0.5));
        attn_out = attn_out->view({new_token_num, meta_.nh * meta_.dh});
        tensor_t attn_out_fin = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(attn_out_fin, attn_out, attn_o_w_[i]->tensor, nullptr);

        tensor_t new_token_input1 = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::add(new_token_input1, new_token_input, attn_out_fin);

        tensor_t normed1 = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::rms_norm(normed1, new_token_input1, mlp_norm_w_[i]->tensor, meta_.epsilon);

        tensor_t gate = Tensor::create({new_token_num, meta_.di}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(gate, normed1, mlp_gate_w_[i]->tensor, nullptr);
        tensor_t swiglu_out = Tensor::create({new_token_num, meta_.di}, meta_.dtype, device_, device_id_);
        tensor_t up = Tensor::create({new_token_num, meta_.di}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(up, normed1, mlp_up_w_[i]->tensor, nullptr);
        llaisys::ops::swiglu(swiglu_out, gate, up);

        tensor_t mlp_out = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(mlp_out, swiglu_out, mlp_down_w_[i]->tensor, nullptr);

        tensor_t new_token_input2 = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::add(new_token_input2, new_token_input1, mlp_out);
        new_token_input = new_token_input2;
    }
    tensor_t fin_norm = Tensor::create({new_token_num, meta_.hs}, meta_.dtype, device_, device_id_);
    llaisys::ops::rms_norm(fin_norm, new_token_input, out_norm_w_.tensor, meta_.epsilon);
    tensor_t re_voc = Tensor::create({new_token_num, meta_.voc}, meta_.dtype, device_, device_id_);
    llaisys::ops::linear(re_voc, fin_norm, out_embed_.tensor, nullptr);

    tensor_t re_token_voc = re_voc->slice(0, new_token_num - 1, new_token_num);
    re_token_voc = re_token_voc->view({meta_.voc});
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
    tensor_t max_val = Tensor::create({1}, meta_.dtype, device_, device_id_);
    llaisys::ops::argmax(max_idx, max_val, re_token_voc);

    cache_len_ = ntoken;
    return *reinterpret_cast<int64_t *>(max_idx->data());
}
} // namespace llaisys::models