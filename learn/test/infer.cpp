#include "model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    size_t new_start = cache_len_;
    size_t new_ntoken = ntoken - new_start;

    if (new_ntoken == 0)
        return -1;

    // ---- 2. position IDs ----
    auto pos_ids = Tensor::create({new_ntoken}, LLAISYS_DTYPE_I64, device_, device_id_);
    int64_t *pos_data = reinterpret_cast<int64_t *>(pos_ids->data());
    for (size_t i = 0; i < new_ntoken; i++)
        pos_data[i] = static_cast<int64_t>(new_start + i);

    // ---- 3. Embedding ----
    auto input_tok = Tensor::create({new_ntoken}, LLAISYS_DTYPE_I64, device_, device_id_);
    std::memcpy(input_tok->data(), token_ids + new_start, new_ntoken * sizeof(int64_t));

    auto hidden = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
    llaisys::ops::embedding(hidden, input_tok, in_embed_.tensor);

    float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));

    // ---- 4. 逐层 Transformer ----
    for (size_t layer = 0; layer < meta_.nlayer; layer++) {

        // 4a. 1RMS Norm (pre-attn)
        auto normed = T
        ensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::rms_norm(normed, hidden, attn_norm_w_[layer]->tensor, meta_.epsilon);

        // 4b. Q
        auto q_2d = Tensor::create({new_ntoken, meta_.nh * meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(q_2d, normed, attn_q_w_[layer]->tensor, attn_q_b_[layer]->tensor);
        auto q = q_2d->view({new_ntoken, meta_.nh, meta_.dh});

        // 4c. K
        auto k_2d = Tensor::create({new_ntoken, meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(k_2d, normed, attn_k_w_[layer]->tensor, attn_k_b_[layer]->tensor);
        auto k = k_2d->view({new_ntoken, meta_.nkvh, meta_.dh});

        // 4d. V
        auto v_2d = Tensor::create({new_ntoken, meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(v_2d, normed, attn_v_w_[layer]->tensor, attn_v_b_[layer]->tensor);
        auto v = v_2d->view({new_ntoken, meta_.nkvh, meta_.dh});

        // 4e. RoPE
        auto q_rope = Tensor::create({new_ntoken, meta_.nh, meta_.dh}, meta_.dtype, device_, device_id_);
        auto k_rope = Tensor::create({new_ntoken, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::rope(q_rope, q, pos_ids, meta_.theta);
        llaisys::ops::rope(k_rope, k, pos_ids, meta_.theta);

        // 4f. 写入 KV Cache
        size_t total_len = new_start + new_ntoken;
        size_t elem_size = k_rope->elementSize();
        size_t k_bytes = new_ntoken * meta_.nkvh * meta_.dh * elem_size;
        size_t v_bytes = new_ntoken * meta_.nkvh * meta_.dh * elem_size;
        size_t off_k = new_start * meta_.nkvh * meta_.dh * elem_size;
        size_t off_v = new_start * meta_.nkvh * meta_.dh * elem_size;
        std::memcpy(cache_k_[layer]->data() + off_k, k_rope->data(), k_bytes);
        std::memcpy(cache_v_[layer]->data() + off_v, v->data(), v_bytes);

        // 4g. Self-Attention
        auto cache_k_view = cache_k_[layer]->slice(0, 0, total_len);
        auto cache_v_view = cache_v_[layer]->slice(0, 0, total_len);
        auto attn_out = Tensor::create({new_ntoken, meta_.nh, meta_.dh}, meta_.dtype, device_, device_id_);
        llaisys::ops::self_attention(attn_out, q_rope, cache_k_view, cache_v_view, scale);

        // 4h. Output 投影
        auto attn_2d = attn_out->view({new_ntoken, meta_.nh * meta_.dh});
        auto attn_o = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(attn_o, attn_2d, attn_o_w_[layer]->tensor, nullptr);

        // 4i. Residual
        auto hidden_new = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::add(hidden_new, hidden, attn_o);
        hidden = hidden_new;

        // 4j. RMS Norm (pre-MLP)
        auto normed2 = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::rms_norm(normed2, hidden, mlp_norm_w_[layer]->tensor, meta_.epsilon);

        // 4k. Gate & Up
        auto gate = Tensor::create({new_ntoken, meta_.di}, meta_.dtype, device_, device_id_);
        auto up = Tensor::create({new_ntoken, meta_.di}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(gate, normed2, mlp_gate_w_[layer]->tensor, nullptr);
        llaisys::ops::linear(up, normed2, mlp_up_w_[layer]->tensor, nullptr);

        // 4l. SwiGLU
        auto swiglu_out = Tensor::create({new_ntoken, meta_.di}, meta_.dtype, device_, device_id_);
        llaisys::ops::swiglu(swiglu_out, gate, up);

        // 4m. Down 投影
        auto mlp_out = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(mlp_out, swiglu_out, mlp_down_w_[layer]->tensor, nullptr);

        // 4n. Residual
        auto hidden_new2 = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::add(hidden_new2, hidden, mlp_out);
        hidden = hidden_new2;
    }

    // ---- 5. Final RMS Norm ----
    auto final_normed = Tensor::create({new_ntoken, meta_.hs}, meta_.dtype, device_, device_id_);
    llaisys::ops::rms_norm(final_normed, hidden, out_norm_w_.tensor, meta_.epsilon);

    // ---- 6. LM Head ----
    auto logits = Tensor::create({new_ntoken, meta_.voc}, meta_.dtype, device_, device_id_);
    llaisys::ops::linear(logits, final_normed, out_embed_.tensor, nullptr);

    // ---- 7. Argmax ----
    auto last_logits_2d = logits->slice(0, new_ntoken - 1, new_ntoken);
    auto last_logits_1d = last_logits_2d->view({meta_.voc});

    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
    auto max_val = Tensor::create({1}, meta_.dtype, device_, device_id_);
    llaisys::ops::argmax(max_idx, max_val, last_logits_1d);

    // ---- 8. 返回 ----
    cache_len_ = ntoken;

    int64_t next_token;
    std::memcpy(&next_token, max_idx->data(), sizeof(int64_t));
    return next_token;
}

} // namespace llaisys::models
