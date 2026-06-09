import json
from pathlib import Path
from typing import Sequence

import ctypes
import safetensors

from models import LlaisysQwen2Meta, LlaisysQwen2Weights

DTYPE_MAP = {
    "float32": 13,
    "float16": 12,
    "bfloat16": 19,
}


class Qwen2:

    def __init__(self, model_path, lib, device=0):
        self.lib = lib
        model_path = Path(model_path)

        # 读 config.json
        with open(model_path / "config.json") as file:
            config = json.load(file)

        meta = LlaisysQwen2Meta()
        meta.dtype     = DTYPE_MAP[config.get("torch_dtype")]
        meta.nlayer    = config.get("num_hidden_layers")
        meta.hs        = config.get("hidden_size")
        meta.nh        = config.get("num_attention_heads")
        meta.nkvh      = config.get("num_key_value_heads")
        meta.dh        = meta.hs // meta.nh
        meta.di        = config.get("intermediate_size")
        meta.maxseq    = config.get("max_position_embeddings")
        meta.voc       = config.get("vocab_size")
        meta.epsilon   = config.get("rms_norm_eps")
        meta.theta     = config.get("rope_theta")
        meta.end_token = config.get("eos_token_id")
        self._end_token = meta.end_token

        # 创建 C++ 模型
        device_ids = (ctypes.c_int * 1)(0)

        self.model_ = lib.llaisysQwen2ModelCreate(
            ctypes.pointer(meta),
            ctypes.c_int(device),
            device_ids,
            ctypes.c_int(1),
        )

        # 拿所有权重张量的指针
        weights_p = lib.llaisysQwen2ModelWeights(self.model_)
        self.weights_ = ctypes.cast(weights_p,
                                    ctypes.POINTER(LlaisysQwen2Weights)).contents

        # 加载 safetensors 权重
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                data  = data_.get_tensor(name_)
                t_ptr = self._find(name_)
                if t_ptr is not None:
                    lib.tensorLoad(t_ptr,
                                   data.ctypes.data_as(ctypes.c_void_p))

    # ----------------------------------------------------------------

    def _find(self, name_):
        w = self.weights_

        if name_ == "model.embed_tokens.weight":
            return w.in_embed
        if name_ == "lm_head.weight":
            return w.out_embed
        if name_ == "model.norm.weight":
            return w.out_norm_w

        if not name_.startswith("model.layers."):
            return None

        parts  = name_.split(".")
        layer  = int(parts[2])
        suffix = ".".join(parts[3:])

        table = {
            "input_layernorm.weight":
                w.attn_norm_w,
            "self_attn.q_proj.weight":
                w.attn_q_w,
            "self_attn.q_proj.bias":
                w.attn_q_b,
            "self_attn.k_proj.weight":
                w.attn_k_w,
            "self_attn.k_proj.bias":
                w.attn_k_b,
            "self_attn.v_proj.weight":
                w.attn_v_w,
            "self_attn.v_proj.bias":
                w.attn_v_b,
            "self_attn.o_proj.weight":
                w.attn_o_w,
            "post_attention_layernorm.weight":
                w.mlp_norm_w,
            "mlp.gate_proj.weight":
                w.mlp_gate_w,
            "mlp.up_proj.weight":
                w.mlp_up_w,
            "mlp.down_proj.weight":
                w.mlp_down_w,
        }

        arr = table.get(suffix)
        if arr is not None:
            return arr[layer]
        return None

    # ----------------------------------------------------------------

    def generate(self, inputs, max_new_tokens=128,
                 top_k=1, top_p=1.0, temperature=1.0):
        tokens = list(inputs)

        for _ in range(max_new_tokens):
            c_tokens = (ctypes.c_int64 * len(tokens))(*tokens)
            nxt = self.lib.llaisysQwen2ModelInfer(
                self.model_, c_tokens, ctypes.c_size_t(len(tokens))
            )
            tokens.append(nxt)
            if nxt == self._end_token:
                break

        return tokens