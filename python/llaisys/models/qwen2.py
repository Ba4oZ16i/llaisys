import json
from pathlib import Path
from typing import Sequence

import ctypes
import safetensors

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights

DTYPE_MAP = {
    "float32":   13,
    "float16":   12,
    "bfloat16":  19,
}
class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, lib=None):
        # TODO: Implement model constructor
        if lib is None:
            lib = LIB_LLAISYS
        self.lib=lib
        model_path = Path(model_path)
        with open(model_path/"config.json")as file:
            config=json.load(file)

        meta=LlaisysQwen2Meta()
        meta.dtype=DTYPE_MAP[config.get("torch_dtype")]
        meta.nlayer=config.get("num_hidden_layers")
        meta.hs=config.get("hidden_size")
        meta.nh=config.get("num_attention_heads")
        meta.nkvh=config.get("num_key_value_heads")
        meta.dh=meta.hs//meta.nh
        meta.di=config.get("intermediate_size")
        meta.maxseq=config.get("max_position_embeddings")
        meta.voc=config.get("vocab_size")
        meta.epsilon=config.get("rms_norm_eps")
        meta.theta=config.get("rope_theta")
        meta.end_token=config.get("eos_token_id")
        device_ids=(ctypes.c_int *1)(0)

        self.end_token_ = meta.end_token
        #一个数组数量为1，然后第一个是0
        self.model_=lib.llaisysQwen2ModelCreate(
            ctypes.pointer(meta),
            ctypes.c_int(device),       
            device_ids,       
            ctypes.c_int(1)  
        )
        weights_p=lib.llaisysQwen2ModelWeights(self.model_)
        self.weights_=ctypes.cast(weights_p,ctypes.POINTER(LlaisysQwen2Weights)).contents



        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                data=data_.get_tensor(name_)
                c_p=self.p2c(name_)
                if c_p is not None:
                    lib.tensorLoad(c_p,data.data_ptr())
                

    def p2c(self, name_):
        if name_ == "model.embed_tokens.weight":
            return self.weights_.in_embed
        if name_ == "lm_head.weight":
            return self.weights_.out_embed
        if name_ == "model.norm.weight":
            return self.weights_.out_norm_w
        char  = name_.split(".")
        layer  = int(char[2])
        char_name = ".".join(char[3:])

        table = {
            "input_layernorm.weight":self.weights_.attn_norm_w,
            "self_attn.q_proj.weight":self.weights_.attn_q_w,
            "self_attn.q_proj.bias":self.weights_.attn_q_b,
            "self_attn.k_proj.weight":self.weights_.attn_k_w,
            "self_attn.k_proj.bias":self.weights_.attn_k_b,
            "self_attn.v_proj.weight":self.weights_.attn_v_w,
            "self_attn.v_proj.bias":self.weights_.attn_v_b,
            "self_attn.o_proj.weight":self.weights_.attn_o_w,
            "post_attention_layernorm.weight":self.weights_.mlp_norm_w,
            "mlp.gate_proj.weight":self.weights_.mlp_gate_w,
            "mlp.up_proj.weight":self.weights_.mlp_up_w,
            "mlp.down_proj.weight":self.weights_.mlp_down_w,
        }
        arr = table.get(char_name)
        return arr[layer]



    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function
        tokens=list(inputs)
        for i in range(max_new_tokens):
            c_tokens = (ctypes.c_int64*len(tokens))(*tokens)
            new_token = self.lib.llaisysQwen2ModelInfer(
                self.model_, c_tokens, ctypes.c_size_t(len(tokens)),
                ctypes.c_float(temperature), ctypes.c_int64(top_k), ctypes.c_float(top_p)
            )
            tokens.append(new_token)
            if new_token == self.end_token_:
                break

        return tokens
