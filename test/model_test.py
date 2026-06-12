import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(conversation, tokenizer, model, max_new_tokens=512, top_p=0.8, top_k=50, temperature=0.8):
    input_content = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    new_tokens = outputs[len(inputs):]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return outputs, reply


if __name__ == "__main__":
    history=[]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=4096, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = load_llaisys_model(args.model, args.device)
    print("模型已载入  你可以对话了", flush=True)
    try:
        while True:
            content = input()
            history.append({"role": "user", "content": content})
            start_time=time.time()
            _, llaisys_output = llaisys_infer(
                history,
                tokenizer,
                model,
                max_new_tokens=args.max_steps,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            end_time=time.time()
            print(f"耗时：{(end_time-start_time):.2f}")
            idx = llaisys_output.find("</think>")
            if idx != -1:
                llaisys_output = llaisys_output[idx + len("</think>"):]
    
            display = llaisys_output.strip()
            history.append({"role": "assistant", "content": llaisys_output})
            print(display, flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        del model
    
