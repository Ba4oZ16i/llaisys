import ctypes
from ctypes import Structure, c_void_p, c_int, c_size_t, c_int64, c_float, POINTER

class LlaisysQwen2Meta(Structure):
    _fields_=[
        ("dtype",c_int),       
        ("nlayer",c_size_t),   
        ("hs",c_size_t),       
        ("nh",c_size_t),       
        ("nkvh",c_size_t),     
        ("dh",c_size_t),       
        ("di",c_size_t),       
        ("maxseq",c_size_t),   
        ("voc",c_size_t),      
        ("epsilon",c_float),   
        ("theta",c_float),     
        ("end_token",c_int64), 
    ]
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_=[
        ("in_embed",c_void_p),
        ("out_embed",c_void_p),
        ("out_norm_w",c_void_p),
        ("attn_norm_w",POINTER(c_void_p)),
        ("attn_q_w",POINTER(c_void_p)),
        ("attn_q_b",POINTER(c_void_p)),
        ("attn_k_w",POINTER(c_void_p)),
        ("attn_k_b",POINTER(c_void_p)),
        ("attn_v_w",POINTER(c_void_p)),
        ("attn_v_b",POINTER(c_void_p)),
        ("attn_o_w",POINTER(c_void_p)),
        ("mlp_norm_w",POINTER(c_void_p)),
        ("mlp_gate_w",POINTER(c_void_p)),
        ("mlp_up_w", POINTER(c_void_p)),
        ("mlp_down_w",POINTER(c_void_p)),
    ]


def load_models(lib):
    lib.llaisysQwen2ModelCreate.argtypes=[
        POINTER(LlaisysQwen2Meta),
        c_int,                      
        POINTER(c_int),           
        c_int                       
    ]
    lib.llaisysQwen2ModelCreate.restype=c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes=[c_void_p]
    lib.llaisysQwen2ModelDestroy.restype=None

    lib.llaisysQwen2ModelWeights.argtypes=[c_void_p]
    lib.llaisysQwen2ModelWeights.restype=c_void_p

    lib.llaisysQwen2ModelInfer.argtypes=[
        c_void_p,              
        POINTER(c_int64),       
        c_size_t                    
    ]
    lib.llaisysQwen2ModelInfer.restype=c_int64
