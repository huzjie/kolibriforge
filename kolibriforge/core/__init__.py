"""Zero-dependency tensor core."""
from .tensor import Tensor, tensor, zeros, ones, randn
from .ops import softmax, layernorm, relu, gelu, silu, cross_entropy, accuracy
from .attention import scaled_dot_product_attention
from .nn import Linear, Embedding

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "randn",
    "softmax", "layernorm", "relu", "gelu", "silu", "cross_entropy", "accuracy",
    "scaled_dot_product_attention", "Linear", "Embedding",
]
