"""
models_jax.py
Simple functional neural networks for JAX:
- MLP: parameters dict + apply function
- ResNet: residual blocks, implemented functionally
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any


def glorot_init(key, shape):
    """Glorot uniform initializer"""
    lim = jnp.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_mlp_params(key, input_dim: int, hidden_dims: List[int], output_dim: int) -> Dict[str, Any]:
    """
    Initialize parameters for a simple MLP.
    Returns dict: {'W0':..., 'b0':..., 'W1':..., 'b1':..., ...}
    """
    keys = jax.random.split(key, len(hidden_dims) + 1)
    params = {}
    in_dim = input_dim
    for i, h in enumerate(hidden_dims):
        Wk = glorot_init(keys[i], (in_dim, h))
        bk = jnp.zeros((h,))
        params[f"W{i}"] = Wk
        params[f"b{i}"] = bk
        in_dim = h
    # output layer
    Wout = glorot_init(keys[-1], (in_dim, output_dim))
    bout = jnp.zeros((output_dim,))
    params[f"W_out"] = Wout
    params[f"b_out"] = bout
    return params


def mlp_apply(params: Dict[str, Any], x: jnp.ndarray, activation: str = "tanh"):
    """
    Apply MLP to input x (shape [batch, input_dim]).
    Returns output shape [batch, output_dim].
    """
    act = jnp.tanh if activation == "tanh" else jax.nn.relu
    h = x
    i = 0
    while f"W{i}" in params:
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        h = act(jnp.dot(h, W) + b)
        i += 1
    # output
    out = jnp.dot(h, params["W_out"]) + params["b_out"]
    return out


# ResNet
def init_resnet_params(key, input_dim: int, hidden_dims: List[int], output_dim: int) -> Dict[str, Any]:
    """
    Initialize ResNet parameters. Implement blocks that take input and add residuals.
    hidden_dims as per-layer widths; assuming an even number so blocks of two layers
    """
    #first map input to hidden_dims[0] then a sequence of residual blocks, then output
    keys = jax.random.split(key, 2 + len(hidden_dims))
    params = {}
    # input layer
    W0 = glorot_init(keys[0], (input_dim, hidden_dims[0]))
    b0 = jnp.zeros((hidden_dims[0],))
    params["W0"] = W0
    params["b0"] = b0

    # residual blocks: each block has two dense layers
    k = 1
    for idx, h in enumerate(hidden_dims):
        #name layers as block_{idx}_a and block_{idx}_b
        Wa = glorot_init(keys[k], (h, h))
        ba = jnp.zeros((h,))
        k += 1
        Wb = glorot_init(keys[k], (h, h))
        bb = jnp.zeros((h,))
        k += 1
        params[f"block{idx}_Wa"] = Wa
        params[f"block{idx}_ba"] = ba
        params[f"block{idx}_Wb"] = Wb
        params[f"block{idx}_bb"] = bb

    # output
    Wout = glorot_init(keys[-1], (hidden_dims[-1], output_dim))
    bout = jnp.zeros((output_dim,))
    params["W_out"] = Wout
    params["b_out"] = bout
    return params


def resnet_apply(params: Dict[str, Any], x: jnp.ndarray, activation: str = "tanh"):
    act = jnp.tanh if activation == "tanh" else jax.nn.relu
    h = act(jnp.dot(x, params["W0"]) + params["b0"])
    # apply residual blocks
    num_blocks = len([k for k in params.keys() if k.startswith("block")]) // 4
    for idx in range(num_blocks):
        Wa = params[f"block{idx}_Wa"]
        ba = params[f"block{idx}_ba"]
        Wb = params[f"block{idx}_Wb"]
        bb = params[f"block{idx}_bb"]
        # two-layer block
        tmp = act(jnp.dot(h, Wa) + ba)
        tmp = act(jnp.dot(tmp, Wb) + bb)
        h = h + tmp  # residual
    out = jnp.dot(h, params["W_out"]) + params["b_out"]
    return out
