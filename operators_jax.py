"""
operators_jax.py
JAX implementations of Laplacian and Biharmonic using different AD mode compositions.

Four modes by composing jacfwd/jacrev:
- reverse_reverse: jacrev(jacrev(f))  (reverse over reverse)
- reverse_forward: jacrev(jacfwd(f))  (reverse over forward)
- forward_forward: jacfwd(jacfwd(f))  (forward over forward)
- forward_reverse: jacfwd(jacrev(f))  (forward over reverse)

Computing full Hessian and take its trace (sum of diagonal).
"""

import jax
import jax.numpy as jnp
from functools import partial

# helper: ensure scalar function
def ensure_scalar_output(fn):
    def wrapped(params, x):
        y = fn(params, x)
        # if output has shape (batch, 1) return scalar per batch by summing outputs (expect batch 1)
        return jnp.sum(y)
    return wrapped


def laplacian_operator(params, apply_fn, x, mode="reverse_reverse"):
    """
    params: parameter dict for the model
    apply_fn: function(params, x) -> output (batch, out_dim). sum output to scalar
    x: jnp array shape (1, input_dim)
    mode: string
    returns: scalar Laplacian (sum over batch)
    """
    # define scalar function f(x) using given params
    def f_flat(x_flat):
        # x_flat is 1D vector of shape (input_dim,)
        x_b = x_flat[None, ...]  # make batch dim
        y = apply_fn(params, x_b)
        #sum outputs to scalar
        return jnp.sum(y)

    # choose composition
    if mode == "reverse_reverse":
        # Hessian via reverse-over-reverse: jacrev(jacrev(f))
        hess_fn = jax.jacrev(jax.jacrev(f_flat))
    elif mode == "reverse_forward":
        # reverse over forward
        hess_fn = jax.jacrev(jax.jacfwd(f_flat))
    elif mode == "forward_forward":
        hess_fn = jax.jacfwd(jax.jacfwd(f_flat))
    elif mode == "forward_reverse":
        hess_fn = jax.jacfwd(jax.jacrev(f_flat))
    else:
        raise ValueError("Unknown mode")

    # compute Hessian matrix (shape: (input_dim, input_dim))
    H = hess_fn(x.reshape(-1))
    # trace
    trace = jnp.trace(H)
    return trace  # scalar


def biharmonic_operator(params, apply_fn, x, mode="reverse_reverse"):
    """
    Compute Δ² f by first computing Laplacian g(x) = Δ f(x), then computing Laplacian of g using same mode composition.
    """
    # first define function to compute laplacian scalar
    def lap_scalar(x_flat):
        return laplacian_operator(params, apply_fn, x_flat[None, :], mode=mode)

    # now Hessian of lap_scalar and take trace
    # reuse compositions 
    if mode == "reverse_reverse":
        hess_lap = jax.jacrev(jax.jacrev(lap_scalar))
    elif mode == "reverse_forward":
        hess_lap = jax.jacrev(jax.jacfwd(lap_scalar))
    elif mode == "forward_forward":
        hess_lap = jax.jacfwd(jax.jacfwd(lap_scalar))
    elif mode == "forward_reverse":
        hess_lap = jax.jacfwd(jax.jacrev(lap_scalar))
    else:
        raise ValueError("Unknown mode")

    H2 = hess_lap(x.reshape(-1))
    return jnp.trace(H2)
