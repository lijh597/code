"""
Simple JAX tests mirroring PyTorch tests
"""

import jax
import jax.numpy as jnp
from models_jax import init_mlp_params, mlp_apply
from operators_jax import laplacian_operator

def test_consistency():
    input_dim = 3
    rng = jax.random.PRNGKey(0)
    params = init_mlp_params(rng, input_dim, [32, 32], 1)
    x = jax.random.normal(rng, (1, input_dim))
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]
    results = {}
    for mode in modes:
        val = laplacian_operator(params, mlp_apply, x, mode=mode)
        results[mode] = float(val)
        print(f"{mode}: {results[mode]:.8f}")
    vals = list(results.values())
    print("Max diff:", max(vals) - min(vals))

def test_quadratic():
    # f(x) = x1^2 + x2^2 + x3^2 -> Laplacian = 6
    input_dim = 3
    def quad_apply(params, x):
        return jnp.sum(x**2, axis=1, keepdims=True)
    params = {}  # not used
    x = jnp.array([[1.0, 2.0, 3.0]])
    val = laplacian_operator(params, quad_apply, x, mode="reverse_reverse")
    print("laplacian:", float(val), "theoretical 6.0")

if __name__ == "__main__":
    test_consistency()
    test_quadratic()
