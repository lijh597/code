"""
operators_tf.py
TensorFlow version of Laplacian and Biharmonic operator computation.
Supports 4 mode names for compatibility, but internally actually only uses reverse-mode autodiff.
"""
import tensorflow as tf

def laplacian_operator(model, x, mode='reverse_reverse'):
    """
    Compute Laplacian Δf = Σ_i ∂²f/∂x_i² using TensorFlow's GradientTape.
    Supports: reverse_reverse, reverse_forward, forward_forward, forward_reverse
    (which is not quite true, because actually all are equivalent due to TensorFlow's reverse-mode autodiff.)
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            y = model(x)
        # compute first derivative inside tape2’s scope
        grad = tape1.gradient(y, x)
        # compute second derivatives (diagonal of Hessian)
        lap_terms = []
        for i in range(x.shape[-1]):
            second = tape2.gradient(grad[..., i], x)
            if second is not None:
                lap_terms.append(second[..., i])
        if len(lap_terms) == 0:
            raise RuntimeError("Failed to compute Laplacian — gradients returned None.")
        lap = tf.add_n(lap_terms)
    return tf.reduce_sum(lap)


def biharmonic_operator(model, x, mode='reverse_reverse'):
    """
    Compute Biharmonic operator Δ²f = Δ(Δf)
    """
    # First Laplacian
    lap = laplacian_operator(model, x, mode)
    # Compute Laplacian of Laplacian
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            lap2 = laplacian_operator(model, x, mode)
        grad_lap = tape1.gradient(lap2, x)
        lap_terms = []
        for i in range(x.shape[-1]):
            second = tape2.gradient(grad_lap[..., i], x)
            if second is not None:
                lap_terms.append(second[..., i])
        if len(lap_terms) == 0:
            raise RuntimeError("Failed to compute Biharmonic — gradients returned None.")
        bih = tf.add_n(lap_terms)
    return tf.reduce_sum(bih)
