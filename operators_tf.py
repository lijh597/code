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
    # outer tape (tape2) to compute the second derivates
    # persistens =true, because .gradient will be called on it multiple times
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x) #gradient ops are recorded wrt x
        # Inner tape (tape1) computes the first derivative grad = ∇_x f, same as above
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            y = model(x)
        # compute first derivative inside tape2’s scope, computation grad visible to tape 2
        grad = tape1.gradient(y, x) # shape same as x
        # compute second derivatives (diagonal of Hessian)

        # collect diagonal entries of the Hessian
        lap_terms = []
        # loop over each input dimension and compute derivative of gradient component
        for i in range(x.shape[-1]):
            # Compute ∂/∂x (grad[..., i]) using tape2, if no dependency, None
            second = tape2.gradient(grad[..., i], x)
            if second is not None:
                lap_terms.append(second[..., i])

        if len(lap_terms) == 0: # error when no second derivates found at all
            raise RuntimeError("Failed to compute Laplacian — gradients returned None.")

        lap = tf.add_n(lap_terms) # Sum diagonal contributions to get the trace of Hessian each sample
    return tf.reduce_sum(lap) #return scalar (sum of lap across batch) for consistent API eith torch ver


def biharmonic_operator(model, x, mode='reverse_reverse'):
    """
    Compute Biharmonic operator Δ²f = Δ(Δf)
    First the Laplacian, then the Laplacian of that scalar result
    """
    # inner Laplacian, connected to x for further differentiation
    lap = laplacian_operator(model, x, mode)
    # compute Laplacian of Laplacian
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            # recompute laplacian inside the tapes, makes sure the operations are recorded in the current differentiation context
            lap2 = laplacian_operator(model, x, mode)
        # first derivative of laplacian
        grad_lap = tape1.gradient(lap2, x)

        # diagonal entries of Hessian(grad_lap) to get ∆(lap)
        lap_terms = []
        for i in range(x.shape[-1]):
            second = tape2.gradient(grad_lap[..., i], x)
            if second is not None:
                lap_terms.append(second[..., i])
        if len(lap_terms) == 0:
            raise RuntimeError("Failed to compute Biharmonic — gradients returned None.")
        bih = tf.add_n(lap_terms)
    return tf.reduce_sum(bih) # summed scalar value for consistency

