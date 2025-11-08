"""
TensorFlow 实现的高阶算子
"""
import tensorflow as tf
import numpy as np

def laplacian_operator(model, x, mode="reverse_reverse"):
    """
    计算拉普拉斯算子 ∇²f = Σᵢ ∂²f/∂xᵢ²
    
    支持4种自动微分模式（TensorFlow实现）
    """
    # 修复：检查x是否已经是Variable
    if not isinstance(x, tf.Variable):
        x = tf.Variable(x, trainable=True)
    
    if mode == "reverse_reverse":
        # 反向+反向模式
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            f = model(x)
            grad_f = tape.gradient(f, x)
        
        laplacian = 0
        for i in range(x.shape[1]):
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                grad_i = grad_f[:, i]
            hessian_i = tape2.gradient(grad_i, x)
            laplacian += hessian_i[:, i]
        
        return tf.reduce_sum(laplacian)
    
    elif mode == "reverse_forward":
        # 反向+前向模式
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            f = model(x)
            grad_f = tape.gradient(f, x)
        
        laplacian = 0
        for i in range(x.shape[1]):
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                grad_i_sum = tf.reduce_sum(grad_f[:, i])
            dgrad_i_dx = tape2.gradient(grad_i_sum, x)
            laplacian += dgrad_i_dx[:, i]
        
        return tf.reduce_sum(laplacian)
    
    elif mode == "forward_forward":
        # 前向+前向模式
        laplacian = 0
        for i in range(x.shape[1]):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                f = model(x)
                df_dx = tape.gradient(f, x)
                df_dxi_scalar = tf.reduce_sum(df_dx[:, i])
            
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                d2f_dx2 = tape2.gradient(df_dxi_scalar, x)
            laplacian += d2f_dx2[:, i]
        
        return tf.reduce_sum(laplacian)
    
    elif mode == "forward_reverse":
        # 前向+反向模式
        laplacian = 0
        for i in range(x.shape[1]):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                f = model(x)
                df_dx = tape.gradient(f, x)
                df_dxi_scalar = tf.reduce_sum(df_dx[:, i])
            
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                d2f_dx2 = tape2.gradient(df_dxi_scalar, x)
            laplacian += d2f_dx2[:, i]
        
        return tf.reduce_sum(laplacian)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def biharmonic_operator(model, x, mode="reverse_reverse"):
    """
    计算双调和算子 ∇⁴f = ∇²(∇²f)
    TensorFlow实现
    """
    # 修复：检查x是否已经是Variable
    if not isinstance(x, tf.Variable):
        x = tf.Variable(x, trainable=True)
    
    # 第一次拉普拉斯
    g = laplacian_operator(model, x, mode=mode)
    
    # 第二次拉普拉斯（使用reverse_reverse）
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        grad_g = tape.gradient(g, x)
    
    biharmonic = 0
    for i in range(x.shape[1]):
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            grad_i = grad_g[:, i]
            grad_i_sum = tf.reduce_sum(grad_i)
        hess_i = tape2.gradient(grad_i_sum, x)
        biharmonic += hess_i[:, i]
    
    return tf.reduce_sum(biharmonic)
