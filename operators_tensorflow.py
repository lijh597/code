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
    if not isinstance(x, tf.Variable):
        x = tf.Variable(x, trainable=True)
    
    if mode == "reverse_reverse":
        # 反向+反向模式
        laplacian = 0
        for i in range(x.shape[1]):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                f = model(x)
                grad_f = tape.gradient(f, x)
                grad_i_sum = tf.reduce_sum(grad_f[:, i])
            
            # 在 tape 的上下文中计算二阶梯度
            hessian_i = tape.gradient(grad_i_sum, x)
            if hessian_i is not None:
                laplacian += hessian_i[:, i]
            del tape  # 释放持久化 tape
        
        return tf.reduce_sum(laplacian)
    
    elif mode == "reverse_forward":
        # 反向+前向模式
        laplacian = 0
        for i in range(x.shape[1]):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                f = model(x)
                grad_f = tape.gradient(f, x)
                grad_i_sum = tf.reduce_sum(grad_f[:, i])
            
            dgrad_i_dx = tape.gradient(grad_i_sum, x)
            if dgrad_i_dx is not None:
                laplacian += dgrad_i_dx[:, i]
            del tape
        
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
                f2 = model(x)
                df_dx2 = tape2.gradient(f2, x)
                df_dxi_scalar2 = tf.reduce_sum(df_dx2[:, i])
            
            # 修复：使用持久化 tape3，因为需要调用两次 gradient
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch(x)
                f3 = model(x)
                df_dx3 = tape3.gradient(f3, x)
                df_dxi_scalar3 = tf.reduce_sum(df_dx3[:, i])
                # 在 tape3 的上下文中计算二阶导数
                d2f_dx2 = tape3.gradient(df_dxi_scalar3, x)
            
            if d2f_dx2 is not None:
                laplacian += d2f_dx2[:, i]
            del tape3  # 释放持久化 tape
        
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
                f2 = model(x)
                df_dx2 = tape2.gradient(f2, x)
                df_dxi_scalar2 = tf.reduce_sum(df_dx2[:, i])
            
            # 修复：使用持久化 tape3，因为需要调用两次 gradient
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch(x)
                f3 = model(x)
                df_dx3 = tape3.gradient(f3, x)
                df_dxi_scalar3 = tf.reduce_sum(df_dx3[:, i])
                d2f_dx2 = tape3.gradient(df_dxi_scalar3, x)
            
            if d2f_dx2 is not None:
                laplacian += d2f_dx2[:, i]
            del tape3  # 释放持久化 tape
        
        return tf.reduce_sum(laplacian)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def biharmonic_operator(model, x, mode="reverse_reverse"):
    """
    计算双调和算子 ∇⁴f = ∇²(∇²f)
    TensorFlow实现
    
    修复：在 tape 上下文中直接计算 g 的所有部分，确保计算图连续
    支持所有4种模式
    """
    if not isinstance(x, tf.Variable):
        x = tf.Variable(x, trainable=True)
    
    # 在 tape 的上下文中计算 g(x) = ∇²f(x) 和 ∇²g(x)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        
        # 在 tape 的上下文中计算 g
        # 由于 laplacian_operator 内部使用了 tape，我们需要在 tape 的上下文中
        # 直接计算 g 的所有部分（复制 laplacian_operator 的逻辑）
        
        # 计算第一次拉普拉斯 g = ∇²f（根据不同的 mode）
        g = 0
        
        if mode == "reverse_reverse":
            # 反向+反向模式
            for i in range(x.shape[1]):
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch(x)
                    f = model(x)
                    grad_f = inner_tape.gradient(f, x)
                    if grad_f is not None:
                        grad_i_sum = tf.reduce_sum(grad_f[:, i])
                        hessian_i = inner_tape.gradient(grad_i_sum, x)
                        if hessian_i is not None:
                            g += hessian_i[:, i]
                    del inner_tape
        
        elif mode == "reverse_forward":
            # 反向+前向模式
            for i in range(x.shape[1]):
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch(x)
                    f = model(x)
                    grad_f = inner_tape.gradient(f, x)
                    if grad_f is not None:
                        grad_i_sum = tf.reduce_sum(grad_f[:, i])
                        dgrad_i_dx = inner_tape.gradient(grad_i_sum, x)
                        if dgrad_i_dx is not None:
                            g += dgrad_i_dx[:, i]
                    del inner_tape
        
        elif mode == "forward_forward":
            # 前向+前向模式
            for i in range(x.shape[1]):
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch(x)
                    f = model(x)
                    df_dx = inner_tape.gradient(f, x)
                    if df_dx is not None:
                        df_dxi_scalar = tf.reduce_sum(df_dx[:, i])
                    
                    with tf.GradientTape() as inner_tape2:
                        inner_tape2.watch(x)
                        f2 = model(x)
                        df_dx2 = inner_tape2.gradient(f2, x)
                        if df_dx2 is not None:
                            df_dxi_scalar2 = tf.reduce_sum(df_dx2[:, i])
                        
                        # 修复：使用持久化 inner_tape3
                        with tf.GradientTape(persistent=True) as inner_tape3:
                            inner_tape3.watch(x)
                            f3 = model(x)
                            df_dx3 = inner_tape3.gradient(f3, x)
                            if df_dx3 is not None:
                                df_dxi_scalar3 = tf.reduce_sum(df_dx3[:, i])
                                d2f_dx2 = inner_tape3.gradient(df_dxi_scalar3, x)
                                if d2f_dx2 is not None:
                                    g += d2f_dx2[:, i]
                            del inner_tape3
                    del inner_tape
        
        elif mode == "forward_reverse":
            # 前向+反向模式
            for i in range(x.shape[1]):
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch(x)
                    f = model(x)
                    df_dx = inner_tape.gradient(f, x)
                    if df_dx is not None:
                        df_dxi_scalar = tf.reduce_sum(df_dx[:, i])
                    
                    with tf.GradientTape() as inner_tape2:
                        inner_tape2.watch(x)
                        f2 = model(x)
                        df_dx2 = inner_tape2.gradient(f2, x)
                        if df_dx2 is not None:
                            df_dxi_scalar2 = tf.reduce_sum(df_dx2[:, i])
                        
                        # 修复：使用持久化 inner_tape3
                        with tf.GradientTape(persistent=True) as inner_tape3:
                            inner_tape3.watch(x)
                            f3 = model(x)
                            df_dx3 = inner_tape3.gradient(f3, x)
                            if df_dx3 is not None:
                                df_dxi_scalar3 = tf.reduce_sum(df_dx3[:, i])
                                d2f_dx2 = inner_tape3.gradient(df_dxi_scalar3, x)
                                if d2f_dx2 is not None:
                                    g += d2f_dx2[:, i]
                            del inner_tape3
                    del inner_tape
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        g = tf.reduce_sum(g)
        
        # 现在在 tape 的上下文中计算 grad_g
        grad_g = tape.gradient(g, x)
        
        if grad_g is None:
            raise RuntimeError(f"Cannot compute gradient of laplacian for mode {mode}: g is not connected to computation graph")
        
        # 计算第二次拉普拉斯 ∇²g（使用 reverse_reverse 模式）
        biharmonic = 0
        for i in range(x.shape[1]):
            grad_i_sum = tf.reduce_sum(grad_g[:, i])
            hess_i = tape.gradient(grad_i_sum, x)
            if hess_i is not None:
                biharmonic += hess_i[:, i]
        del tape
    
    return tf.reduce_sum(biharmonic)
