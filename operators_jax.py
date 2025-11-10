"""
JAX 实现的高阶算子
"""
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev


def laplacian_operator(model, x, mode="reverse_reverse"):
    x = jnp.asarray(x)          # x 形状: (1, d)
    xi = x[0]                   # 单个样本向量 (d,)

    def f_single(v):
        return model(v[None, :]).sum()   # 标量输出

    if mode == "reverse_reverse":
        H = jax.jacrev(jax.jacrev(f_single))(xi)
    elif mode == "reverse_forward":
        H = jax.jacfwd(jax.jacrev(f_single))(xi)
    elif mode == "forward_forward":
        H = jax.jacfwd(jax.jacfwd(f_single))(xi)
    elif mode == "forward_reverse":
        H = jax.jacrev(jax.jacfwd(f_single))(xi)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return jnp.trace(H)


def biharmonic_operator(model, x, mode="reverse_reverse"):
    x = jnp.asarray(x)
    xi = x[0]

    # 第一次拉普拉斯
    g = laplacian_operator(model, x, mode=mode)

    # 对 g 再做一次拉普拉斯（使用 RR 提取 Hessian 迹）
    def g_single(v):
        return laplacian_operator(model, v[None, :], mode=mode)

    H_g = jax.jacrev(jax.jacrev(g_single))(xi)
    return jnp.trace(H_g)

'''
def laplacian_operator(model, x, mode="reverse_reverse"):
    """
    计算拉普拉斯算子 ∇²f = Σᵢ ∂²f/∂xᵢ²
    
    支持4种自动微分模式（JAX实现）
    """
    x = jnp.array(x)
    
    if mode == "reverse_reverse":
        # 反向+反向模式
        def f(x):
            return model(x).sum()
        
        # 计算一阶梯度
        grad_f = grad(f)(x)
        
        # 计算二阶梯度（海森矩阵对角线）
        laplacian = 0
        for i in range(x.shape[1]):
            def grad_i_fn(x):
                return grad(f)(x)[:, i].sum()
            hessian_i = grad(grad_i_fn)(x)
            laplacian += hessian_i[:, i]
        
        return laplacian.sum()
    
    elif mode == "reverse_forward":
        # 反向+前向模式：先反向得到梯度，再对每个分量求导
        def f(x):
            return model(x).sum()
        
        grad_f = grad(f)(x)
        
        laplacian = 0
        for i in range(x.shape[1]):
            def grad_i_sum_fn(x):
                return grad(f)(x)[:, i].sum()
            dgrad_i_dx = grad(grad_i_sum_fn)(x)
            laplacian += dgrad_i_dx[:, i]
        
        return laplacian.sum()
    
    elif mode == "forward_forward":
        # 前向+前向模式：使用jacfwd
        def f(x):
            return model(x)
        
        laplacian = 0
        for i in range(x.shape[1]):
            # 第一次前向：得到 df/dx
            jac_f = jacfwd(f)(x)  # shape: [batch, output, input]
            # 选择第i个输入分量
            df_dxi = jac_f[:, :, i].sum()  # 标量
            
            # 第二次前向：对 df_dxi 关于 x 求导
            def df_dxi_fn(x):
                jac = jacfwd(f)(x)
                return jac[:, :, i].sum()
            
            d2f_dx2 = jacfwd(df_dxi_fn)(x)  # shape: [batch, input]
            laplacian += d2f_dx2[:, i]
        
        return laplacian.sum()
    
    elif mode == "forward_reverse":
        # 前向+反向模式：先用jacfwd，再用grad
        def f(x):
            return model(x)
        
        laplacian = 0
        for i in range(x.shape[1]):
            # 第一次前向：得到 df/dx
            def df_dxi_fn(x):
                jac = jacfwd(f)(x)
                return jac[:, :, i].sum()
            
            # 第二次反向：对 df_dxi 关于 x 求导
            d2f_dx2 = grad(df_dxi_fn)(x)
            laplacian += d2f_dx2[:, i]
        
        return laplacian.sum()
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: reverse_reverse, reverse_forward, forward_forward, forward_reverse")


def biharmonic_operator(model, x, mode="reverse_reverse"):
    """
    计算双调和算子 ∇⁴f = ∇²(∇²f)
    JAX实现
    """
    x = jnp.array(x)
    
    # 第一次拉普拉斯：按指定 mode 计算 g(x)=∇²f(x)
    g = laplacian_operator(model, x, mode=mode)
    
    # 第二次拉普拉斯：对 g(x) 用"反向+反向"计算 ∇²g(x)
    def g_fn(x):
        return laplacian_operator(model, x, mode=mode)
    
    # 计算一阶梯度
    grad_g = grad(g_fn)(x)
    
    # 计算二阶梯度（海森矩阵对角线）
    biharmonic = 0
    for i in range(x.shape[1]):
        def grad_i_fn(x):
            return grad(g_fn)(x)[:, i].sum()
        hess_i = grad(grad_i_fn)(x)
        biharmonic += hess_i[:, i]
    
    return biharmonic.sum()
    '''