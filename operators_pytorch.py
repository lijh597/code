import torch
import time
import numpy as np

def laplacian_operator(model, x, mode="reverse_reverse"):
    """
    计算拉普拉斯算子 ∇²f = Σᵢ ∂²f/∂xᵢ²
    
    支持4种自动微分模式：
    - reverse_reverse: 反向+反向
    - reverse_forward: 反向+前向
    - forward_forward: 前向+前向
    - forward_reverse: 前向+反向
    """
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    
    if mode == "reverse_reverse":
        # 反向+反向模式
        f = model(x)
        grad_f = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        
        # 计算二阶梯度（海森矩阵对角线迹）
        laplacian = 0
        for i in range(x.shape[1]):
            grad_i = grad_f[:, i]
            hessian_i = torch.autograd.grad(
                grad_i.sum(), x, retain_graph=True, create_graph=True)[0]
            laplacian += hessian_i[:, i]
        
        return laplacian.sum()
    
    elif mode == "reverse_forward":
        # 反向+前向模式：先得到 ∂f/∂x，再对每个分量沿自身方向取一次导数
        f = model(x)
        grad_f = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        
        laplacian = 0
        for i in range(x.shape[1]):
            grad_i_sum = grad_f[:, i].sum()
            dgrad_i_dx = torch.autograd.grad(
                grad_i_sum, x, create_graph=True, retain_graph=True)[0]
            laplacian += dgrad_i_dx[:, i]
        
        return laplacian.sum()
    
    elif mode == "forward_forward":
        # 前向+前向（用反向原语模拟，在同一个 x 上两次求导）
        laplacian = 0
        for i in range(x.shape[1]):
            f = model(x)
            df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]          # ∂f/∂x
            df_dxi_scalar = df_dx[:, i].sum()                                      # 选第 i 分量
            d2f_dx2 = torch.autograd.grad(                                         # ∂²f/∂x∂xᵢ
                df_dxi_scalar, x, create_graph=True, retain_graph=True
            )[0]
            laplacian += d2f_dx2[:, i]                                             # 对角元素
        return laplacian.sum()
    
    elif mode == "forward_reverse":
        # 前向+反向（先得到分量导数，再反向一次取对角）
        laplacian = 0
        for i in range(x.shape[1]):
            f = model(x)
            df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]          # ∂f/∂x
            df_dxi_scalar = df_dx[:, i].sum()
            d2f_dx2 = torch.autograd.grad(                                         # ∂²f/∂x∂xᵢ
                df_dxi_scalar, x, retain_graph=True, create_graph=True
            )[0]
            laplacian += d2f_dx2[:, i]
        return laplacian.sum()
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: reverse_reverse, reverse_forward, forward_forward, forward_reverse")


def biharmonic_operator(model, x, mode="reverse_reverse"):
    """
    计算双调和算子 ∇⁴f = ∇²(∇²f)
    实现要点：
    1) 先按给定 mode 计算 g(x)=∇²f(x)，确保 g 对 x 有计算图(create_graph=True)。
    2) 再对 g(x) 用"反向+反向"计算 ∇²g(x) 的迹（即对角和）。
    """
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    # 第一次拉普拉斯：按指定 mode 计算 g(x)=∇²f(x)
    # 注意：确保 x 在计算前有 requires_grad=True 且连接计算图
    g = laplacian_operator(model, x, mode=mode)  # 标量，且应与 x 有依赖
    
    # 验证 g 是否连接到 x 的计算图
    if not g.requires_grad:
        # 如果 g 没有梯度，说明计算图断开，需要重新计算
        # 这种情况不应该发生，但如果发生了，我们需要重新构建
        raise RuntimeError(f"Laplacian result for mode {mode} is not connected to computation graph")
    
    # 第二次拉普拉斯：对 g(x) 再做一次"反向+反向"（迹）
    # 确保 g 可以关于 x 求导
    try:
        grad_g = torch.autograd.grad(g, x, create_graph=True, retain_graph=True)[0]  # ∇g
    except RuntimeError as e:
        # 如果无法求导，说明计算图断开
        raise RuntimeError(f"Cannot compute gradient of laplacian for mode {mode}: {e}. "
                          f"This may indicate that the laplacian_operator did not properly "
                          f"create a computation graph.")
    
    biharmonic = 0
    for i in range(x.shape[1]):
        grad_i = grad_g[:, i]
        hess_i = torch.autograd.grad(
            grad_i.sum(), x, retain_graph=True, create_graph=True
        )[0]
        biharmonic += hess_i[:, i]
    return biharmonic.sum()