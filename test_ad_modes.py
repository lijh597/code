"""
测试4种自动微分模式实现的正确性
"""
import torch
import numpy as np
from models import MLP
from operators import laplacian_operator

def test_with_mlp_consistency():
    """使用MLP测试4种模式的一致性"""
    print("=" * 60)
    print("测试1: MLP模型 - 4种模式结果一致性检查")
    print("=" * 60)
    
    # 创建测试模型
    input_dim = 3
    model = MLP(
        input_dim=input_dim,
        hidden_dims=[32, 32],
        output_dim=1,
        activation='tanh'
    )
    
    # 创建测试输入
    torch.manual_seed(42)  # 固定随机种子
    x = torch.randn(1, input_dim, requires_grad=True)
    
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]
    results = {}
    
    print(f"测试输入维度: {input_dim}")
    print(f"输入值: {x.data.numpy()}")
    print()
    
    # 计算基准值（reverse_reverse）
    try:
        lap_ref = laplacian_operator(model, x.clone(), mode="reverse_reverse")
        results["reverse_reverse"] = lap_ref.item()
        print(f"✓ reverse_reverse: {lap_ref.item():.8f} (基准)")
    except Exception as e:
        print(f"✗ reverse_reverse 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试其他模式
    for mode in modes[1:]:
        try:
            # 每次重新创建输入以避免计算图污染
            x_test = torch.randn(1, input_dim, requires_grad=True)
            x_test.data = x.data.clone()  # 使用相同的输入值
            
            lap_value = laplacian_operator(model, x_test, mode=mode)
            results[mode] = lap_value.item()
            
            # 计算与基准的差异
            diff = abs(lap_value.item() - results["reverse_reverse"])
            rel_error = diff / (abs(results["reverse_reverse"]) + 1e-10)
            
            status = "✓" if rel_error < 1e-4 else "⚠"
            print(f"{status} {mode:20s}: {lap_value.item():.12f} "
                  f"(差异: {diff:.2e}, 相对误差: {rel_error:.2e})")
        except Exception as e:
            print(f"✗ {mode:20s} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = None
    
    # 统计结果
    print("\n" + "-" * 60)
    print("一致性分析:")
    successful_modes = [m for m in modes if results.get(m) is not None]
    if len(successful_modes) > 1:
        values = [results[m] for m in successful_modes]
        max_diff = max(values) - min(values)
        print(f"成功实现的模式数: {len(successful_modes)}/{len(modes)}")
        print(f"最大值与最小值差异: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("✓ 所有模式结果一致！")
        else:
            print("⚠ 模式间存在差异，可能需要检查实现")
    else:
        print("⚠ 只有一个模式成功，无法进行一致性检查")
    
    return results


def test_different_input_dims():
    """测试不同输入维度"""
    print("\n" + "=" * 60)
    print("测试2: 不同输入维度的稳定性")
    print("=" * 60)
    
    input_dims = [1, 2, 3, 5]
    mode = "reverse_reverse"
    
    for dim in input_dims:
        try:
            model = MLP(
                input_dim=dim,
                hidden_dims=[16, 16],
                output_dim=1,
                activation='tanh'
            )
            x = torch.randn(1, dim, requires_grad=True)
            lap_value = laplacian_operator(model, x, mode=mode)
            print(f"✓ 维度 {dim:2d}: 拉普拉斯算子值 = {lap_value.item():.8f}")
        except Exception as e:
            print(f"✗ 维度 {dim:2d}: 失败 - {e}")


def test_simple_quadratic():
    """测试简单的二次函数：f(x) = x₁² + x₂² + x₃²，其拉普拉斯算子应为常数"""
    print("\n" + "=" * 60)
    print("测试3: 简单二次函数")
    print("=" * 60)
    print("函数: f(x) = x₁² + x₂² + x₃²")
    print("理论拉普拉斯算子: 6.0 (常数)")
    print("注意: 使用神经网络近似，会有偏差")
    print("=" * 60)
    
    # 使用一个简单的线性层来近似这个函数
    # 实际上，我们需要一个能精确表示 f(x) = x·x 的网络
    # 这里我们用一个特殊的网络结构
    
    class QuadraticModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            # 创建一个能表示 x₁² + x₂² + x₃² 的网络
            # 使用两层：第一层平方，第二层求和
            self.linear = torch.nn.Linear(dim, dim, bias=False)
            # 权重设为单位矩阵，然后用ReLU²来近似平方
            
        def forward(self, x):
            # 简单的平方和
            return (x ** 2).sum(dim=1, keepdim=True)
    
    dim = 3
    model = QuadraticModel(dim)
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    
    try:
        lap_value = laplacian_operator(model, x, mode="reverse_reverse")
        theoretical = 6.0  # 2 + 2 + 2 = 6
        diff = abs(lap_value.item() - theoretical)
        print(f"测试值: {lap_value.item():.8f}")
        print(f"理论值: {theoretical:.8f}")
        print(f"差异: {diff:.2e}")
        if diff < 1e-5:
            print("✓ 数值精确匹配！")
        else:
            print(f"⚠ 存在数值误差（可能是浮点精度问题）")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_all_modes_multiple_runs():
    """多次运行测试，检查稳定性"""
    print("\n" + "=" * 60)
    print("测试4: 多次运行稳定性检查")
    print("=" * 60)
    
    input_dim = 3
    model = MLP(input_dim=input_dim, hidden_dims=[32, 32], output_dim=1, activation='tanh')
    
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]
    num_runs = 5
    
    print(f"运行 {num_runs} 次测试...")
    print()
    
    for mode in modes:
        values = []
        for i in range(num_runs):
            torch.manual_seed(42 + i)  # 每次使用不同的种子
            x = torch.randn(1, input_dim, requires_grad=True)
            try:
                lap_value = laplacian_operator(model, x, mode=mode)
                values.append(lap_value.item())
            except Exception as e:
                print(f"  ✗ {mode}: 运行 {i+1} 失败 - {e}")
                break
        
        if len(values) == num_runs:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"✓ {mode:20s}: 均值={mean_val:.8f}, 标准差={std_val:.2e}")
        else:
            print(f"✗ {mode:20s}: 未能完成所有运行")


if __name__ == "__main__":
    print("\n开始验证4种自动微分模式实现...\n")
    
    # 运行所有测试
    results = test_with_mlp_consistency()
    test_different_input_dims()
    test_simple_quadratic()
    test_all_modes_multiple_runs()
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    print("\n建议:")
    print("1. 如果某些模式失败，检查错误信息并修正实现")
    print("2. 如果数值不一致，检查前向模式的实现方式")
    print("3. 如果所有模式都成功，可以继续下一步：添加精度评估")
