"""测试ResNet和双调和算子"""
import torch
from models import ResNet
from operators import laplacian_operator, biharmonic_operator

# 测试ResNet基本功能
print("=" * 60)
print("测试1: ResNet基本功能")
print("=" * 60)
try:
    model = ResNet(input_dim=3, hidden_dims=[64, 64], output_dim=1, activation='tanh')
    x = torch.randn(1, 3, requires_grad=True)
    output = model(x)
    print(f"✓ ResNet输出成功: {output.item():.6f}")
except Exception as e:
    print(f"✗ ResNet失败: {e}")
    import traceback
    traceback.print_exc()

# 测试ResNet的拉普拉斯算子
print("\n" + "=" * 60)
print("测试2: ResNet拉普拉斯算子")
print("=" * 60)
try:
    model = ResNet(input_dim=3, hidden_dims=[64, 64], output_dim=1, activation='tanh')
    x = torch.randn(1, 3, requires_grad=True)
    
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]
    results = {}
    
    for mode in modes:
        try:
            lap = laplacian_operator(model, x.clone(), mode=mode)
            results[mode] = lap.item()
            print(f"✓ {mode:20s}: {lap.item():.8f}")
        except Exception as e:
            print(f"✗ {mode:20s}: 失败 - {e}")
    
    # 检查一致性
    if len(results) == 4:
        values = list(results.values())
        max_diff = max(values) - min(values)
        print(f"\n一致性检查: 最大值与最小值差异 = {max_diff:.2e}")
        if max_diff < 1e-4:
            print("✓ 所有模式结果一致！")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试ResNet的双调和算子
print("\n" + "=" * 60)
print("测试3: ResNet双调和算子")
print("=" * 60)
try:
    model = ResNet(input_dim=3, hidden_dims=[64, 64], output_dim=1, activation='tanh')
    x = torch.randn(1, 3, requires_grad=True)
    
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]
    results = {}
    
    for mode in modes:
        try:
            bih = biharmonic_operator(model, x.clone(), mode=mode)
            results[mode] = bih.item()
            print(f"✓ {mode:20s}: {bih.item():.8f}")
        except Exception as e:
            print(f"✗ {mode:20s}: 失败 - {e}")
    
    # 检查一致性
    if len(results) == 4:
        values = list(results.values())
        max_diff = max(values) - min(values)
        print(f"\n一致性检查: 最大值与最小值差异 = {max_diff:.2e}")
        if max_diff < 1e-4:
            print("✓ 所有模式结果一致！")
        else:
            print(f"⚠ 存在差异，可能需要检查实现")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
