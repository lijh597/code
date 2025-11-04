# test_components.py
import torch
from models import MLP, ResNet
from operators import laplacian_operator, biharmonic_operator

# 测试MLP模型
print("测试MLP模型...")
model = MLP(input_dim=5, hidden_dims=[64, 64], output_dim=1)
x = torch.randn(1, 5, requires_grad=True)
output = model(x)
print(f"MLP输出: {output}")

# 测试拉普拉斯算子
print("测试拉普拉斯算子...")
laplacian = laplacian_operator(model, x)
print(f"拉普拉斯算子结果: {laplacian}")

# 测试双调和算子
print("测试双调和算子...")
biharmonic = biharmonic_operator(model, x)
print(f"双调和算子结果: {biharmonic}")