"""测试所有框架是否安装成功"""
print("测试框架导入...")

# 测试 PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch 未安装: {e}")

# 测试 TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow 未安装: {e}")

# 测试 JAX
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    print(f"✓ JAX {jax.__version__}")
    print(f"✓ Flax 已安装")
except ImportError as e:
    print(f"✗ JAX/Flax 未安装: {e}")

print("\n测试完成！")
