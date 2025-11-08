"""
统一算子接口，根据配置选择框架
"""
from config import CURRENT_FRAMEWORKS

# 默认使用第一个框架
_default_framework = CURRENT_FRAMEWORKS[0] if CURRENT_FRAMEWORKS else 'pytorch'

if _default_framework == 'pytorch':
    from operators_pytorch import laplacian_operator, biharmonic_operator
elif _default_framework == 'tensorflow':
    from operators_tensorflow import laplacian_operator, biharmonic_operator
elif _default_framework == 'jax':
    from operators_jax import laplacian_operator, biharmonic_operator
else:
    raise ValueError(f"Unknown framework: {_default_framework}")