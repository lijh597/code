"""
JAX 实现的模型
"""
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

class MLP(nn.Module):
    """多层感知机模型（JAX/Flax实现）"""
    
    hidden_dims: list
    output_dim: int
    activation: str = 'tanh'
    
    def setup(self):
        # Flax 会把 hidden_dims 转成 tuple，这里统一转回 list 再追加输出层
        dims = list(self.hidden_dims) + [self.output_dim]
        self.layers = [nn.Dense(d) for d in dims]
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 除输出层外添加激活函数
                if self.activation.lower() == 'tanh':
                    x = jnp.tanh(x)
                else:
                    x = nn.relu(x)
        return x


class ResNet(nn.Module):
    """残差网络模型（JAX/Flax实现）"""
    
    hidden_dims: list
    output_dim: int
    activation: str = 'tanh'
    
    def setup(self):
        hidden_dims = list(self.hidden_dims)

        # 在 setup 中定义输入投影层（避免在 __call__ 动态创建）
        self.input_projection = nn.Dense(hidden_dims[0])

        # 输出层
        self.output_layer = nn.Dense(self.output_dim)
        
        # 创建残差块（先用本地 list，最后一次性赋值为 tuple）
        blocks = []
        if len(hidden_dims) == 1:
            block = nn.Sequential([
                nn.Dense(hidden_dims[0]),
                lambda x: jnp.tanh(x) if self.activation.lower() == 'tanh' else nn.relu(x),
                nn.Dense(hidden_dims[0])
            ])
            blocks.append(block)
        else:
            for i in range(len(hidden_dims)-1):
                block = nn.Sequential([
                    nn.Dense(hidden_dims[i+1]),
                    lambda x: jnp.tanh(x) if self.activation.lower() == 'tanh' else nn.relu(x),
                    nn.Dense(hidden_dims[i+1])
                ])
                blocks.append(block)
        self.blocks = tuple(blocks)
    
    def __call__(self, x):
        # 仅当输入维度不匹配时启用投影层（子模块已在 setup 中定义）
        if x.shape[-1] != self.hidden_dims[0]:
            x = self.input_projection(x)
        
        # 通过残差块
        identity = x
        for block in self.blocks:
            out = block(x)
            if out.shape == identity.shape:
                out = out + identity
            identity = out
            if self.activation.lower() == 'tanh':
                x = jnp.tanh(out)
            else:
                x = nn.relu(out)
        
        return self.output_layer(x)