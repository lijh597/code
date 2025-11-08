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
        dims = self.hidden_dims + [self.output_dim]
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
        # 输入投影层（如果需要）
        self.input_projection = None  # 将在__call__中处理
        
        # 输出层
        self.output_layer = nn.Dense(self.output_dim)
        
        # 创建残差块
        self.blocks = []
        if len(self.hidden_dims) == 1:
            # 如果只有一个隐藏层维度，创建一个简单的残差块
            block = nn.Sequential([
                nn.Dense(self.hidden_dims[0]),
                lambda x: jnp.tanh(x) if self.activation.lower() == 'tanh' else nn.relu(x),
                nn.Dense(self.hidden_dims[0])
            ])
            self.blocks.append(block)
        else:
            # 正常的多个残差块
            for i in range(len(self.hidden_dims)-1):
                block = nn.Sequential([
                    nn.Dense(self.hidden_dims[i+1]),
                    lambda x: jnp.tanh(x) if self.activation.lower() == 'tanh' else nn.relu(x),
                    nn.Dense(self.hidden_dims[i+1])
                ])
                self.blocks.append(block)
    
    def __call__(self, x):
        # 输入投影（如果需要）
        if x.shape[-1] != self.hidden_dims[0]:
            x = nn.Dense(self.hidden_dims[0])(x)
        
        # 通过残差块
        identity = x
        for block in self.blocks:
            out = block(x)
            # 残差连接：只有当形状匹配时才相加
            if out.shape == identity.shape:
                out = out + identity
            identity = out
            if self.activation.lower() == 'tanh':
                x = jnp.tanh(out)
            else:
                x = nn.relu(out)
        
        # 输出层
        return self.output_layer(x)