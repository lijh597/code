import torch
import torch.nn as nn

class MLP(nn.Module):
    """多层感知机模型"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # 除输出层外添加激活函数
                layers.append(nn.Tanh() if activation.lower() == 'tanh' else nn.ReLU())
                
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ResNet(nn.Module):
    """残差网络模型"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(ResNet, self).__init__()
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dims[0]) if input_dim != hidden_dims[0] else None
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # 激活函数
        self.activation = nn.Tanh() if activation.lower() == 'tanh' else nn.ReLU()
        
        # 创建残差块 - 修复：确保至少有一个块
        self.blocks = nn.ModuleList()
        if len(hidden_dims) == 1:
            # 如果只有一个隐藏层维度，创建一个简单的残差块
            block = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                self.activation,
                nn.Linear(hidden_dims[0], hidden_dims[0])
            )
            self.blocks.append(block)
        else:
            # 正常的多个残差块
            for i in range(len(hidden_dims)-1):
                # 如果维度相同，使用残差连接；否则需要投影
                if hidden_dims[i] == hidden_dims[i+1]:
                    block = nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        self.activation,
                        nn.Linear(hidden_dims[i+1], hidden_dims[i+1])
                    )
                else:
                    # 维度不同时需要投影层
                    block = nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        self.activation,
                        nn.Linear(hidden_dims[i+1], hidden_dims[i+1])
                    )
                self.blocks.append(block)
    
    def forward(self, x):
        # 输入投影
        if self.input_projection:
            x = self.input_projection(x)
        
        # 通过残差块
        identity = x
        for block in self.blocks:
            out = block(x)
            # 残差连接：只有当形状匹配时才相加
            if out.shape == identity.shape:
                out = out + identity
            identity = out
            x = self.activation(out)
        
        # 输出层
        return self.output_layer(x)