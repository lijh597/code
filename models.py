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
        # 简化版ResNet实现
        self.input_projection = nn.Linear(input_dim, hidden_dims[0]) if input_dim != hidden_dims[0] else None
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.activation = nn.Tanh() if activation.lower() == 'tanh' else nn.ReLU()
        
        # 创建残差块
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                self.activation,
                nn.Linear(hidden_dims[i+1], hidden_dims[i+1])
            )
            self.blocks.append(block)
    
    def forward(self, x):
        if self.input_projection:
            x = self.input_projection(x)
            
        identity = x
        for block in self.blocks:
            out = block(x)
            # 残差连接
            if out.shape == identity.shape:
                out += identity
            identity = out
            x = self.activation(out)
            
        return self.output_layer(x)