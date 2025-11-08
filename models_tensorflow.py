"""
TensorFlow 实现的模型
"""
import tensorflow as tf
from tensorflow import keras

class MLP(keras.Model):
    """多层感知机模型（TensorFlow实现）"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(MLP, self).__init__()
        self.layers_list = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            self.layers_list.append(keras.layers.Dense(dims[i+1], use_bias=True))
            if i < len(dims)-2:  # 除输出层外添加激活函数
                if activation.lower() == 'tanh':
                    self.layers_list.append(keras.layers.Activation('tanh'))
                else:
                    self.layers_list.append(keras.layers.ReLU())
    
    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


class ResNet(keras.Model):
    """残差网络模型（TensorFlow实现）"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super(ResNet, self).__init__()
        
        # 输入投影层
        if input_dim != hidden_dims[0]:
            self.input_projection = keras.layers.Dense(hidden_dims[0], use_bias=True)
        else:
            self.input_projection = None
        
        # 输出层
        self.output_layer = keras.layers.Dense(output_dim, use_bias=True)
        
        # 激活函数
        self.activation = keras.layers.Activation('tanh') if activation.lower() == 'tanh' else keras.layers.ReLU()
        
        # 创建残差块
        self.blocks = []
        if len(hidden_dims) == 1:
            # 如果只有一个隐藏层维度，创建一个简单的残差块
            block = keras.Sequential([
                keras.layers.Dense(hidden_dims[0], use_bias=True),
                self.activation,
                keras.layers.Dense(hidden_dims[0], use_bias=True)
            ])
            self.blocks.append(block)
        else:
            # 正常的多个残差块
            for i in range(len(hidden_dims)-1):
                block = keras.Sequential([
                    keras.layers.Dense(hidden_dims[i+1], use_bias=True),
                    self.activation,
                    keras.layers.Dense(hidden_dims[i+1], use_bias=True)
                ])
                self.blocks.append(block)
    
    def call(self, x):
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
