# ============================================================
# TensorFlow versions of MLP and ResNet architecture
# These mimic the PyTorch models (models.py) but use tf.keras
# ============================================================

import tensorflow as tf

# MLP definition
# fully-connected feedforward neural network
class MLP(tf.keras.Model):
    """多层感知机模型"""
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super().__init__()
        act = tf.keras.activations.tanh if activation == 'tanh' else tf.keras.activations.relu
        layers = []
        for h in hidden_dims:
            layers.append(tf.keras.layers.Dense(h, activation=act))
        layers.append(tf.keras.layers.Dense(output_dim))
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)


# ResNet definition
# Residual MLP with skip connections every block, helps with deeper networks and smoother gradients.
class ResNetBlock(tf.keras.layers.Layer):
    """残差网络模型"""
    def __init__(self, dim, activation='tanh'):
        super().__init__()
        act = tf.keras.activations.tanh if activation == 'tanh' else tf.keras.activations.relu
        self.d1 = tf.keras.layers.Dense(dim, activation=act)
        self.d2 = tf.keras.layers.Dense(dim, activation=act)

    def call(self, x):
        return self.d2(self.d1(x)) + x


class ResNet(tf.keras.Model):
    """残差网络模型"""
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        super().__init__()
        self.input_layer = tf.keras.layers.Dense(hidden_dims[0], activation=activation)
        self.blocks = [ResNetBlock(h, activation=activation) for h in hidden_dims]
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)
