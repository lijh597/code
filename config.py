# 参数配置
INPUT_DIMS = [3,10,50]#[1, 2, 3, 5, 10, 50, 100]
OUTPUT_DIMS = [3,10,50]#[1, 2, 3, 5, 10, 50, 100]
HIDDEN_DIMS = [64]#[64, 128, 512]
HIDDEN_LAYERS = [4]#[4, 6, 8]
ACTIVATIONS = ['tanh', 'relu']
PRECISIONS = ['float32']#, 'float64']

# ResNet对应的残差块数
RESNET_BLOCKS = [2, 3, 4]

# 自动微分模式
AD_MODES = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]

# 测试运行次数
NUM_RUNS = 3#10