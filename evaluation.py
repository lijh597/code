import torch
import time
import numpy as np
from models import MLP, ResNet
from operators import laplacian_operator, biharmonic_operator
from config import INPUT_DIMS, HIDDEN_DIMS, HIDDEN_LAYERS, ACTIVATIONS, NUM_RUNS, AD_MODES, PRECISIONS
from utils import save_results_to_csv, plot_performance, calculate_mse

def evaluate_single_config(model, x, operator='laplacian', dtype=torch.float32):
    """
    评估单个配置下的所有AD模式
    
    Args:
        model: 神经网络模型
        x: 输入数据
        operator: 算子类型 ('laplacian' 或 'biharmonic')
        dtype: 数据类型
    
    Returns:
        results: 包含所有模式结果的列表
    """
    results = []
    
    # 设置数据类型
    if dtype == torch.float64:
        model = model.double()
        x = x.double()
    else:
        model = model.float()
        x = x.float()
    
    x.requires_grad_(True)
    
    # 1. 首先计算基准值（reverse_reverse模式）
    baseline_mode = "reverse_reverse"
    
    if operator == 'laplacian':
        baseline_func = laplacian_operator
    else:
        baseline_func = biharmonic_operator
    
    try:
        start_time = time.time()
        baseline_value = baseline_func(model, x.clone(), mode=baseline_mode)
        baseline_time = time.time() - start_time
        
        baseline_value_item = baseline_value.item() if baseline_value is not None else None
        
        # 记录基准结果
        results.append({
            'mode': baseline_mode,
            'operator': operator,
            'time': baseline_time,
            'value': baseline_value_item,
            'mse': 0.0  # 基准模式MSE为0
        })
        
        # 2. 测试其他模式，计算MSE
        for mode in AD_MODES[1:]:  # 跳过基准模式
            try:
                start_time = time.time()
                value = baseline_func(model, x.clone(), mode=mode)
                comp_time = time.time() - start_time
                
                value_item = value.item() if value is not None else None
                
                # 计算MSE（相对于基准）
                if baseline_value_item is not None and value_item is not None:
                    mse = calculate_mse(value_item, baseline_value_item)
                else:
                    mse = None
                
                results.append({
                    'mode': mode,
                    'operator': operator,
                    'time': comp_time,
                    'value': value_item,
                    'mse': mse
                })
            except Exception as e:
                print(f"  ⚠ 模式 {mode} 失败: {e}")
                results.append({
                    'mode': mode,
                    'operator': operator,
                    'time': None,
                    'value': None,
                    'mse': None
                })
    except Exception as e:
        print(f"  ✗ 基准模式计算失败: {e}")
        return []
    
    return results

def evaluate_performance(verbose=True):
    """
    性能评估主函数
    
    Args:
        verbose: 是否打印详细输出
    
    Returns:
        results: 所有评估结果的列表
    """
    results = []
    total_configs = 0
    
    # 测试不同参数组合（先测试小规模配置）
    test_configs = {
        'input_dims': INPUT_DIMS[:4],  # [1, 2, 3, 5]
        'hidden_dims': HIDDEN_DIMS[:2],  # [64, 128]
        'num_layers': HIDDEN_LAYERS[:2],  # [4, 6]
        'activations': ACTIVATIONS[:1],  # ['tanh']
        'precisions': PRECISIONS[:1],  # ['float32']
    }
    
    for input_dim in test_configs['input_dims']:
        for hidden_dim in test_configs['hidden_dims']:
            for num_layers in test_configs['num_layers']:
                for activation in test_configs['activations']:
                    for precision in test_configs['precisions']:
                        # 设置数据类型
                        dtype = torch.float32 if precision == 'float32' else torch.float64
                        
                        # 测试MLP模型
                        model = MLP(
                            input_dim=input_dim,
                            hidden_dims=[hidden_dim] * num_layers,
                            output_dim=1,
                            activation=activation
                        )
                        
                        # 创建测试数据
                        torch.manual_seed(42)  # 固定随机种子以保证可复现
                        x = torch.randn(1, input_dim, dtype=dtype, requires_grad=True)
                        
                        # 评估拉普拉斯算子
                        if verbose:
                            print(f"测试配置: MLP, dim={input_dim}, hidden={hidden_dim}, layers={num_layers}, "
                                  f"activation={activation}, precision={precision}")
                        
                        lap_results = evaluate_single_config(model, x, operator='laplacian', dtype=dtype)
                        for r in lap_results:
                            r.update({
                                'model': 'MLP',
                                'input_dim': input_dim,
                                'hidden_dim': hidden_dim,
                                'num_layers': num_layers,
                                'activation': activation,
                                'precision': precision
                            })
                            results.append(r)
                        
                        # 评估双调和算子
                        bih_results = evaluate_single_config(model, x, operator='biharmonic', dtype=dtype)
                        for r in bih_results:
                            r.update({
                                'model': 'MLP',
                                'input_dim': input_dim,
                                'hidden_dim': hidden_dim,
                                'num_layers': num_layers,
                                'activation': activation,
                                'precision': precision
                            })
                            results.append(r)
                        
                        total_configs += 1
    
    if verbose:
        print(f"\n✓ 完成评估，共测试 {total_configs} 个配置")
        print(f"✓ 总结果数: {len(results)}")
    
    return results

def print_summary(results):
    """打印评估结果摘要"""
    print("\n" + "=" * 80)
    print("评估结果摘要")
    print("=" * 80)
    
    # 按算子分组
    for operator in ['laplacian', 'biharmonic']:
        op_results = [r for r in results if r.get('operator') == operator]
        if not op_results:
            continue
        
        print(f"\n【{operator.upper()} 算子】")
        print("-" * 80)
        
        # 统计每个模式的性能
        for mode in AD_MODES:
            mode_results = [r for r in op_results if r.get('mode') == mode and r.get('value') is not None]
            if not mode_results:
                continue
            
            avg_time = np.mean([r['time'] for r in mode_results if r['time'] is not None])
            
            if mode == AD_MODES[0]:  # 基准模式
                print(f"{mode:20s}: 平均时间 = {avg_time:.6f}s (基准)")
            else:
                mses = [r['mse'] for r in mode_results if r.get('mse') is not None]
                avg_mse = np.mean(mses) if mses else None
                print(f"{mode:20s}: 平均时间 = {avg_time:.6f}s, 平均MSE = {avg_mse:.2e}")

def run_benchmark():
    """运行完整基准测试"""
    print("开始性能评估...")
    print("=" * 80)
    
    results = evaluate_performance(verbose=True)
    
    # 打印摘要
    print_summary(results)
    
    # 保存结果
    print("\n保存结果到 CSV 文件...")
    save_results_to_csv(results, "results.csv")
    print("✓ 结果已保存到 results.csv")
    
    # 绘制图表
    print("生成性能图表...")
    plot_performance(results)
    print("✓ 图表已保存到 performance_plot.png")
    
    return results

if __name__ == "__main__":
    results = run_benchmark()