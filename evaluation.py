import torch
import time
import numpy as np
from models import MLP, ResNet
from operators import laplacian_operator, biharmonic_operator
from config import INPUT_DIMS, HIDDEN_DIMS, HIDDEN_LAYERS, ACTIVATIONS, NUM_RUNS, AD_MODES, PRECISIONS
from utils import save_results_to_csv, plot_performance, calculate_mse

def measure_baselines(model, x):
    """
    返回: f_time, grad_time
    - f_time: 计算 f(x) 的时间
    - grad_time: 计算一阶导 ∇f(x) 的时间（以标量 f.sum() 反向为准）
    """
    x = x.requires_grad_(True)
    start = time.time()
    fx = model(x)
    f_time = time.time() - start

    start = time.time()
    grad = torch.autograd.grad(fx.sum(), x, retain_graph=False, create_graph=False)[0]
    grad_time = time.time() - start
    return float(f_time), float(grad_time)

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
    
    # 基线时间（函数值与一阶导）
    baseline_f_time, baseline_grad_time = measure_baselines(model, x)

    # 1. 基准模式（reverse_reverse）
    baseline_mode = "reverse_reverse"
    baseline_func = laplacian_operator if operator == 'laplacian' else biharmonic_operator

    try:
        start_time = time.time()
        baseline_value = baseline_func(model, x.clone(), mode=baseline_mode)
        baseline_time = time.time() - start_time

        baseline_value_item = baseline_value.item() if baseline_value is not None else None

        results.append({
            'mode': baseline_mode,
            'operator': operator,
            'time': baseline_time,
            'value': baseline_value_item,
            'mse': 0.0,
            'baseline_f_time': baseline_f_time,
            'baseline_grad_time': baseline_grad_time,
        })

        # 2. 其他模式
        for mode in AD_MODES[1:]:
            start_time = time.time()
            value = baseline_func(model, x.clone(), mode=mode)
            comp_time = time.time() - start_time
            value_item = value.item() if value is not None else None
            mse = calculate_mse(value_item, baseline_value_item) if (baseline_value_item is not None and value_item is not None) else None

            results.append({
                'mode': mode,
                'operator': operator,
                'time': comp_time,
                'value': value_item,
                'mse': mse,
                'baseline_f_time': baseline_f_time,
                'baseline_grad_time': baseline_grad_time,
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

    # 全量参数
    test_configs = {
        'input_dims': INPUT_DIMS,          # 全量
        'hidden_dims': HIDDEN_DIMS,        # 全量
        'num_layers': HIDDEN_LAYERS,       # 全量
        'activations': ACTIVATIONS,        # 全量
        'precisions': PRECISIONS,          # float32/float64
    }

    # 两类模型
    def build_model(model_name, input_dim, hidden_dim, num_layers, activation):
        if model_name == 'MLP':
            return MLP(input_dim=input_dim,
                       hidden_dims=[hidden_dim] * num_layers,
                       output_dim=1,
                       activation=activation)
        elif model_name == 'ResNet':
            # 复用 hidden_dims 作为每层宽度
            return ResNet(input_dim=input_dim,
                          hidden_dims=[hidden_dim] * num_layers,
                          output_dim=1,
                          activation=activation)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    for model_name in ['MLP', 'ResNet']:
        for input_dim in test_configs['input_dims']:
            for hidden_dim in test_configs['hidden_dims']:
                for num_layers in test_configs['num_layers']:
                    for activation in test_configs['activations']:
                        for precision in test_configs['precisions']:
                            dtype = torch.float32 if precision == 'float32' else torch.float64

                            if verbose:
                                print(f"测试配置: {model_name}, dim={input_dim}, hidden={hidden_dim}, "
                                      f"layers={num_layers}, activation={activation}, precision={precision}")

                            # 多次运行取均值
                            lap_runs, bih_runs = [], []
                            for _ in range(NUM_RUNS):
                                torch.manual_seed(42)
                                model = build_model(model_name, input_dim, hidden_dim, num_layers, activation)
                                x = torch.randn(1, input_dim, dtype=dtype, requires_grad=True)

                                lap_runs.extend(evaluate_single_config(model, x, operator='laplacian', dtype=dtype))
                                # 重新构建模型与 x，避免计算图残留与缓存影响
                                model = build_model(model_name, input_dim, hidden_dim, num_layers, activation)
                                x = torch.randn(1, input_dim, dtype=dtype, requires_grad=True)
                                bih_runs.extend(evaluate_single_config(model, x, operator='biharmonic', dtype=dtype))

                            # 聚合（时间取均值，value/mse 取均值）
                            def aggregate(run_list):
                                from collections import defaultdict
                                buckets = defaultdict(list)
                                for r in run_list:
                                    key = (r['operator'], r['mode'])
                                    buckets[key].append(r)
                                out = []
                                for (operator, mode), lst in buckets.items():
                                    avg_time = float(np.mean([z['time'] for z in lst if z['time'] is not None]))
                                    avg_val = float(np.mean([z['value'] for z in lst if z['value'] is not None])) if any(z['value'] is not None for z in lst) else None
                                    avg_mse = float(np.mean([z['mse'] for z in lst if z['mse'] is not None])) if any(z['mse'] is not None for z in lst) else None
                                    bf = float(np.mean([z['baseline_f_time'] for z in lst if z.get('baseline_f_time') is not None]))
                                    bg = float(np.mean([z['baseline_grad_time'] for z in lst if z.get('baseline_grad_time') is not None]))
                                    out.append({
                                        'operator': operator,
                                        'mode': mode,
                                        'time': avg_time,
                                        'value': avg_val,
                                        'mse': avg_mse,
                                        'baseline_f_time': bf,
                                        'baseline_grad_time': bg,
                                    })
                                return out

                            lap_final = aggregate(lap_runs)
                            bih_final = aggregate(bih_runs)

                            # 附加公共字段
                            for r in lap_final + bih_final:
                                r.update({
                                    'model': model_name,
                                    'input_dim': input_dim,
                                    'hidden_dim': hidden_dim,
                                    'num_layers': num_layers,
                                    'activation': activation,
                                    'precision': precision,
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