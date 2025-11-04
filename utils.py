import torch
import numpy as np
import matplotlib.pyplot as plt

def save_results_to_csv(results, filename):
    """将结果保存为CSV文件"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['model', 'input_dim', 'hidden_dim', 'num_layers', 'activation', 
                      'precision', 'operator', 'mode', 'time', 'value', 'mse']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # 确保所有字段都存在
            row = {}
            for field in fieldnames:
                row[field] = result.get(field, '')
            writer.writerow(row)

def plot_performance(results):
    """绘制性能图表"""
    # 提取不同模式的性能数据
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]
    
    # 绘制计算时间对比
    plt.figure(figsize=(15, 5))
    
    # 子图1：计算时间 vs 输入维度
    plt.subplot(1, 3, 1)
    for mode in modes:
        mode_results = [r for r in results if r.get('mode') == mode and r.get('operator') == 'laplacian']
        if mode_results:
            input_dims = [r['input_dim'] for r in mode_results]
            times = [r['time'] for r in mode_results]
            plt.plot(input_dims, times, 'o-', label=mode)
    plt.xlabel('Input Dimension')
    plt.ylabel('Computation Time (s)')
    plt.title('Laplacian Operator: Time vs Input Dimension')
    plt.legend()
    plt.grid(True)
    
    # 子图2：MSE vs 模式
    plt.subplot(1, 3, 2)
    mode_mses = {}
    for mode in modes[1:]:  # 跳过基准模式
        mode_results = [r for r in results if r.get('mode') == mode and r.get('operator') == 'laplacian' and r.get('mse') is not None]
        if mode_results:
            mses = [r['mse'] for r in mode_results]
            mode_mses[mode] = np.mean(mses)
    
    if mode_mses:
        modes_list = list(mode_mses.keys())
        mse_values = [mode_mses[m] for m in modes_list]
        plt.bar(modes_list, mse_values)
        plt.xlabel('AD Mode')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE vs AD Mode (relative to reverse_reverse)')
        plt.yscale('log')
        plt.grid(True, axis='y')
    
    # 子图3：计算时间 vs 模式
    plt.subplot(1, 3, 3)
    mode_times = {}
    for mode in modes:
        mode_results = [r for r in results if r.get('mode') == mode and r.get('operator') == 'laplacian']
        if mode_results:
            times = [r['time'] for r in mode_results]
            mode_times[mode] = np.mean(times)
    
    if mode_times:
        modes_list = list(mode_times.keys())
        time_values = [mode_times[m] for m in modes_list]
        plt.bar(modes_list, time_values)
        plt.xlabel('AD Mode')
        plt.ylabel('Mean Computation Time (s)')
        plt.title('Average Time vs AD Mode')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('performance_plot.png', dpi=150)
    plt.show()

def calculate_mse(predictions, targets):
    """计算均方误差
    
    Args:
        predictions: 预测值（可以是tensor或numpy数组或标量）
        targets: 目标值（可以是tensor或numpy数组或标量）
    
    Returns:
        MSE值（标量）
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # 转换为numpy数组
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    
    # 计算MSE
    mse = np.mean((predictions - targets) ** 2)
    return float(mse)