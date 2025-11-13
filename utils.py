import torch
import numpy as np
import matplotlib.pyplot as plt

def save_results_to_csv(results, filename):
    """将结果保存为CSV文件"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'framework', 'model', 'input_dim', 'hidden_dim', 'num_layers', 'activation', 
            'precision', 'operator', 'mode', 'time', 'value', 'mse',
            'baseline_f_time', 'baseline_grad_time'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # 确保所有字段都存在
            row = {}
            for field in fieldnames:
                row[field] = result.get(field, '')
            writer.writerow(row)

def plot_performance(results, filename):
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
    
    # 子图2：MSE vs 模式（相对 reverse_reverse）
    plt.subplot(1, 3, 2)
    mode_mses = {}
    for mode in modes[1:]:  # 跳过基准模式
        mode_results = [r for r in results
                        if r.get('mode') == mode
                        and r.get('operator') == 'laplacian'
                        and r.get('mse') is not None]
        if mode_results:
            mses = [r['mse'] for r in mode_results]
            mode_mses[mode] = np.mean(mses)

    if mode_mses:
        modes_list = list(mode_mses.keys())
        mse_values = [mode_mses[m] for m in modes_list]
        any_positive = np.any(np.array(mse_values) > 0)

        # 如果全为0，不用对数坐标，设置一个很小的上限以便显示
        if any_positive:
            plt.bar(modes_list, mse_values)
            plt.yscale('log')
            plt.ylabel('Mean Squared Error (log scale)')
        else:
            eps = 1e-12
            plt.bar(modes_list, [eps for _ in modes_list])
            plt.ylim(0, 1e-10)
            plt.ylabel('Mean Squared Error')
            # 可选：提示信息
            # for i, m in enumerate(modes_list):
            #     plt.text(i, 5e-11, 'all zero', ha='center', va='bottom', fontsize=8)

        plt.xlabel('AD Mode')
        plt.title('MSE vs AD Mode (relative to reverse_reverse)')
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
    plt.savefig(filename, dpi=150)
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

def save_dataset_npy(test_points, function_values, laplacian_values, biharmonic_values, 
                     model_info, filename='dataset.npy'):
    """
    保存测试数据集为.npy格式
    
    Args:
        test_points: 测试点坐标 (list of arrays, 每个元素长度可能不同)
        function_values: 函数值 (numpy array, shape: [n_samples])
        laplacian_values: 拉普拉斯算子值 (numpy array, shape: [n_samples])
        biharmonic_values: 双调和算子值 (numpy array, shape: [n_samples])
        model_info: 模型信息列表（每个采样点对应的模型配置）
        filename: 保存文件名
    """
    # 将 test_points 转换为对象数组（因为不同配置的 input_dim 不同）
    # 确保每个元素都是 numpy 数组
    test_points_arrays = []
    for pt in test_points:
        if isinstance(pt, np.ndarray):
            test_points_arrays.append(pt)
        else:
            test_points_arrays.append(np.array(pt))
    
    test_points_array = np.array(test_points_arrays, dtype=object)
    
    dataset = {
        'test_points': test_points_array,  # 使用对象数组
        'function_values': np.array(function_values),
        'laplacian_values': np.array(laplacian_values),
        'biharmonic_values': np.array(biharmonic_values),
        'model_info': model_info
    }
    np.save(filename, dataset, allow_pickle=True)  # 需要 allow_pickle=True 来保存对象数组
    print(f"数据集已保存到 {filename}")

def save_dataset_csv(test_points, function_values, laplacian_values, biharmonic_values,
                     model_info, filename='dataset.csv'):
    """
    保存测试数据集为CSV格式（便于查看）
    
    Args:
        test_points: 测试点坐标列表（每个点可能是不同维度）
        function_values: 函数值 (numpy array, shape: [n_samples])
        laplacian_values: 拉普拉斯算子值 (numpy array, shape: [n_samples])
        biharmonic_values: 双调和算子值 (numpy array, shape: [n_samples])
        model_info: 模型信息列表
        filename: 保存文件名
    """
    import csv
    import numpy as np
    
    # 不再尝试转换为同质数组，因为维度可能不同
    # test_points = np.asarray(test_points)  # 删除这行
    
    function_values = np.asarray(function_values)
    laplacian_values = np.asarray(laplacian_values)
    biharmonic_values = np.asarray(biharmonic_values)
    
    n_samples = len(test_points)
    
    # 找到最大 input_dim，用于确定 CSV 列数
    max_input_dim = 0
    for i, info in enumerate(model_info):
        if i < len(model_info):
            max_input_dim = max(max_input_dim, info.get('input_dim', 0))
    
    with open(filename, 'w', newline='') as csvfile:
        # 构建字段名：x_0, x_1, ..., x_{max_dim-1}, f(x), laplacian, biharmonic, model, input_dim, output_dim, hidden_dim, num_layers, activation, precision
        fieldnames = [f'x_{i}' for i in range(max_input_dim)] + ['f(x)', 'laplacian', 'biharmonic', 
                                                              'model', 'input_dim', 'output_dim', 'hidden_dim', 'num_layers', 'activation', 'precision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(n_samples):
            # 将每个点转换为 numpy 数组并展平
            point = np.array(test_points[i]).flatten()
            actual_dim = len(point)
            
            # 构建行数据
            row = {}
            # 填充坐标列（如果维度小于最大维度，多余列留空）
            for j in range(max_input_dim):
                if j < actual_dim:
                    row[f'x_{j}'] = point[j]
                else:
                    row[f'x_{j}'] = ''  # 维度不足时留空
            
            row['f(x)'] = function_values[i]
            row['laplacian'] = laplacian_values[i]
            row['biharmonic'] = biharmonic_values[i]
            
            # 添加模型信息
            if i < len(model_info):
                info = model_info[i]
                row['model'] = info.get('model', '')
                row['input_dim'] = info.get('input_dim', '')
                row['output_dim'] = info.get('output_dim', '')
                row['hidden_dim'] = info.get('hidden_dim', '')
                row['num_layers'] = info.get('num_layers', '')
                row['activation'] = info.get('activation', '')
                row['precision'] = info.get('precision', '')
            writer.writerow(row)
    
    print(f"数据集已保存到 {filename} (CSV格式)")