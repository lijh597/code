import torch
import time
import numpy as np
from models_pytorch import MLP, ResNet
from operators import laplacian_operator, biharmonic_operator
from config import INPUT_DIMS, HIDDEN_DIMS, HIDDEN_LAYERS, ACTIVATIONS, NUM_RUNS, AD_MODES, PRECISIONS, CURRENT_FRAMEWORKS, FRAMEWORKS, OUTPUT_DIMS
from utils import save_results_to_csv, plot_performance, calculate_mse

# 在文件开头添加框架选择逻辑
from config import CURRENT_FRAMEWORKS, FRAMEWORKS

def get_framework_imports(framework_name):
    """根据框架名称返回对应的导入"""
    if framework_name == 'pytorch':
        from models_pytorch import MLP, ResNet
        from operators_pytorch import laplacian_operator, biharmonic_operator
        import torch
        return {
            'MLP': MLP,
            'ResNet': ResNet,
            'laplacian_operator': laplacian_operator,
            'biharmonic_operator': biharmonic_operator,
            'torch': torch,
            'framework': 'pytorch'
        }
    elif framework_name == 'tensorflow':
        from models_tensorflow import MLP, ResNet
        from operators_tensorflow import laplacian_operator, biharmonic_operator
        import tensorflow as tf
        return {
            'MLP': MLP,
            'ResNet': ResNet,
            'laplacian_operator': laplacian_operator,
            'biharmonic_operator': biharmonic_operator,
            'tf': tf,
            'framework': 'tensorflow'
        }
    elif framework_name == 'jax':
        from models_jax import MLP, ResNet
        from operators_jax import laplacian_operator, biharmonic_operator
        import jax.numpy as jnp
        return {
            'MLP': MLP,
            'ResNet': ResNet,
            'laplacian_operator': laplacian_operator,
            'biharmonic_operator': biharmonic_operator,
            'jnp': jnp,
            'framework': 'jax'
        }
    else:
        raise ValueError(f"Unknown framework: {framework_name}")

def measure_baselines(model, x, framework_name='pytorch'):
    """
    返回: f_time, grad_time
    - f_time: 计算 f(x) 的时间
    - grad_time: 计算一阶导 ∇f(x) 的时间（以标量 f.sum() 反向为准）
    
    Args:
        model: 神经网络模型
        x: 输入数据
        framework_name: 框架名称 ('pytorch', 'tensorflow', 'jax')
    """
    import time
    
    if framework_name == 'pytorch':
        x = x.requires_grad_(True)
        start = time.time()
        fx = model(x)
        f_time = time.time() - start

        start = time.time()
        grad = torch.autograd.grad(fx.sum(), x, retain_graph=False, create_graph=False)[0]
        grad_time = time.time() - start
        return float(f_time), float(grad_time)
    
    elif framework_name == 'tensorflow':
        import tensorflow as tf
        with tf.GradientTape() as tape:
            tape.watch(x)
            start = time.time()
            fx = model(x)
            f_time = time.time() - start
        
        start = time.time()
        grad = tape.gradient(tf.reduce_sum(fx), x)
        grad_time = time.time() - start
        return float(f_time), float(grad_time)
    
    elif framework_name == 'jax':
        from jax import grad
        import jax.numpy as jnp
        
        def f(x):
            return model(x).sum()
        
        start = time.time()
        fx = model(x)
        f_time = time.time() - start
        
        start = time.time()
        grad_fn = grad(f)
        grad_val = grad_fn(x)
        grad_time = time.time() - start
        return float(f_time), float(grad_time)
    
    else:
        raise ValueError(f"Unknown framework: {framework_name}")

def evaluate_single_config(model, x, operator='laplacian', dtype=None, framework_name='pytorch',
                          laplacian_operator=None, biharmonic_operator=None):
    """
    评估单个配置下的所有AD模式（支持多框架）
    
    Args:
        model: 神经网络模型
        x: 输入数据
        operator: 算子类型 ('laplacian' 或 'biharmonic')
        dtype: 数据类型（框架特定）
        framework_name: 框架名称 ('pytorch', 'tensorflow', 'jax')
        laplacian_operator: 拉普拉斯算子函数
        biharmonic_operator: 双调和算子函数
    
    Returns:
        results: 包含所有模式结果的列表
    """
    results = []
    
    # 根据框架设置数据类型和准备输入
    if framework_name == 'pytorch':
        import torch
        if dtype is None:
            dtype = torch.float32
        if dtype == torch.float64:
            model = model.double()
            x = x.double()
        else:
            model = model.float()
            x = x.float()
        x.requires_grad_(True)
        
        def to_numpy(value):
            return value.item() if hasattr(value, 'item') else float(value)
        
        def clone_input(x):
            return x.clone()
    
    elif framework_name == 'tensorflow':
        import tensorflow as tf
        if dtype is None:
            dtype = tf.float32
        # TensorFlow 模型会自动处理数据类型
        
        def to_numpy(value):
            return float(value.numpy())
        
        def clone_input(x):
            return tf.Variable(x, trainable=True)
    
    elif framework_name == 'jax':
        import jax.numpy as jnp
        if dtype is None:
            dtype = jnp.float32
        x = jnp.array(x, dtype=dtype)
        
        def to_numpy(value):
            return float(value)
        
        def clone_input(x):
            return jnp.array(x)
    
    else:
        raise ValueError(f"Unknown framework: {framework_name}")
    
    # 基线时间（函数值与一阶导）
    baseline_f_time, baseline_grad_time = measure_baselines(model, x, framework_name)

    # 1. 基准模式（reverse_reverse）
    baseline_mode = "reverse_reverse"
    baseline_func = laplacian_operator if operator == 'laplacian' else biharmonic_operator

    try:
        start_time = time.time()
        baseline_value = baseline_func(model, clone_input(x), mode=baseline_mode)
        baseline_time = time.time() - start_time

        baseline_value_item = to_numpy(baseline_value) if baseline_value is not None else None

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
            value = baseline_func(model, clone_input(x), mode=mode)
            comp_time = time.time() - start_time
            value_item = to_numpy(value) if value is not None else None
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
        import traceback
        traceback.print_exc()
        return []

    return results

def evaluate_performance(frameworks=None, verbose=True, collect_dataset=True):
    """
    性能评估主函数（支持多框架）
    
    Args:
        frameworks: 要测试的框架列表，如果为None则使用CURRENT_FRAMEWORKS
        verbose: 是否打印详细输出
        collect_dataset: 是否收集数据集（用于保存）
    
    Returns:
        results: 所有评估结果的列表（包含framework字段）
        dataset_data_by_framework: 按框架分组的数据集字典 {framework_name: dataset_data}
    """
    if frameworks is None:
        frameworks = CURRENT_FRAMEWORKS
    
    all_results = []
    # 修改：按框架分组保存数据集
    dataset_data_by_framework = {}
    
    # 对每个框架分别评估
    for framework_name in frameworks:
        if verbose:
            print(f"\n{'='*80}")
            print(f"测试框架: {framework_name.upper()}")
            print(f"{'='*80}")
        
        # 获取框架特定的导入
        fw_imports = get_framework_imports(framework_name)
        MLP = fw_imports['MLP']
        ResNet = fw_imports['ResNet']
        laplacian_operator = fw_imports['laplacian_operator']
        biharmonic_operator = fw_imports['biharmonic_operator']
        
        # 根据框架选择数据类型和随机数生成
        if framework_name == 'pytorch':
            torch = fw_imports['torch']
            def randn(shape, dtype):
                return torch.randn(shape, dtype=dtype, requires_grad=True)
            def manual_seed(seed):
                torch.manual_seed(seed)
        elif framework_name == 'tensorflow':
            tf = fw_imports['tf']
            def randn(shape, dtype):
                return tf.Variable(tf.random.normal(shape, dtype=dtype))
            def manual_seed(seed):
                tf.random.set_seed(seed)
        elif framework_name == 'jax':
            jnp = fw_imports['jnp']
            import jax.random as jr
            key = jr.PRNGKey(42)
            def randn(shape, dtype):
                nonlocal key
                key, subkey = jr.split(key)
                return jr.normal(subkey, shape).astype(dtype)
            def manual_seed(seed):
                nonlocal key
                key = jr.PRNGKey(seed)
        
        # 全量参数
        test_configs = {
            'input_dims': INPUT_DIMS,          # 全量
            'output_dims': OUTPUT_DIMS,        # 添加输出维度
            'hidden_dims': HIDDEN_DIMS,        # 全量
            'num_layers': HIDDEN_LAYERS,       # 全量
            'activations': ACTIVATIONS,        # 全量
            'precisions': PRECISIONS,          # float32/float64
        }

        # 两类模型
        def build_model(model_name, input_dim, hidden_dim, num_layers, activation, output_dim):
            if framework_name == 'jax':
                # JAX/Flax 模型需要初始化参数
                import jax.numpy as jnp
                import jax.random as jr
                
                # 创建模型实例
                if model_name == 'MLP':
                    model_def = MLP(
                        hidden_dims=[hidden_dim] * num_layers,
                        output_dim=output_dim,  # 使用参数
                        activation=activation
                    )
                elif model_name == 'ResNet':
                    model_def = ResNet(
                        hidden_dims=[hidden_dim] * num_layers,
                        output_dim=output_dim,  # 使用参数
                        activation=activation
                    )
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # 初始化参数（使用 dummy input）
                key = jr.PRNGKey(42)
                dummy_input = jnp.ones((1, input_dim), dtype=jnp.float32)
                params = model_def.init(key, dummy_input)

                # 返回一个可调用的模型函数
                def model_fn(x):
                    return model_def.apply(params, x)
                
                return model_fn
            else:
                # PyTorch 和 TensorFlow 模型直接返回
                if model_name == 'MLP':
                    return MLP(input_dim=input_dim,
                               hidden_dims=[hidden_dim] * num_layers,
                               output_dim=output_dim,  # 使用参数
                               activation=activation)
                elif model_name == 'ResNet':
                    return ResNet(input_dim=input_dim,
                                  hidden_dims=[hidden_dim] * num_layers,
                                  output_dim=output_dim,  # 使用参数
                                  activation=activation)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

        # 修改：为每个框架单独收集数据集
        framework_dataset_data = {
            'points': [],
            'f_values': [],
            'lap_values': [],
            'bih_values': [],
            'model_info': []
        }

        for model_name in ['MLP', 'ResNet']:
            for input_dim in test_configs['input_dims']:
                for output_dim in test_configs['output_dims']:  # 添加输出维度循环
                    for hidden_dim in test_configs['hidden_dims']:
                        for num_layers in test_configs['num_layers']:
                            for activation in test_configs['activations']:
                                for precision in test_configs['precisions']:
                                    # 根据框架设置 dtype
                                    if framework_name == 'pytorch':
                                        import torch
                                        dtype = torch.float32 if precision == 'float32' else torch.float64
                                    elif framework_name == 'tensorflow':
                                        import tensorflow as tf
                                        dtype = tf.float32 if precision == 'float32' else tf.float64
                                    elif framework_name == 'jax':
                                        import jax.numpy as jnp
                                        dtype = jnp.float32 if precision == 'float32' else jnp.float64

                                    if verbose:
                                        print(f"测试配置: {model_name}, input_dim={input_dim}, output_dim={output_dim}, hidden={hidden_dim}, "
                                              f"layers={num_layers}, activation={activation}, precision={precision}")

                                    # 收集数据集（每个配置收集一个代表性测试点）
                                    if collect_dataset:
                                        manual_seed(42)  # 固定随机种子
                                        model_sample = build_model(model_name, input_dim, hidden_dim, num_layers, activation, output_dim)
                                        # 根据框架设置模型数据类型（如果需要）
                                        if framework_name == 'pytorch':
                                            if dtype == torch.float64:
                                                model_sample = model_sample.double()
                                        elif framework_name == 'tensorflow':
                                            import tensorflow as tf
                                            if dtype == tf.float64:
                                                # TensorFlow 模型会自动处理数据类型，可能需要特殊处理
                                                pass  # 或者根据需要进行转换
                                        elif framework_name == 'jax':
                                            import jax.numpy as jnp
                                            if dtype == jnp.float64:
                                                # JAX 模型会自动处理数据类型
                                                pass
                                        # TensorFlow 和 JAX 会自动处理数据类型，不需要手动转换
                                        
                                        x_sample = randn((1, input_dim), dtype)  # 添加 input_dim
                                        if framework_name == 'pytorch':
                                            f_x = model_sample(x_sample).sum().item()  # 多维输出时求和
                                            lap_value = laplacian_operator(model_sample, x_sample.clone(), mode='reverse_reverse').item()
                                            bih_value = biharmonic_operator(model_sample, x_sample.clone(), mode='reverse_reverse').item()
                                            framework_dataset_data['points'].append(x_sample.detach().cpu().numpy().flatten())
                                        elif framework_name == 'tensorflow':
                                            import tensorflow as tf
                                            f_x = float(tf.reduce_sum(model_sample(x_sample)).numpy())
                                            lap_value = float(laplacian_operator(model_sample, tf.Variable(x_sample), mode='reverse_reverse').numpy())
                                            bih_value = float(biharmonic_operator(model_sample, tf.Variable(x_sample), mode='reverse_reverse').numpy())
                                            framework_dataset_data['points'].append(x_sample.numpy().flatten())
                                        elif framework_name == 'jax':
                                            import jax.numpy as jnp
                                            f_x = float(jnp.sum(model_sample(x_sample)))
                                            lap_value = float(laplacian_operator(model_sample, x_sample, mode='reverse_reverse'))
                                            bih_value = float(biharmonic_operator(model_sample, x_sample, mode='reverse_reverse'))
                                            framework_dataset_data['points'].append(jnp.array(x_sample).flatten())
                                        
                                        framework_dataset_data['f_values'].append(f_x)
                                        framework_dataset_data['lap_values'].append(lap_value)
                                        framework_dataset_data['bih_values'].append(bih_value)
                                        framework_dataset_data['model_info'].append({
                                            'model': model_name,
                                            'input_dim': input_dim,
                                            'output_dim': output_dim,  # 添加输出维度
                                            'hidden_dim': hidden_dim,
                                            'num_layers': num_layers,
                                            'activation': activation,
                                            'precision': precision
                                        })

                                    # 多次运行取均值
                                    lap_runs, bih_runs = [], []
                                    for _ in range(NUM_RUNS):
                                        manual_seed(42)
                                        model = build_model(model_name, input_dim, hidden_dim, num_layers, activation, output_dim)
                                        x = randn((1, input_dim), dtype)  # 添加 input_dim

                                        lap_runs.extend(evaluate_single_config(
                                            model, x, operator='laplacian', dtype=dtype, 
                                            framework_name=framework_name,
                                            laplacian_operator=laplacian_operator,
                                            biharmonic_operator=biharmonic_operator
                                        ))
                                        # 重新构建模型与 x，避免计算图残留与缓存影响
                                        model = build_model(model_name, input_dim, hidden_dim, num_layers, activation, output_dim)
                                        x = randn((1, input_dim), dtype)  # 修复：添加 input_dim
                                        bih_runs.extend(evaluate_single_config(
                                            model, x, operator='biharmonic', dtype=dtype,
                                            framework_name=framework_name,
                                            laplacian_operator=laplacian_operator,
                                            biharmonic_operator=biharmonic_operator
                                        ))

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
                                            'output_dim': output_dim,  # 添加输出维度
                                            'hidden_dim': hidden_dim,
                                            'num_layers': num_layers,
                                            'activation': activation,
                                            'precision': precision,
                                            'framework': framework_name
                                        })
                                        all_results.append(r)

        # 修改：保存每个框架的数据集
        if collect_dataset and framework_dataset_data['points']:
            dataset_data_by_framework[framework_name] = {
                'points': framework_dataset_data['points'],
                'f_values': framework_dataset_data['f_values'],
                'lap_values': framework_dataset_data['lap_values'],
                'bih_values': framework_dataset_data['bih_values'],
                'model_info': framework_dataset_data['model_info']
            }

    if verbose:
        print(f"\n✓ 完成评估，共测试 {len(all_results)} 个配置")
        print(f"✓ 总结果数: {len(all_results)}")

    # 返回结果和按框架分组的数据集
    return all_results, dataset_data_by_framework if collect_dataset else None


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

def save_dataset_npy(test_points, function_values, laplacian_values, biharmonic_values, 
                     model_info, filename='dataset.npy'):
    """
    保存测试数据集为.npy格式
    
    Args:
        test_points: 测试点坐标 (numpy array, shape: [n_samples, input_dim])
        function_values: 函数值 (numpy array, shape: [n_samples])
        laplacian_values: 拉普拉斯算子值 (numpy array, shape: [n_samples])
        biharmonic_values: 双调和算子值 (numpy array, shape: [n_samples])
        model_info: 模型信息列表（每个采样点对应的模型配置）
        filename: 保存文件名
    """
    dataset = {
        'test_points': np.array(test_points),
        'function_values': np.array(function_values),
        'laplacian_values': np.array(laplacian_values),
        'biharmonic_values': np.array(biharmonic_values),
        'model_info': model_info
    }
    np.save(filename, dataset)
    print(f"数据集已保存到 {filename}")

def save_dataset_csv(test_points, function_values, laplacian_values, biharmonic_values,
                     model_info, filename='dataset.csv'):
    """
    保存测试数据集为CSV格式（便于查看）
    
    Args:
        test_points: 测试点坐标 (numpy array, shape: [n_samples, input_dim])
        function_values: 函数值 (numpy array, shape: [n_samples])
        laplacian_values: 拉普拉斯算子值 (numpy array, shape: [n_samples])
        biharmonic_values: 双调和算子值 (numpy array, shape: [n_samples])
        model_info: 模型信息列表
        filename: 保存文件名
    """
    import csv
    import numpy as np
    
    test_points = np.asarray(test_points)
    function_values = np.asarray(function_values)
    laplacian_values = np.asarray(laplacian_values)
    biharmonic_values = np.asarray(biharmonic_values)
    
    n_samples, input_dim = test_points.shape
    
    with open(filename, 'w', newline='') as csvfile:
        # 构建字段名：x_0, x_1, ..., x_n, f(x), laplacian, biharmonic, model, input_dim, hidden_dim, num_layers, activation, precision
        fieldnames = [f'x_{i}' for i in range(input_dim)] + ['f(x)', 'laplacian', 'biharmonic', 
                                                              'model', 'input_dim', 'hidden_dim', 'num_layers', 'activation', 'precision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(n_samples):
            row = {f'x_{j}': test_points[i, j] for j in range(input_dim)}
            row['f(x)'] = function_values[i]
            row['laplacian'] = laplacian_values[i]
            row['biharmonic'] = biharmonic_values[i]
            # 添加模型信息
            if i < len(model_info):
                info = model_info[i]
                row['model'] = info.get('model', '')
                row['input_dim'] = info.get('input_dim', '')
                row['hidden_dim'] = info.get('hidden_dim', '')
                row['num_layers'] = info.get('num_layers', '')
                row['activation'] = info.get('activation', '')
                row['precision'] = info.get('precision', '')
            writer.writerow(row)
    
    print(f"数据集已保存到 {filename} (CSV格式)")

def run_benchmark():
    """运行完整基准测试"""
    print("开始性能评估...")
    print("=" * 80)
    
    # 运行评估并收集数据集
    results, dataset_data_by_framework = evaluate_performance(verbose=True, collect_dataset=True)
    
    # 打印摘要
    print_summary(results)
    
    # 修改：按框架分别保存结果和数据集
    if dataset_data_by_framework:
        from utils import save_dataset_npy, save_dataset_csv
        
        # 获取所有测试的框架
        frameworks = list(dataset_data_by_framework.keys())
        
        # 按框架分组结果
        results_by_framework = {}
        for r in results:
            framework = r.get('framework', 'unknown')
            if framework not in results_by_framework:
                results_by_framework[framework] = []
            results_by_framework[framework].append(r)
        
        # 为每个框架分别保存
        for framework_name in frameworks:
            print(f"\n保存 {framework_name.upper()} 框架的结果...")
            
            # 保存结果 CSV
            if framework_name in results_by_framework:
                results_csv_filename = f"{framework_name}_results.csv"
                save_results_to_csv(results_by_framework[framework_name], results_csv_filename)
                print(f"✓ 结果已保存到 {results_csv_filename}")
            
            # 保存数据集
            if framework_name in dataset_data_by_framework:
                dataset_npy_filename = f"{framework_name}_dataset.npy"
                dataset_csv_filename = f"{framework_name}_dataset.csv"
                
                dataset_data = dataset_data_by_framework[framework_name]
                save_dataset_npy(dataset_data['points'], dataset_data['f_values'], 
                                dataset_data['lap_values'], dataset_data['bih_values'],
                                dataset_data['model_info'], dataset_npy_filename)
                save_dataset_csv(dataset_data['points'], dataset_data['f_values'],
                               dataset_data['lap_values'], dataset_data['bih_values'],
                               dataset_data['model_info'], dataset_csv_filename)
                print(f"✓ 数据集已保存到 {dataset_npy_filename} 和 {dataset_csv_filename}")

    # 按框架分别绘图（文件名使用前缀）
    from utils import plot_performance
    for framework_name, fw_results in results_by_framework.items():
        fig_name = f"{framework_name}_performance_plot.png"
        print(f"\n生成 {framework_name.upper()} 框架性能图表...")
        plot_performance(fw_results, filename=fig_name)
        print(f"✓ 图表已保存到 {fig_name}")
    
    # 已按框架分别保存图表
    return results

if __name__ == "__main__":
    results = run_benchmark()