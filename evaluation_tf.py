"""
evaluation_tf.py
Benchmarking for TensorFlow version (Laplacian & Biharmonic operators)
Outputs results_tf.csv and performance_tf.png
"""
import time
import numpy as np
import tensorflow as tf
from models_tf import MLP, ResNet
from operators_tf import laplacian_operator, biharmonic_operator
from config import INPUT_DIMS, HIDDEN_DIMS, HIDDEN_LAYERS, ACTIVATIONS, NUM_RUNS, PRECISIONS
from utils_tf import save_results_to_csv, plot_performance

def evaluate_single_config_tf(model, x, operator='laplacian', dtype=tf.float32):
    """
    Evaluate all modes for one configuration
    Currently supports reverse_reverse (TensorFlow reverse-mode AD)
    """
    results = []
    baseline_func = laplacian_operator if operator == 'laplacian' else biharmonic_operator

    x = tf.cast(x, dtype)
    start_time = time.time()
    baseline_value = baseline_func(model, x)
    comp_time = time.time() - start_time

    value_item = float(baseline_value.numpy())
    results.append({
        'mode': 'reverse_reverse',
        'operator': operator,
        'time': float(comp_time),
        'value': value_item,
        'mse': 0.0
    })
    return results


def evaluate_performance_tf(verbose=True):
    """
    Main TensorFlow performance evaluation loop
    Mirrors the structure of evaluation.py for PyTorch
    """
    results = []
    total_configs = 0

    def build_model(model_name, input_dim, hidden_dim, num_layers, activation):
        if model_name == 'MLP':
            return MLP(input_dim=input_dim,
                       hidden_dims=[hidden_dim] * num_layers,
                       output_dim=1,
                       activation=activation)
        elif model_name == 'ResNet':
            return ResNet(input_dim=input_dim,
                          hidden_dims=[hidden_dim] * num_layers,
                          output_dim=1,
                          activation=activation)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    for model_name in ['MLP', 'ResNet']:
        for input_dim in INPUT_DIMS:
            for hidden_dim in HIDDEN_DIMS:
                for num_layers in HIDDEN_LAYERS:
                    for activation in ACTIVATIONS:
                        for precision in PRECISIONS:
                            dtype = tf.float32 if precision == 'float32' else tf.float64
                            if verbose:
                                print(f"Testing: {model_name}, dim={input_dim}, hidden={hidden_dim}, "
                                      f"layers={num_layers}, activation={activation}, precision={precision}")

                            lap_runs, bih_runs = [], []
                            for _ in range(NUM_RUNS):
                                tf.random.set_seed(42)
                                model = build_model(model_name, input_dim, hidden_dim, num_layers, activation)
                                x = tf.random.normal((1, input_dim), dtype=dtype)
                                lap_runs.extend(evaluate_single_config_tf(model, x, operator='laplacian', dtype=dtype))
                                model = build_model(model_name, input_dim, hidden_dim, num_layers, activation)
                                x = tf.random.normal((1, input_dim), dtype=dtype)
                                bih_runs.extend(evaluate_single_config_tf(model, x, operator='biharmonic', dtype=dtype))

                            for r in lap_runs + bih_runs:
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
        print(f"\n✓ TensorFlow evaluation done for {total_configs} configs")
        print(f"✓ Total results: {len(results)}")

    return results


def run_benchmark_tf():
    """Run full TensorFlow benchmark and save results"""
    print("=" * 80)
    print("TensorFlow Performance Evaluation")
    print("=" * 80)
    results = evaluate_performance_tf(verbose=True)

    print("\nSaving TensorFlow results...")
    save_results_to_csv(results, "results_tf.csv")
    print("✓ Results saved as results_tf.csv")

    print("\nGenerating TensorFlow performance plot...")
    plot_performance(results, output_filename="performance_tf.png")
    print("✓ Plot saved as performance_tf.png")

    return results


if __name__ == "__main__":
    run_benchmark_tf()
