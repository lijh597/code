"""
evaluation_tf.py
Benchmarking for TensorFlow version (Laplacian & Biharmonic operators)
Outputs results_tf.csv and performance_tf.png
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  #to overcome the libiomp5 error from tf and matplotlib
# fyi: the other warning regaring the tensor flow gradient calling, is, as read, intentional for we are doing higher ones
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
    This function:
      - casts input to desired dtype,
      - computes operator value (laplacian or biharmonic),
      - measures wall-clock time for the operator call,
      - returns results in a list of dicts (compatible with the torch results format)
    """
    results = []
    baseline_func = laplacian_operator if operator == 'laplacian' else biharmonic_operator

    x = tf.cast(x, dtype) #ensure input is correct dtype
    start_time = time.time() # time measure
    baseline_value = baseline_func(model, x)
    comp_time = time.time() - start_time

    value_item = float(baseline_value.numpy()) # convert to python float for CSV-safeness
    # single entry per configuration; mode field kept as 'reverse_reverse'
    results.append({
        'mode': 'reverse_reverse',
        'operator': operator,
        'time': float(comp_time),
        'value': value_item,
        'mse': 0.0 # tf only has one mode, MSE vs baseline is zero by definition
    })
    return results


def evaluate_performance_tf(verbose=True):
    """
    Main TensorFlow performance evaluation loop that mirrors the structure of evaluation.py for PyTorch
    Loops over all configurations (input_dim, hidden_dim, num_layers, activation, precision)
    and runs evaluate_single_config_tf NUM_RUNS times to collect timing/value data
    """
    results = []
    total_configs = 0

    # Helper to build models based on string name
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

    # iterate over both model types and all parameter combinations
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
                            # repeat runs for averaging and to reduce noise
                            for _ in range(NUM_RUNS):
                                tf.random.set_seed(42) # seed 42 oc
                                # build model fresh to avoid reusing weights/graphs between runs
                                model = build_model(model_name, input_dim, hidden_dim, num_layers, activation)
                                # random input vector (batch size 1)
                                x = tf.random.normal((1, input_dim), dtype=dtype)
                                # evaluate Laplacian and Biharmonic and collect results
                                lap_runs.extend(evaluate_single_config_tf(model, x, operator='laplacian', dtype=dtype))
                                # rebuild model and sample
                                model = build_model(model_name, input_dim, hidden_dim, num_layers, activation)
                                x = tf.random.normal((1, input_dim), dtype=dtype)
                                bih_runs.extend(evaluate_single_config_tf(model, x, operator='biharmonic', dtype=dtype))

                            # tag each result
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
    """Run full TensorFlow benchmark and save results in csv and plot"""
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
