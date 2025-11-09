"""
evaluation_jax.py
Benchmarking loop for JAX. Produces results_jax.csv and performance_jax.png
"""

import time
import numpy as np
import jax # pyhton 3.9 or higher for newer versions
import jax.numpy as jnp
from models_jax import init_mlp_params, mlp_apply, init_resnet_params, resnet_apply
from operators_jax import laplacian_operator, biharmonic_operator
from config import INPUT_DIMS, HIDDEN_DIMS, HIDDEN_LAYERS, ACTIVATIONS, NUM_RUNS, PRECISIONS, AD_MODES
from utils_jax import save_results_to_csv, plot_performance

def build_model_and_params(model_name, input_dim, hidden_dim, num_layers, activation, rng):
    if model_name == 'MLP':
        hidden_dims = [hidden_dim] * num_layers
        params = init_mlp_params(rng, input_dim, hidden_dims, output_dim=1)
        apply_fn = lambda p, x: mlp_apply(p, x, activation=activation)
    elif model_name == 'ResNet':
        hidden_dims = [hidden_dim] * num_layers
        params = init_resnet_params(rng, input_dim, hidden_dims, output_dim=1)
        apply_fn = lambda p, x: resnet_apply(p, x, activation=activation)
    else:
        raise ValueError("Unknown model")
    return params, apply_fn

def evaluate_single_config(params, apply_fn, x, operator='laplacian', mode='reverse_reverse'):
    """
    Compute operator (laplacian or biharmonic) under a specific AD mode.
    Measures wall time and returns value.
    """
    start = time.time()
    if operator == 'laplacian':
        val = laplacian_operator(params, apply_fn, x, mode=mode)
    else:
        val = biharmonic_operator(params, apply_fn, x, mode=mode)
    t = time.time() - start
    return float(t), float(val)

def evaluate_performance_jax(verbose=True):
    results = []
    total_configs = 0

    for model_name in ['MLP', 'ResNet']:
        for input_dim in INPUT_DIMS:
            for hidden_dim in HIDDEN_DIMS:
                for num_layers in HIDDEN_LAYERS:
                    for activation in ACTIVATIONS:
                        for precision in PRECISIONS:
                            dtype = jnp.float32 if precision == 'float32' else jnp.float64
                            if verbose:
                                print(f"Testing: {model_name}, dim={input_dim}, hidden={hidden_dim}, layers={num_layers}, activation={activation}, precision={precision}")

                            lap_runs = []
                            bih_runs = []
                            for run_i in range(NUM_RUNS):
                                # RNG for parameters
                                rng = jax.random.PRNGKey(42 + run_i)
                                params, apply_fn = build_model_and_params(model_name, input_dim, hidden_dim, num_layers, activation, rng)
                                # random input
                                x = jax.random.normal(rng, (1, input_dim), dtype=dtype)
                                # compute baseline and other modes
                                # compute reverse_reverse baseline first
                                baseline_mode = AD_MODES[0]
                                t_baseline, val_baseline = evaluate_single_config(params, apply_fn, x, operator='laplacian', mode=baseline_mode)
                                lap_runs.append({'mode': baseline_mode, 'time': t_baseline, 'value': val_baseline})
                                # other modes
                                for mode in AD_MODES[1:]:
                                    t_mode, val = evaluate_single_config(params, apply_fn, x, operator='laplacian', mode=mode)
                                    # compute simple mse vs baseline (scalar compare)
                                    mse = float((val - val_baseline) ** 2)
                                    lap_runs.append({'mode': mode, 'time': t_mode, 'value': val, 'mse': mse})
                                # biharmonic (rebuild to avoid cache interactions)
                                params2, apply_fn2 = build_model_and_params(model_name, input_dim, hidden_dim, num_layers, activation, rng)
                                x2 = jax.random.normal(rng, (1, input_dim), dtype=dtype)
                                baseline_mode = AD_MODES[0]
                                t_baseline_b, val_baseline_b = evaluate_single_config(params2, apply_fn2, x2, operator='biharmonic', mode=baseline_mode)
                                bih_runs.append({'mode': baseline_mode, 'time': t_baseline_b, 'value': val_baseline_b})
                                for mode in AD_MODES[1:]:
                                    t_mode_b, val_b = evaluate_single_config(params2, apply_fn2, x2, operator='biharmonic', mode=mode)
                                    mse_b = float((val_b - val_baseline_b) ** 2)
                                    bih_runs.append({'mode': mode, 'time': t_mode_b, 'value': val_b, 'mse': mse_b})

                            # Aggregate means
                            def aggregate(run_list):
                                import numpy as _np
                                from collections import defaultdict
                                buckets = defaultdict(list)
                                for r in run_list:
                                    key = r['mode']
                                    buckets[key].append(r)
                                out = []
                                for mode, lst in buckets.items():
                                    avg_time = float(_np.mean([z['time'] for z in lst if z.get('time') is not None]))
                                    avg_val = float(_np.mean([z['value'] for z in lst if z.get('value') is not None]))
                                    mses = [z.get('mse') for z in lst if z.get('mse') is not None]
                                    avg_mse = float(_np.mean(mses)) if mses else None
                                    out.append({'mode': mode, 'time': avg_time, 'value': avg_val, 'mse': avg_mse})
                                return out

                            lap_final = aggregate(lap_runs)
                            bih_final = aggregate(bih_runs)

                            # attach metadata, append to results
                            for r in lap_final + bih_final:
                                r.update({'model': model_name, 'input_dim': input_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'activation': activation, 'precision': precision, 'operator': 'laplacian' if r in lap_final else 'biharmonic', 'framework': 'jax'})
                                results.append(r)
                            total_configs += 1
    if verbose:
        print(f"\n✓ JAX evaluation done for {total_configs} configs")
    return results

def run_benchmark_jax():
    print("Starting JAX benchmark...")
    results = evaluate_performance_jax(verbose=True)
    save_results_to_csv(results, "results_jax.csv")
    plot_performance(results, output_filename="performance_jax.png")
    print("✓ JAX results saved: results_jax.csv and performance_jax.png")
    return results

if __name__ == "__main__":
    run_benchmark_jax()
