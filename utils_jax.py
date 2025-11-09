"""
utils_jax.py
Same plotting and CSV saving helpers as in utils.py, keeps three subplots
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

def save_results_to_csv(results, filename):
    fieldnames = ['model', 'input_dim', 'hidden_dim', 'num_layers', 'activation', 'precision', 'operator', 'mode', 'time', 'value', 'mse','baseline_f_time', 'baseline_grad_time', 'framework']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {field: r.get(field, '') for field in fieldnames}
            writer.writerow(row)

def plot_performance(results, output_filename="performance_jax.png"):
    """
    Three subplots as in torch:
    1) Laplacian: time vs input dim (lines per mode)
    2) MSE vs AD mode (log scale)
    3) Average time vs AD mode (bar)
    Expects 'results' as list of dicts similar to PyTorch results.
    """
    import numpy as np
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]

    # Subplot 1
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    for mode in modes:
        mode_results = [r for r in results if r.get('mode') == mode and r.get('operator') == 'laplacian']
        if not mode_results:
            continue
        input_dims = [r['input_dim'] for r in mode_results]
        times = [r['time'] for r in mode_results]
        plt.plot(input_dims, times, 'o-', label=mode)
    plt.xlabel('Input Dimension')
    plt.ylabel('Computation Time (s)')
    plt.title('Laplacian Operator: Time vs Input Dimension')
    plt.legend()
    plt.grid(True)

    # Subplot 2: MSE vs mode
    plt.subplot(1, 3, 2)
    mode_mses = {}
    for mode in modes[1:]:
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
        if any_positive:
            plt.bar(modes_list, mse_values)
            plt.yscale('log')
            plt.ylabel('Mean Squared Error (log scale)')
        else:
            eps = 1e-12
            plt.bar(modes_list, [eps for _ in modes_list])
            plt.ylim(0, 1e-10)
            plt.ylabel('Mean Squared Error')
    plt.xlabel('AD Mode')
    plt.title('MSE vs AD Mode (relative to reverse_reverse)')
    plt.grid(True, axis='y')

    # Subplot 3: average time vs mode
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
    plt.savefig(output_filename, dpi=150)
    plt.show()
