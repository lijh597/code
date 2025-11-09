"""
utils_tf.py
TensorFlow-compatible utility functions for saving results and plotting.
Identical to utils.py, but without importing torch.
"""
import numpy as np
import matplotlib.pyplot as plt

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['model', 'input_dim', 'hidden_dim', 'num_layers', 'activation',
                      'precision', 'operator', 'mode', 'time', 'value', 'mse',
                      'baseline_f_time', 'baseline_grad_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)

def plot_performance(results, output_filename='performance_tf.png'):
    """Plot performance metrics"""
    modes = ["reverse_reverse", "reverse_forward", "forward_forward", "forward_reverse"]

    plt.figure(figsize=(10, 5)) # chaned from 15,5, becasue only two left

    # Subplot 1: computation time vs input dim
    plt.subplot(1, 2, 1) # again changed from 131
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

    ##### Subplot 2: MSE vs mode, not for tf becasue we only have "one" mode
    # plt.subplot(1, 3, 2)
    # mode_mses = {}
    # for mode in modes[1:]:
    #     mode_results = [r for r in results
    #                     if r.get('mode') == mode
    #                     and r.get('operator') == 'laplacian'
    #                     and r.get('mse') is not None]
    #     if mode_results:
    #         mses = [r['mse'] for r in mode_results]
    #         mode_mses[mode] = np.mean(mses)
    #
    # if mode_mses:
    #     modes_list = list(mode_mses.keys())
    #     mse_values = [mode_mses[m] for m in modes_list]
    #     plt.bar(modes_list, mse_values)
    #     plt.yscale('log')
    #     plt.ylabel('Mean Squared Error (log)')
    #     plt.xlabel('AD Mode')
    #     plt.title('MSE vs AD Mode (relative to reverse_reverse)')
    #     plt.grid(True, axis='y')

    # Subplot 3: average time vs mode
    plt.subplot(1, 2, 2) # changed from  133
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
        plt.ylabel('Mean Time (s)')
        plt.title('Average Time vs AD Mode')
        plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    #plt.show()
    plt.close('all') # so that the code has a nice exit
