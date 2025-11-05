"""
compare_frameworks.py
Load PyTorch results (results.csv) and TensorFlow results (results_tf.csv),
combine them, save combined CSV, and draw comparison plots.

Usage:
    python compare_frameworks.py
Outputs:
    - results_combined.csv
    - compare_time_by_input_dim.png
    - compare_time_by_mode.png
    - compare_mse_by_mode.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Filenames expected
TORCH_CSV = "results.csv"
TF_CSV = "results_tf.csv"
OUT_CSV = "results_combined.csv"

def load_if_exists(fname, framework_label):
    if not os.path.exists(fname):
        print(f"⚠ file {fname} not found, skipping {framework_label}.")
        return pd.DataFrame()
    df = pd.read_csv(fname)
    if 'framework' not in df.columns:
        df['framework'] = framework_label
    return df

def combine_and_save():
    # Load both (if present)
    df_torch = load_if_exists(TORCH_CSV, 'pytorch')
    df_tf = load_if_exists(TF_CSV, 'tensorflow')

    if df_torch.empty and df_tf.empty:
        print("No input CSV files found (results.csv or results_tf.csv). Nothing to do.")
        return None

    df = pd.concat([df_torch, df_tf], ignore_index=True, sort=False)

    # Try to normalize some columns (e.g., ensure numeric types)
    for col in ['input_dim', 'hidden_dim', 'num_layers', 'time', 'value', 'mse']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_csv(OUT_CSV, index=False)
    print(f"Combined results saved to {OUT_CSV}")
    return df

def plot_time_by_input_dim(df, output="compare_time_by_input_dim.png"):
    """
    Plot computation time vs input dimension for Laplacian,
    grouped by framework & mode (one plot).
    """
    df_plot = df[(df['operator'] == 'laplacian') & df['time'].notna()]
    if df_plot.empty:
        print("No laplacian timing data found to plot.")
        return

    # Plot settings
    plt.figure(figsize=(10,6))
    markers = {'pytorch':'o', 'tensorflow':'s'}
    modes = df_plot['mode'].unique()
    frameworks = df_plot['framework'].unique()

    for framework in frameworks:
        for mode in modes:
            subset = df_plot[(df_plot['framework']==framework) & (df_plot['mode']==mode)]
            if subset.empty:
                continue
            # group by input_dim mean
            grp = subset.groupby('input_dim')['time'].mean().reset_index()
            plt.plot(grp['input_dim'], grp['time'], marker=markers.get(framework,'o'),
                     label=f"{framework} | {mode}")

    plt.xlabel('Input dimension')
    plt.ylabel('Mean computation time (s)')
    plt.title('Laplacian time vs input dim (by framework & AD mode)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")

def plot_time_by_mode(df, output="compare_time_by_mode.png"):
    """
    Bar chart: average time per mode, separate bars for frameworks.
    """
    df_plot = df[(df['operator'] == 'laplacian') & df['time'].notna()]
    if df_plot.empty:
        print("No laplacian timing data found to plot.")
        return

    summary = df_plot.groupby(['framework','mode'])['time'].mean().unstack(level=0)
    summary = summary.fillna(0)

    summary.plot(kind='bar', figsize=(10,6))
    plt.ylabel('Mean computation time (s)')
    plt.title('Average Time by AD Mode (per framework)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")

def plot_mse_by_mode(df, output="compare_mse_by_mode.png"):
    """
    Bar chart: average MSE per mode (relative to baseline), compare frameworks
    """
    df_plot = df[(df['operator'] == 'laplacian') & df['mse'].notna()]
    if df_plot.empty:
        print("No MSE data found to plot.")
        return

    summary = df_plot.groupby(['framework','mode'])['mse'].mean().unstack(level=0)
    summary = summary.fillna(0)

    summary.plot(kind='bar', figsize=(10,6))
    plt.ylabel('Mean Squared Error')
    plt.title('Average MSE by AD Mode (per framework)')
    plt.yscale('log')  # MSEs typically vary orders of magnitude
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")

def main():
    df = combine_and_save()
    if df is None:
        return
    plot_time_by_input_dim(df)
    plot_time_by_mode(df)
    plot_mse_by_mode(df)
    print("All comparison plots saved.")

if __name__ == "__main__":
    main()
