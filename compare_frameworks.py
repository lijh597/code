"""
compare_frameworks.py
Compares PyTorch, TensorFlow, and JAX evaluation results.

Generates:
- comparison_time_vs_mode.png
- comparison_laplacian_time_vs_input.png

Notes:
- TensorFlow has only one AD mode (reverse_reverse),
  so MSE comparison is skipped.
- All CSVs must exist in the same folder:
    results_torch.csv
    results_tf.csv
    results_jax.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utility functions ----------

def load_results(filename, framework_name):
    """Load results CSV for one framework."""
    if not os.path.exists(filename):
        print(f"⚠ Missing {filename}, skipping.")
        return []
    data = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = {
                    "framework": framework_name,
                    "operator": row.get("operator", ""),
                    "mode": row.get("mode", ""),
                    "input_dim": int(float(row.get("input_dim", 0))),
                    "time": float(row.get("time", 0.0)),
                    "mse": float(row.get("mse", 0.0)) if row.get("mse") else None,
                }
                data.append(r)
            except Exception as e:
                print(f"⚠ Skipping line in {framework_name}: {e}")
    print(f"✓ Loaded {len(data)} entries from {filename}")
    return data


def aggregate_average(data, operator="laplacian"):
    """Group by framework + mode and compute average time and mse."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in data:
        if r["operator"] == operator:
            buckets[(r["framework"], r["mode"])].append(r)
    summary = []
    for (fw, mode), lst in buckets.items():
        times = [x["time"] for x in lst if x["time"] > 0]
        mses = [x["mse"] for x in lst if x["mse"] is not None]
        avg_time = np.mean(times) if times else 0
        avg_mse = np.mean(mses) if mses else 0
        summary.append({"framework": fw, "mode": mode, "avg_time": avg_time, "avg_mse": avg_mse})
    return summary


def aggregate_time_vs_input(data, operator="laplacian"):
    """Compute average computation time vs input dimension."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in data:
        if r["operator"] == operator:
            buckets[(r["framework"], r["input_dim"])].append(r)
    summary = []
    for (fw, dim), lst in buckets.items():
        times = [x["time"] for x in lst]
        avg_time = np.mean(times)
        summary.append({"framework": fw, "input_dim": dim, "avg_time": avg_time})
    return summary


# ---------- Main comparison plotting ----------

def plot_comparisons():
    # Load all data
    torch_data = load_results("results_torch.csv", "torch")
    tf_data = load_results("results_tf.csv", "tensorflow")
    jax_data = load_results("results_jax.csv", "jax")

    all_data = torch_data + tf_data + jax_data

    # 1️⃣ Average time vs AD mode (skip tf missing modes)
    summary = aggregate_average(all_data, operator="laplacian")
    plt.figure(figsize=(8, 5))
    frameworks = sorted(list(set([r["framework"] for r in summary])))

    for fw in frameworks:
        if fw == "tensorflow":
            # Only one mode; skip or mark differently
            tf_times = [r["avg_time"] for r in summary if r["framework"] == fw]
            tf_modes = [r["mode"] for r in summary if r["framework"] == fw]
            plt.bar([f"{fw}_{m}" for m in tf_modes], tf_times, label=fw)
        else:
            modes = [r["mode"] for r in summary if r["framework"] == fw]
            times = [r["avg_time"] for r in summary if r["framework"] == fw]
            plt.plot(modes, times, 'o-', label=fw)

    plt.title("Average Laplacian Time vs AD Mode")
    plt.xlabel("AD Mode")
    plt.ylabel("Mean Computation Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_time_vs_mode.png", dpi=150)
    plt.close()

    # 2️⃣ Laplacian time vs input dimension
    summary2 = aggregate_time_vs_input(all_data, operator="laplacian")
    plt.figure(figsize=(8, 5))
    for fw in frameworks:
        dims = [r["input_dim"] for r in summary2 if r["framework"] == fw]
        times = [r["avg_time"] for r in summary2 if r["framework"] == fw]
        plt.plot(dims, times, 'o-', label=fw)
    plt.title("Laplacian: Time vs Input Dimension")
    plt.xlabel("Input Dimension")
    plt.ylabel("Mean Computation Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_laplacian_time_vs_input.png", dpi=150)
    plt.close()

    print("✓ Comparison plots saved: comparison_time_vs_mode.png, comparison_laplacian_time_vs_input.png")


if __name__ == "__main__":
    plot_comparisons()
