"""
Generate comparison plots from stress test results.
"""

import json
import glob
import structlog
import tyro
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


logger = structlog.get_logger()
sns.set_style("whitegrid")


def load_results(results_dir: str) -> Dict:
    """Load all stress test and GPU results."""
    results_dir = Path(results_dir)

    # Load stress test results
    stress_files = sorted(results_dir.glob("stress_*.json"))
    gpu_files = sorted(results_dir.glob("gpu_*.json"))

    data = {
        "huggingface": {"stress": [], "gpu": []},
        "onnx_trt": {"stress": [], "gpu": []},
    }

    for file in stress_files:
        with open(file) as f:
            result = json.load(f)
            if "huggingface" in file.name:
                backend = "huggingface"
            elif "onnx_trt" in file.name:
                backend = "onnx_trt"
            else:
                continue
            data[backend]["stress"].append(result)

    for file in gpu_files:
        with open(file) as f:
            result = json.load(f)
            if "huggingface" in file.name:
                backend = "huggingface"
            elif "onnx_trt" in file.name:
                backend = "onnx_trt"
            else:
                continue
            data[backend]["gpu"].append(result)

    logger.info(
        "loaded_results",
        hf_stress=len(data["huggingface"]["stress"]),
        hf_gpu=len(data["huggingface"]["gpu"]),
        onnx_trt_stress=len(data["onnx_trt"]["stress"]),
        onnx_trt_gpu=len(data["onnx_trt"]["gpu"]),
    )

    return data


def plot_throughput_comparison(data: Dict, output_file: str):
    """Plot throughput (RPS) comparison by batch size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    backends = ["huggingface", "onnx_trt"]
    colors = {"huggingface": "#FF6B6B", "onnx_trt": "#4ECDC4"}
    labels = {"huggingface": "HuggingFace/LangChain", "onnx_trt": "ONNX+TensorRT"}

    for backend in backends:
        stress_results = sorted(data[backend]["stress"], key=lambda x: x["batch_size"])
        batch_sizes = [r["batch_size"] for r in stress_results]
        rps = [r["requests_per_second"] for r in stress_results]

        ax.plot(batch_sizes, rps, marker='o', linewidth=2, markersize=8,
                color=colors[backend], label=labels[backend])

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Requests per Second", fontsize=12)
    ax.set_title("Throughput Comparison: HuggingFace vs ONNX+TensorRT", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("plot_saved", file=output_file)


def plot_latency_comparison(data: Dict, output_file: str):
    """Plot latency percentiles comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    percentiles = ["p50", "p90", "p99"]
    titles = ["P50 Latency", "P90 Latency", "P99 Latency"]

    backends = ["huggingface", "onnx_trt"]
    colors = {"huggingface": "#FF6B6B", "onnx_trt": "#4ECDC4"}
    labels = {"huggingface": "HuggingFace/LangChain", "onnx_trt": "ONNX+TensorRT"}

    for idx, (percentile, title) in enumerate(zip(percentiles, titles)):
        ax = axes[idx]

        for backend in backends:
            stress_results = sorted(data[backend]["stress"], key=lambda x: x["batch_size"])
            batch_sizes = [r["batch_size"] for r in stress_results]
            latencies = [r["latency_metrics"][percentile] for r in stress_results]

            ax.plot(batch_sizes, latencies, marker='o', linewidth=2, markersize=8,
                    color=colors[backend], label=labels[backend])

        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_ylabel("Latency (seconds)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("plot_saved", file=output_file)


def plot_gpu_utilization(data: Dict, output_file: str):
    """Plot GPU utilization comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    backends = ["huggingface", "onnx_trt"]
    colors = {"huggingface": "#FF6B6B", "onnx_trt": "#4ECDC4"}
    labels = {"huggingface": "HuggingFace/LangChain", "onnx_trt": "ONNX+TensorRT"}

    for backend in backends:
        gpu_results = data[backend]["gpu"]
        if not gpu_results:
            continue

        # Calculate average GPU utilization for each configuration
        avg_utils = []
        batch_sizes = []

        # Group by batch size (infer from filename or use index)
        for i, gpu_data in enumerate(gpu_results):
            gpu_utils = [sample["gpu_utilization"] for sample in gpu_data]
            avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
            avg_utils.append(avg_util)

            # Try to extract batch size from stress results
            if i < len(data[backend]["stress"]):
                batch_sizes.append(data[backend]["stress"][i]["batch_size"])
            else:
                batch_sizes.append((i + 1) * 16)  # Fallback

        ax.plot(batch_sizes, avg_utils, marker='o', linewidth=2, markersize=8,
                color=colors[backend], label=labels[backend])

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Average GPU Utilization (%)", fontsize=12)
    ax.set_title("GPU Utilization Comparison", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("plot_saved", file=output_file)


def generate_summary_report(data: Dict, output_file: str):
    """Generate a text summary report."""
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EMBEDDING SERVICE PERFORMANCE COMPARISON\n")
        f.write("HuggingFace/LangChain vs ONNX+TensorRT\n")
        f.write("=" * 70 + "\n\n")

        for backend in ["huggingface", "onnx_trt"]:
            backend_name = "HuggingFace/LangChain" if backend == "huggingface" else "ONNX+TensorRT"
            f.write(f"\n{backend_name} Backend:\n")
            f.write("-" * 70 + "\n")

            stress_results = sorted(data[backend]["stress"], key=lambda x: x["batch_size"])

            for result in stress_results:
                f.write(f"\nBatch Size: {result['batch_size']}\n")
                f.write(f"  Requests/sec: {result['requests_per_second']:.2f}\n")
                f.write(f"  Success rate: {result['success_rate_percent']:.2f}%\n")
                f.write(f"  Latency P50: {result['latency_metrics']['p50']:.4f}s\n")
                f.write(f"  Latency P90: {result['latency_metrics']['p90']:.4f}s\n")
                f.write(f"  Latency P99: {result['latency_metrics']['p99']:.4f}s\n")

        f.write("\n" + "=" * 70 + "\n")

    logger.info("summary_report_saved", file=output_file)


def main(results_dir: str = "results"):
    """Generate all plots and reports."""
    logger.info("generating_plots", results_dir=results_dir)

    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load all results
    data = load_results(results_dir)

    # Generate plots
    plot_throughput_comparison(data, str(plots_dir / "throughput_comparison.png"))
    plot_latency_comparison(data, str(plots_dir / "latency_comparison.png"))
    plot_gpu_utilization(data, str(plots_dir / "gpu_utilization.png"))

    # Generate summary report
    generate_summary_report(data, str(results_dir / "summary_report.txt"))

    logger.info("all_plots_generated")


if __name__ == "__main__":
    tyro.cli(main)
