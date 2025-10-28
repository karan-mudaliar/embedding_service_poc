"""
Main orchestrator script for running the embedding service comparison.
Handles service lifecycle, test execution, and results aggregation.
"""

import subprocess
import time
import json
import threading
import signal
import sys
import structlog
import tyro
from typing import List
from pydantic import BaseModel, Field
from pathlib import Path
from data_loader import load_test_sentences, save_test_data
from gpu_monitor import GPUMonitor


logger = structlog.get_logger()


class ComparisonConfig(BaseModel):
    """Configuration for the comparison tests."""
    duration_minutes: int = Field(default=10, description="Test duration per configuration in minutes")
    batch_sizes: List[int] = Field(default=[16, 32, 64], description="Batch sizes to test")
    max_concurrent_requests: int = Field(default=10, description="Number of concurrent requests")
    num_sentences: int = Field(default=100000, description="Number of sentences to load from dataset")
    data_file: str = Field(default="test_sentences_100k.json", description="Dataset file")
    results_dir: str = Field(default="results", description="Directory to save results")


class ServiceManager:
    """Manages embedding service lifecycle."""

    def __init__(self, service_script: str, port: int):
        self.service_script = service_script
        self.port = port
        self.process = None

    def start(self):
        """Start the service in background."""
        logger.info("starting_service", script=self.service_script, port=self.port)

        self.process = subprocess.Popen(
            ["python", self.service_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for service to be ready
        logger.info("waiting_for_service_to_start")
        time.sleep(45)  # Give service time to load model

        logger.info("service_started", pid=self.process.pid)

    def stop(self):
        """Stop the service."""
        if self.process:
            logger.info("stopping_service", pid=self.process.pid)
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("service_did_not_stop_killing")
                self.process.kill()
                self.process.wait()

            logger.info("service_stopped")
            self.process = None


class ComparisonRunner:
    """Orchestrates the full comparison workflow."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def prepare_dataset(self):
        """Load and save dataset."""
        logger.info("preparing_dataset", num_sentences=self.config.num_sentences)

        if Path(self.config.data_file).exists():
            logger.info("dataset_already_exists", file=self.config.data_file)
            return

        sentences = load_test_sentences(num_samples=self.config.num_sentences)
        save_test_data(sentences, self.config.data_file)

        logger.info("dataset_prepared")

    def run_stress_test(self, service_url: str, batch_size: int, backend_name: str) -> str:
        """
        Run a single stress test configuration.

        Returns:
            Path to results file
        """
        logger.info(
            "running_stress_test",
            backend=backend_name,
            batch_size=batch_size,
            duration_minutes=self.config.duration_minutes,
        )

        # Start GPU monitoring in background
        gpu_output_file = str(self.results_dir / f"gpu_{backend_name}_batch{batch_size}_{int(time.time())}.json")
        gpu_monitor = GPUMonitor(sample_interval=5.0, output_file=gpu_output_file)

        monitor_thread = threading.Thread(target=gpu_monitor.start)
        monitor_thread.start()

        # Run stress test
        cmd = [
            "python",
            "stress_test.py",
            "--config.service-url",
            service_url,
            "--config.duration-minutes",
            str(self.config.duration_minutes),
            "--config.batch-size",
            str(batch_size),
            "--config.max-concurrent-requests",
            str(self.config.max_concurrent_requests),
            "--config.data-file",
            self.config.data_file,
        ]

        logger.info("executing_stress_test", command=" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Stop GPU monitoring
        gpu_monitor.stop()
        monitor_thread.join(timeout=10)

        if result.returncode != 0:
            logger.error("stress_test_failed", stderr=result.stderr)
            raise RuntimeError(f"Stress test failed: {result.stderr}")

        logger.info("stress_test_completed")

        # Find the most recent stress test results file
        import glob
        results_files = sorted(glob.glob("stress_test_results_*.json"))
        if not results_files:
            raise RuntimeError("No stress test results file found")

        latest_results = results_files[-1]

        # Move results to results directory with descriptive name
        new_results_path = self.results_dir / f"stress_{backend_name}_batch{batch_size}_{int(time.time())}.json"
        Path(latest_results).rename(new_results_path)

        return str(new_results_path)

    def test_backend(self, backend_name: str, service_script: str, port: int) -> List[str]:
        """
        Test a single backend with all batch size configurations.

        Returns:
            List of result file paths
        """
        logger.info("testing_backend", backend=backend_name)

        service = ServiceManager(service_script, port)
        service.start()

        result_files = []

        try:
            for batch_size in self.config.batch_sizes:
                results_file = self.run_stress_test(
                    f"http://localhost:{port}",
                    batch_size,
                    backend_name,
                )
                result_files.append(results_file)

                # Small delay between tests
                time.sleep(10)

        finally:
            service.stop()

        logger.info("backend_testing_complete", backend=backend_name, num_results=len(result_files))

        return result_files

    def run(self):
        """Run the full comparison."""
        logger.info("starting_comparison", config=self.config.model_dump())

        # Prepare dataset
        self.prepare_dataset()

        # Test HuggingFace backend
        logger.info("testing_huggingface_backend")
        hf_results = self.test_backend("huggingface", "service_huggingface.py", 8000)

        # Small delay between backends
        time.sleep(30)

        # Test ONNX+TensorRT backend
        logger.info("testing_onnx_tensorrt_backend")
        onnx_trt_results = self.test_backend("onnx_trt", "service_onnx_trt.py", 8002)

        # Aggregate all results
        all_results = {
            "huggingface": hf_results,
            "onnx_trt": onnx_trt_results,
        }

        # Save results index
        results_index_file = self.results_dir / "results_index.json"
        with open(results_index_file, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info("comparison_complete", results_index=str(results_index_file))

        # Generate plots
        logger.info("generating_comparison_plots")
        subprocess.run(["python", "plot_results.py", "--results-dir", str(self.results_dir)])

        logger.info("all_done")


def main(config: ComparisonConfig):
    """Main entry point."""
    runner = ComparisonRunner(config)
    runner.run()


if __name__ == "__main__":
    tyro.cli(main)
