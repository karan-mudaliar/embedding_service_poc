"""
GPU monitoring utility for tracking GPU metrics during stress tests.
Runs in the background and samples GPU metrics at regular intervals.
"""

import time
import json
import structlog
import pynvml
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field


logger = structlog.get_logger()


class GPUMetrics(BaseModel):
    """GPU metrics at a point in time."""
    timestamp: float = Field(..., description="Unix timestamp")
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    memory_used_mb: float = Field(..., description="GPU memory used in MB")
    memory_total_mb: float = Field(..., description="Total GPU memory in MB")
    memory_utilization: float = Field(..., description="Memory utilization percentage")
    temperature: float = Field(..., description="GPU temperature in Celsius")
    power_usage_w: float = Field(..., description="Power usage in watts")


class GPUMonitor:
    """Background GPU monitoring."""

    def __init__(self, sample_interval: float = 5.0, output_file: str = "gpu_metrics.json"):
        """
        Initialize GPU monitor.

        Args:
            sample_interval: Seconds between samples
            output_file: File to save GPU metrics
        """
        self.sample_interval = sample_interval
        self.output_file = output_file
        self.metrics: List[GPUMetrics] = []
        self.running = False

        # Initialize NVML
        try:
            import os
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
            logger.info("cuda_visible_devices", value=cuda_visible)

            pynvml.nvmlInit()

            # Get number of GPUs
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info("gpu_devices_found", count=device_count)

            if device_count == 0:
                raise RuntimeError("No GPU devices found")

            # Find first available GPU (SLURM might not allocate GPU 0)
            self.handle = None
            for i in range(device_count):
                try:
                    self.handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    device_name = pynvml.nvmlDeviceGetName(self.handle)
                    logger.info("gpu_monitor_initialized", gpu_index=i, device=device_name)
                    break
                except Exception as e:
                    logger.warning("gpu_not_accessible", gpu_index=i, error=str(e))
                    continue

            if self.handle is None:
                raise RuntimeError("No accessible GPU found")

        except Exception as e:
            logger.error("gpu_monitor_init_failed", error=str(e))
            raise

    def sample_gpu(self) -> GPUMetrics:
        """Sample current GPU metrics."""
        try:
            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

            # Get memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            # Get temperature
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            # Get power usage
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert mW to W

            metrics = GPUMetrics(
                timestamp=time.time(),
                gpu_utilization=float(utilization.gpu),
                memory_used_mb=float(memory.used / 1024 / 1024),
                memory_total_mb=float(memory.total / 1024 / 1024),
                memory_utilization=float(memory.used / memory.total * 100),
                temperature=float(temperature),
                power_usage_w=float(power),
            )

            return metrics
        except Exception as e:
            logger.error("gpu_sample_failed", error=str(e))
            raise

    def start(self):
        """Start monitoring (blocks until stopped)."""
        self.running = True
        logger.info("gpu_monitor_started", sample_interval=self.sample_interval)

        try:
            while self.running:
                metrics = self.sample_gpu()
                self.metrics.append(metrics)

                logger.debug(
                    "gpu_sample",
                    gpu_util=metrics.gpu_utilization,
                    mem_util=metrics.memory_utilization,
                    temp=metrics.temperature,
                )

                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            logger.info("gpu_monitor_interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop monitoring and save metrics."""
        self.running = False
        logger.info("gpu_monitor_stopped", total_samples=len(self.metrics))

        # Save metrics to file
        self.save_metrics()

        # Cleanup NVML
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def save_metrics(self):
        """Save collected metrics to JSON file."""
        if not self.metrics:
            logger.warning("no_gpu_metrics_to_save")
            return

        metrics_data = [m.dict() for m in self.metrics]

        with open(self.output_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info("gpu_metrics_saved", output_file=self.output_file, num_samples=len(self.metrics))

    def get_summary(self) -> Dict:
        """Get summary statistics of collected metrics."""
        if not self.metrics:
            return {}

        gpu_utils = [m.gpu_utilization for m in self.metrics]
        mem_utils = [m.memory_utilization for m in self.metrics]
        temps = [m.temperature for m in self.metrics]
        powers = [m.power_usage_w for m in self.metrics]

        summary = {
            "num_samples": len(self.metrics),
            "duration_seconds": self.metrics[-1].timestamp - self.metrics[0].timestamp if len(self.metrics) > 1 else 0,
            "gpu_utilization": {
                "mean": sum(gpu_utils) / len(gpu_utils),
                "min": min(gpu_utils),
                "max": max(gpu_utils),
            },
            "memory_utilization": {
                "mean": sum(mem_utils) / len(mem_utils),
                "min": min(mem_utils),
                "max": max(mem_utils),
            },
            "temperature": {
                "mean": sum(temps) / len(temps),
                "min": min(temps),
                "max": max(temps),
            },
            "power_usage_watts": {
                "mean": sum(powers) / len(powers),
                "min": min(powers),
                "max": max(powers),
            },
        }

        return summary


if __name__ == "__main__":
    # Test GPU monitor
    monitor = GPUMonitor(sample_interval=1.0, output_file="test_gpu_metrics.json")

    logger.info("testing_gpu_monitor_for_10_seconds")

    import threading

    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor.start)
    monitor_thread.start()

    # Let it run for 10 seconds
    time.sleep(10)

    # Stop monitoring
    monitor.stop()
    monitor_thread.join()

    # Print summary
    summary = monitor.get_summary()
    logger.info("gpu_monitor_summary", **summary)
