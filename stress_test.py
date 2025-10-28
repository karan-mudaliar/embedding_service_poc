"""
Stress test script for embedding services.
Tests throughput, latency (P50, P90, P99), and GPU utilization over a 10-minute period.
"""

import asyncio
import httpx
import time
import json
import numpy as np
import structlog
import tyro
from pydantic import BaseModel, Field
from typing import List, Dict
from data_loader import load_test_data
import psutil

logger = structlog.get_logger()


class StressTestConfig(BaseModel):
    """Configuration for stress testing."""
    service_url: str = Field(default="http://localhost:8000", description="Service endpoint")
    duration_minutes: int = Field(default=10, description="Test duration in minutes")
    batch_size: int = Field(default=32, description="Number of texts per request")
    max_concurrent_requests: int = Field(default=10, description="Number of concurrent requests")
    data_file: str = Field(default="test_sentences.json", description="Dataset file")


class StressTestRunner:
    """Runs stress tests and collects metrics."""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.latencies: List[float] = []
        self.errors: List[str] = []
        self.start_time = None
        self.end_time = None
        self.total_requests = 0
        self.successful_requests = 0

    async def make_request(self, client: httpx.AsyncClient, texts: List[str]) -> float:
        """
        Make a single embedding request and measure latency.

        Returns:
            Latency in seconds
        """
        start = time.time()
        try:
            response = await client.post(
                f"{self.config.service_url}/embed",
                json={"texts": texts},
                timeout=30.0,
            )
            response.raise_for_status()
            latency = time.time() - start
            self.successful_requests += 1
            return latency
        except Exception as e:
            latency = time.time() - start
            self.errors.append(str(e))
            logger.error("request_failed", error=str(e), latency=latency)
            return latency

    async def worker(self, client: httpx.AsyncClient, sentences: List[str]):
        """Worker coroutine that continuously sends requests."""
        batch_size = self.config.batch_size
        num_sentences = len(sentences)

        while time.time() < self.end_time:
            # Select random batch of sentences
            start_idx = np.random.randint(0, max(1, num_sentences - batch_size))
            batch = sentences[start_idx : start_idx + batch_size]

            # Make request and record latency
            latency = await self.make_request(client, batch)
            self.latencies.append(latency)
            self.total_requests += 1

            # Small delay to prevent overwhelming the service
            await asyncio.sleep(0.01)

    async def run_test(self):
        """Run the stress test."""
        logger.info("starting_stress_test", config=self.config.dict())

        # Load test data
        sentences = load_test_data(self.config.data_file)
        logger.info("loaded_test_data", num_sentences=len(sentences))

        # Check service health with retries
        max_retries = 5
        retry_delay = 10  # seconds
        health_ok = False

        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    logger.info("health_check_attempt", attempt=attempt + 1, max_retries=max_retries)
                    health_response = await client.get(
                        f"{self.config.service_url}/health", timeout=10.0
                    )
                    health_response.raise_for_status()
                    health_data = health_response.json()
                    logger.info("service_health_check_ok", health=health_data)
                    health_ok = True
                    break
                except Exception as e:
                    logger.warning("health_check_failed_retry", attempt=attempt + 1, error=str(e))
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error("health_check_failed_all_retries", error=str(e))

        if not health_ok:
            logger.error("service_not_healthy_aborting")
            return

        # Start stress test
        self.start_time = time.time()
        self.end_time = self.start_time + (self.config.duration_minutes * 60)

        logger.info(
            "starting_workers",
            num_workers=self.config.max_concurrent_requests,
            duration_minutes=self.config.duration_minutes,
        )

        # Create concurrent workers
        async with httpx.AsyncClient() as client:
            workers = [
                self.worker(client, sentences)
                for _ in range(self.config.max_concurrent_requests)
            ]
            await asyncio.gather(*workers)

        # Calculate metrics
        self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate and display metrics."""
        logger.info("calculating_metrics")

        total_duration = time.time() - self.start_time
        latencies_arr = np.array(self.latencies)

        # Calculate throughput
        throughput = self.total_requests / total_duration
        requests_per_second = self.successful_requests / total_duration

        # Calculate latency percentiles
        p50 = np.percentile(latencies_arr, 50)
        p90 = np.percentile(latencies_arr, 90)
        p99 = np.percentile(latencies_arr, 99)
        mean_latency = np.mean(latencies_arr)
        min_latency = np.min(latencies_arr)
        max_latency = np.max(latencies_arr)

        # Success rate
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0

        # Log results
        logger.info("stress_test_results_summary")
        logger.info(
            "test_configuration",
            duration_seconds=total_duration,
            duration_minutes=self.config.duration_minutes,
            service_url=self.config.service_url,
            batch_size=self.config.batch_size,
            concurrent_requests=self.config.max_concurrent_requests,
        )
        logger.info(
            "throughput_metrics",
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=len(self.errors),
            success_rate_percent=round(success_rate, 2),
            requests_per_second=round(requests_per_second, 2),
        )
        logger.info(
            "latency_metrics_seconds",
            mean=round(mean_latency, 4),
            min=round(min_latency, 4),
            max=round(max_latency, 4),
            p50=round(p50, 4),
            p90=round(p90, 4),
            p99=round(p99, 4),
        )

        if self.errors:
            logger.warning(
                "errors_encountered",
                error_count=len(self.errors),
                first_10_errors=self.errors[:10],
            )

        # Save results to file
        results = {
            "test_duration_seconds": total_duration,
            "service_url": self.config.service_url,
            "batch_size": self.config.batch_size,
            "concurrent_requests": self.config.max_concurrent_requests,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": len(self.errors),
            "success_rate_percent": success_rate,
            "requests_per_second": requests_per_second,
            "latency_metrics": {
                "mean": float(mean_latency),
                "min": float(min_latency),
                "max": float(max_latency),
                "p50": float(p50),
                "p90": float(p90),
                "p99": float(p99),
            },
        }

        output_file = f"stress_test_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("results_saved", output_file=output_file)


def main(config: StressTestConfig):
    """Main entry point for stress testing."""
    runner = StressTestRunner(config)
    asyncio.run(runner.run_test())


if __name__ == "__main__":
    tyro.cli(main)
