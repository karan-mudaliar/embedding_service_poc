# Embedding Service Performance Comparison PoC

This project compares the performance of **HuggingFace/LangChain** vs **vLLM** backends for embedding generation using ModernBERT.

## Overview

The goal is to demonstrate how vLLM can significantly improve throughput and GPU utilization compared to the traditional HuggingFace backend for embedding workloads.

**Key Metrics:**
- Throughput (requests/second)
- Latency percentiles (P50, P90, P99)
- GPU utilization
- Success rate

## Project Structure

```
.
├── environment.yml              # Conda environment with CUDA 12.1 support
├── data_loader.py              # MS MARCO dataset loader
├── models.py                   # Pydantic models for API
├── service_huggingface.py      # HuggingFace/LangChain backend (port 8000)
├── service_vllm.py             # vLLM backend (port 8001)
├── stress_test.py              # Stress testing script with metrics
└── README.md                   # This file
```

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate embedding_poc
```

### 2. Prepare Test Dataset

Load 10,000 passages from MS MARCO dataset:

```bash
python data_loader.py
```

This will create `test_sentences.json` with the test data.

## Running the Services

### HuggingFace Backend

```bash
python service_huggingface.py
```

Service runs on `http://localhost:8000`

### vLLM Backend

```bash
python service_vllm.py
```

Service runs on `http://localhost:8001`

## Running Stress Tests

### Test HuggingFace Backend (10 minutes)

```bash
python stress_test.py --service-url http://localhost:8000 --duration-minutes 10 --batch-size 32 --max-concurrent-requests 10
```

### Test vLLM Backend (10 minutes)

```bash
python stress_test.py --service-url http://localhost:8001 --duration-minutes 10 --batch-size 32 --max-concurrent-requests 10
```

### Custom Configuration

```bash
python stress_test.py \
  --service-url http://localhost:8000 \
  --duration-minutes 5 \
  --batch-size 16 \
  --max-concurrent-requests 20 \
  --data-file test_sentences.json
```

## Results

Results are saved to `stress_test_results_{timestamp}.json` and logged using structlog.

**Example metrics:**
```json
{
  "test_duration_seconds": 600,
  "total_requests": 15000,
  "successful_requests": 14950,
  "requests_per_second": 24.92,
  "latency_metrics": {
    "mean": 0.0421,
    "p50": 0.0389,
    "p90": 0.0567,
    "p99": 0.0892
  }
}
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "backend": "vllm",
  "model": "answerdotai/ModernBERT-base",
  "gpu_available": true
}
```

### Generate Embeddings
```bash
POST /embed
```

Request:
```json
{
  "texts": ["Hello world", "This is a test"]
}
```

Response:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "model": "answerdotai/ModernBERT-base",
  "num_embeddings": 2,
  "embedding_dim": 768
}
```

## Expected Results

Based on typical vLLM optimizations, we expect:

- **2-5x higher throughput** with vLLM
- **Lower P99 latency** due to continuous batching
- **Better GPU utilization** (closer to 90%)
- **More consistent latencies** (smaller variance)

## GPU Requirements

- CUDA 12.1
- Tested on T4 GPU (Colab)
- ~4GB VRAM for ModernBERT-base

## Notes

- Data is preloaded before testing to avoid I/O bottlenecks
- Both services use the same model for fair comparison
- Concurrent requests simulate real-world load
- structlog used for structured logging (no print statements)
