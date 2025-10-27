# Embedding Service Performance Comparison PoC

Comparing **HuggingFace/LangChain** vs **vLLM** backends for embedding generation using ModernBERT on SLURM cluster with L40s GPU.

## Overview

This project demonstrates how vLLM can significantly improve throughput and GPU utilization compared to the traditional HuggingFace backend for embedding workloads.

**Key Metrics Tracked:**
- Throughput (requests/second)
- Latency percentiles (P50, P90, P99)
- GPU utilization (%)
- GPU memory usage
- Success rate

## Project Structure

```
.
├── environment.yml           # Conda environment with CUDA 12.1, vLLM 0.11.0
├── setup_cluster.sh         # Environment setup script for SLURM
├── submit_job.sh            # SLURM batch job script
│
├── data_loader.py           # MS MARCO dataset loader (100K passages)
├── models.py                # Pydantic models for API
├── gpu_monitor.py           # Background GPU metrics collector
│
├── service_huggingface.py   # HuggingFace/LangChain backend (port 8000)
├── service_vllm.py          # vLLM backend (port 8001)
├── stress_test.py           # Stress testing script with metrics
│
├── run_comparison.py        # Main orchestrator script
├── plot_results.py          # Generate comparison plots
└── README.md                # This file
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/embedding_service_poc.git
cd embedding_service_poc
```

### 2. Submit SLURM Job

```bash
sbatch submit_job.sh
```

This will:
- Check/create conda environment
- Load 100K MS MARCO passages
- Test HuggingFace backend with batch sizes [16, 32, 64]
- Test vLLM backend with batch sizes [16, 32, 64]
- Generate comparison plots and summary report

### 3. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output
tail -f slurm_<JOB_ID>.out

# View errors (if any)
tail -f slurm_<JOB_ID>.err
```

### 4. View Results

Results are saved in the `results/` directory:

```bash
results/
├── stress_huggingface_batch16_*.json    # Stress test results
├── stress_huggingface_batch32_*.json
├── stress_huggingface_batch64_*.json
├── stress_vllm_batch16_*.json
├── stress_vllm_batch32_*.json
├── stress_vllm_batch64_*.json
├── gpu_huggingface_batch16_*.json       # GPU metrics
├── gpu_vllm_batch16_*.json
├── ...
├── summary_report.txt                    # Text summary
├── results_index.json                    # Index of all results
└── plots/
    ├── throughput_comparison.png
    ├── latency_comparison.png
    └── gpu_utilization.png
```

## Manual Setup (Optional)

If you want to run tests manually:

### 1. Setup Environment

```bash
source setup_cluster.sh
```

### 2. Prepare Dataset

```bash
python data_loader.py
```

### 3. Run Comparison

```bash
python run_comparison.py \
  --duration-minutes 10 \
  --batch-sizes 16 32 64 \
  --max-concurrent-requests 10 \
  --num-sentences 100000
```

### 4. Generate Plots

```bash
python plot_results.py --results-dir results
```

## Custom Configuration

### Change Test Duration

```bash
python run_comparison.py --duration-minutes 5
```

### Test Different Batch Sizes

```bash
python run_comparison.py --batch-sizes 8 16 32 64 128
```

### Adjust Concurrency

```bash
python run_comparison.py --max-concurrent-requests 20
```

### Use More Data

```bash
python run_comparison.py --num-sentences 500000
```

## API Endpoints

Both services expose the same API:

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

## Architecture

### Test Workflow

```
SLURM Job (4 hours, 1x L40s GPU, 32GB RAM, 8 CPUs)
│
├── Setup Phase
│   ├── Load modules (anaconda3, cuda/12.1)
│   ├── Check/create conda environment
│   ├── Preload 100K MS MARCO sentences
│   └── Verify GPU availability
│
├── Test Phase 1: HuggingFace Backend
│   └── For each batch_size in [16, 32, 64]:
│       ├── Start service_huggingface.py (background)
│       ├── Start GPU monitor (background)
│       ├── Run stress test (10 minutes)
│       ├── Stop GPU monitor
│       ├── Save results
│       └── Stop service
│
├── Test Phase 2: vLLM Backend
│   └── For each batch_size in [16, 32, 64]:
│       ├── Start service_vllm.py (background)
│       ├── Start GPU monitor (background)
│       ├── Run stress test (10 minutes)
│       ├── Stop GPU monitor
│       ├── Save results
│       └── Stop service
│
└── Analysis Phase
    ├── Aggregate all results
    ├── Generate comparison plots
    └── Create summary report
```

### Key Design Decisions

1. **Data Preloading**: 100K sentences loaded once into memory to avoid I/O bottlenecks
2. **Sequential Testing**: One backend at a time to avoid resource contention
3. **Background GPU Monitoring**: Samples GPU metrics every 5 seconds
4. **Same Model**: Both backends use `answerdotai/ModernBERT-base` for fair comparison
5. **Structured Logging**: All logs use structlog for consistency

## Expected Results

Based on vLLM optimizations:

- **2-5x higher throughput** with vLLM
- **Lower P99 latency** due to continuous batching
- **Better GPU utilization** (70-90% vs 30-50%)
- **More consistent latencies** (smaller variance)

## System Requirements

- SLURM cluster with GPU nodes
- L40s GPU (48GB VRAM) or similar
- CUDA 12.1
- 32GB RAM
- 8 CPU cores

## Troubleshooting

### GPU Not Visible

Check SLURM allocation includes GPU:
```bash
srun --partition=d2r2 --gres=gpu:1 --pty bash
nvidia-smi
```

### Environment Issues

Recreate environment:
```bash
conda env remove -n embedding_poc
conda env create -f environment.yml
```

### Port Already in Use

Services automatically use ports 8000 and 8001. If occupied, modify service files.

### Out of Memory

Reduce batch size or concurrent requests:
```bash
python run_comparison.py --batch-sizes 8 16 --max-concurrent-requests 5
```

## Citation

Dataset: MS MARCO (Microsoft Machine Reading Comprehension)
- 8.8M passages from Bing search
- https://microsoft.github.io/msmarco/

Model: ModernBERT by Answer.AI
- https://huggingface.co/answerdotai/ModernBERT-base

## License

MIT
