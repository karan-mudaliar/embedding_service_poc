# Embedding Service Performance Comparison PoC

Comparing **HuggingFace/LangChain** vs **ONNX Runtime + TensorRT** backends for embedding generation using ModernBERT on SLURM cluster with L40s GPU.

## Overview

This project demonstrates how ONNX Runtime with TensorRT can significantly improve throughput and GPU utilization compared to the traditional HuggingFace backend for embedding workloads.

**Key Optimizations in ONNX+TensorRT:**
- FP16 precision (faster inference, half the memory)
- Kernel fusion (combines operations)
- TensorRT engine caching
- CUDA execution optimization

**Key Metrics Tracked:**
- Throughput (requests/second)
- Latency percentiles (P50, P90, P99)
- GPU utilization (%)
- GPU memory usage
- Success rate

## Project Structure

```
.
├── environment.yml           # Conda environment with CUDA 12.1, ONNX+TensorRT
├── setup_cluster.sh         # Environment setup script for SLURM
├── submit_job.sh            # SLURM batch job script
│
├── data_loader.py           # MS MARCO dataset loader (100K passages)
├── models.py                # Pydantic models for API
├── gpu_monitor.py           # Background GPU metrics collector
│
├── service_huggingface.py   # HuggingFace/LangChain backend (port 8000)
├── service_onnx_trt.py      # ONNX+TensorRT backend (port 8002)
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
- Install ONNX Runtime + TensorRT
- Load 100K MS MARCO passages
- Test HuggingFace backend with batch sizes [16, 32, 64]
- Test ONNX+TensorRT backend with batch sizes [16, 32, 64]
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
├── stress_onnx_trt_batch16_*.json
├── stress_onnx_trt_batch32_*.json
├── stress_onnx_trt_batch64_*.json
├── gpu_huggingface_batch16_*.json       # GPU metrics
├── gpu_onnx_trt_batch16_*.json
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
  --config.duration-minutes 10 \
  --config.batch-sizes 16 32 64 \
  --config.max-concurrent-requests 10 \
  --config.num-sentences 100000
```

### 4. Generate Plots

```bash
python plot_results.py --results-dir results
```

## Custom Configuration

### Change Test Duration

```bash
python run_comparison.py --config.duration-minutes 5
```

### Test Different Batch Sizes

```bash
python run_comparison.py --config.batch-sizes 8 16 32 64 128
```

### Adjust Concurrency

```bash
python run_comparison.py --config.max-concurrent-requests 20
```

### Use More Data

```bash
python run_comparison.py --config.num-sentences 500000
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
  "backend": "onnx_tensorrt",
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
│   ├── Install ONNX Runtime + TensorRT
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
├── Test Phase 2: ONNX+TensorRT Backend
│   └── For each batch_size in [16, 32, 64]:
│       ├── Start service_onnx_trt.py (background)
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

Based on ONNX+TensorRT optimizations:

- **1.5-3x higher throughput** with ONNX+TensorRT
- **Lower P99 latency** due to kernel fusion and FP16
- **Better GPU utilization** (60-80% vs 30-50%)
- **More consistent latencies** (smaller variance)
- **Lower memory usage** (FP16 uses half the memory)

## Backends Comparison

### HuggingFace/LangChain (Baseline)
- Full FP32 precision
- Standard PyTorch inference
- No special optimizations
- Port: 8000

### ONNX Runtime + TensorRT
- FP16 precision (2x faster)
- TensorRT kernel fusion
- CUDA graph optimization
- Engine caching
- Port: 8002

## System Requirements

- SLURM cluster with GPU nodes
- L40s GPU (48GB VRAM) or similar
- CUDA 12.1
- TensorRT 8.6+
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
source setup_cluster.sh
```

### TensorRT Initialization Fails

If TensorRT fails, the ONNX service will automatically fallback to CUDA execution provider. Check logs for warnings.

### Port Already in Use

Services use ports 8000 and 8002. If occupied, modify service files.

### Out of Memory

Reduce batch size or concurrent requests:
```bash
python run_comparison.py --config.batch-sizes 8 16 --config.max-concurrent-requests 5
```

## Citation

Dataset: MS MARCO (Microsoft Machine Reading Comprehension)
- 8.8M passages from Bing search
- https://microsoft.github.io/msmarco/

Model: ModernBERT by Answer.AI
- https://huggingface.co/answerdotai/ModernBERT-base

Optimization: ONNX Runtime + TensorRT
- https://onnxruntime.ai/
- https://developer.nvidia.com/tensorrt

## License

MIT
