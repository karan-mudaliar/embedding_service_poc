#!/bin/bash
#SBATCH --job-name=embedding_comparison
#SBATCH --partition=d2r2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=========================================="
echo "Embedding Service Comparison Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Setup environment
echo "Setting up environment..."
source setup_cluster.sh

echo ""
echo "=========================================="
echo "Running Comparison Tests"
echo "=========================================="
echo ""

# Run comparison with default settings
python run_comparison.py \
  --config.duration-minutes 10 \
  --config.batch-sizes 16 32 64 \
  --config.max-concurrent-requests 10 \
  --config.num-sentences 100000

echo ""
echo "=========================================="
echo "Job Complete"
echo "=========================================="
echo "End Time: $(date)"
echo ""
echo "Results are in the 'results/' directory"
echo "Plots are in 'results/plots/'"
echo "Summary report: 'results/summary_report.txt'"
