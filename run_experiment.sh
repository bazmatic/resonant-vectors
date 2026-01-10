#!/bin/bash

# Script to run an experiment and automatically process metrics
# Usage: ./run_experiment.sh <experiment_name> [--no-clear]

set -e  # Exit on error

# Check if experiment name is provided
if [ -z "$1" ]; then
    echo "Error: Experiment name is required"
    echo "Usage: $0 <experiment_name> [--no-clear]"
    exit 1
fi

EXPERIMENT_NAME="$1"
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"

# Check for --no-clear flag
NO_CLEAR_FLAG=""
if [ "$2" == "--no-clear" ]; then
    NO_CLEAR_FLAG="--no-clear"
fi

# Create experiment directory
mkdir -p "${EXPERIMENT_DIR}"

echo "=========================================="
echo "Starting experiment: ${EXPERIMENT_NAME}"
if [ -n "$NO_CLEAR_FLAG" ]; then
    echo "Database will NOT be cleared"
fi
echo "=========================================="
echo ""

# Run the experiment
echo "Running training..."
python main.py ${NO_CLEAR_FLAG}

# Check if metrics file was created
if [ ! -f "training_metrics.json" ]; then
    echo "Error: training_metrics.json was not created"
    exit 1
fi

echo ""
echo "=========================================="
echo "Processing metrics..."
echo "=========================================="

# Process metrics and generate plots
python metrics_plotter.py training_metrics.json "${EXPERIMENT_DIR}"

# Copy metrics file to experiment directory
cp training_metrics.json "${EXPERIMENT_DIR}/training_metrics.json"

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo "Results saved to: ${EXPERIMENT_DIR}"
echo ""
