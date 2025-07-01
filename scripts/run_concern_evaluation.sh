#!/bin/bash

# Complete Concern Classification Evaluation Pipeline
# This script generates datasets and runs comprehensive evaluation

set -e  # Exit on any error

echo "🌩️ Concern is All You Need - Evaluation Pipeline"
echo "================================================="
echo

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${PROJECT_ROOT}/results"
OUTPUT_FILE="${RESULTS_DIR}/concern_evaluation_${TIMESTAMP}.md"

echo "📁 Project root: ${PROJECT_ROOT}"
echo "📊 Results will be saved to: ${OUTPUT_FILE}"
echo

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Step 1: Generate datasets
echo "🔧 Step 1: Generating concern combination datasets..."
echo "=================================================="
cd "${PROJECT_ROOT}"

chmod +x "${SCRIPT_DIR}/generate_concern_combinations.sh"
"${SCRIPT_DIR}/generate_concern_combinations.sh"

echo
echo "✅ Dataset generation completed!"

# Check if datasets were generated
DATASETS_DIR="${PROJECT_ROOT}/datasets/tangled/codefuse-hqcm"
DATASET_COUNT=$(find "${DATASETS_DIR}" -name "*.json" -type f | wc -l)

if [ "${DATASET_COUNT}" -eq 0 ]; then
    echo "❌ Error: No datasets were generated!"
    exit 1
fi

echo "📊 Generated ${DATASET_COUNT} dataset files"
echo

# Step 2: Run evaluation
echo "🚀 Step 2: Running concern classification evaluation..."
echo "===================================================="

# Check if .env file exists
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    echo "⚠️  Warning: .env file not found. Make sure OPENAI_API_KEY is set in environment."
fi

# Run evaluation script
cd "${PROJECT_ROOT}"
python "${SCRIPT_DIR}/evaluate_concerns.py" \
    --datasets-dir "${DATASETS_DIR}" \
    --output-file "${OUTPUT_FILE}"

echo
echo "✅ Evaluation completed!"

# Step 3: Display summary
echo "📋 Step 3: Evaluation Summary"
echo "============================="
echo "📂 Datasets evaluated: ${DATASET_COUNT}"
echo "📄 Results saved to: ${OUTPUT_FILE}"
echo "🕒 Evaluation completed at: $(date)"
echo

# Show first few lines of results
echo "📊 Preview of results:"
echo "----------------------"
head -20 "${OUTPUT_FILE}"
echo
echo "... (see full results in ${OUTPUT_FILE})"
echo

# Final instructions
echo "🎉 Evaluation pipeline completed successfully!"
echo
echo "📖 Next steps:"
echo "  1. Review the full results: cat ${OUTPUT_FILE}"
echo "  2. Copy the markdown table for your documentation"
echo "  3. Analyze performance patterns across concern compositions"
echo
echo "🔍 Quick analysis commands:"
echo "  - View full results: cat ${OUTPUT_FILE}"
echo "  - Count total evaluations: grep -c '|.*|.*|.*|.*|.*|' ${OUTPUT_FILE}"
echo "  - Find best performing types: grep -E '10{1,2}\.0%' ${OUTPUT_FILE}"
echo

# Optional: Open results file if on macOS
if command -v open &> /dev/null && [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🖥️  Opening results file..."
    open "${OUTPUT_FILE}"
fi 