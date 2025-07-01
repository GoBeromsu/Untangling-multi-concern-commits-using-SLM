#!/bin/bash

# Generate Concern Combinations Dataset Script
# This script generates various combinations of concern types for evaluation

set -e  # Exit on any error

# Configuration
DATASET_PATH="datasets/candidates/codefuse-hqcm/dataset/test.json"
OUTPUT_DIR="datasets/tangled/codefuse-hqcm"
NUM_CASES=10
SEED=42

echo "🚀 Generating Concern Combination Datasets..."
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📊 Cases per combination: ${NUM_CASES}"
echo

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Define concern types by category
PURPOSE_TYPES=("feat" "fix" "style" "refactor")
OBJECT_TYPES=("docs" "test" "cicd" "build")

echo "=== 1. Object (Single) ==="
for type in "${OBJECT_TYPES[@]}"; do
    echo "🔄 Generating: ${type}"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${type}" \
        --concerns 1 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 2. Purpose (Single) ==="
for type in "${PURPOSE_TYPES[@]}"; do
    echo "🔄 Generating: ${type}"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${type}" \
        --concerns 1 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 3. Object + Object ==="
declare -a object_pairs=(
    "docs,test"
    "docs,cicd"
    "docs,build"
    "test,cicd"
    "test,build"
    "cicd,build"
)

for pair in "${object_pairs[@]}"; do
    echo "🔄 Generating: ${pair} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${pair}" \
        --concerns 2 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 4. Object + Purpose ==="
declare -a object_purpose_pairs=(
    "docs,feat"
    "docs,fix"
    "test,feat"
    "test,fix"
    "cicd,refactor"
    "build,style"
)

for pair in "${object_purpose_pairs[@]}"; do
    echo "🔄 Generating: ${pair} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${pair}" \
        --concerns 2 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 5. Purpose + Purpose ==="
declare -a purpose_pairs=(
    "feat,fix"
    "feat,style"
    "feat,refactor"
    "fix,style"
    "fix,refactor"
    "style,refactor"
)

for pair in "${purpose_pairs[@]}"; do
    echo "🔄 Generating: ${pair} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${pair}" \
        --concerns 2 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 6. Purpose x 3 ==="
declare -a purpose_triples=(
    "feat,fix,style"
    "feat,fix,refactor"
    "feat,style,refactor"
    "fix,style,refactor"
)

for triple in "${purpose_triples[@]}"; do
    echo "🔄 Generating: ${triple} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${triple}" \
        --concerns 3 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 7. Object x 3 ==="
declare -a object_triples=(
    "docs,test,cicd"
    "docs,test,build"
    "docs,cicd,build"
    "test,cicd,build"
)

for triple in "${object_triples[@]}"; do
    echo "🔄 Generating: ${triple} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${triple}" \
        --concerns 3 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 8. Purpose x 2 + Object ==="
declare -a purpose2_object1=(
    "feat,fix,docs"
    "feat,style,test"
    "feat,refactor,cicd"
    "fix,style,build"
    "fix,refactor,docs"
    "style,refactor,test"
)

for combo in "${purpose2_object1[@]}"; do
    echo "🔄 Generating: ${combo} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${combo}" \
        --concerns 3 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "=== 9. Purpose + Object x 2 ==="
declare -a purpose1_object2=(
    "feat,docs,test"
    "fix,docs,cicd"
    "style,test,build"
    "refactor,cicd,build"
    "feat,test,cicd"
    "fix,docs,build"
)

for combo in "${purpose1_object2[@]}"; do
    echo "🔄 Generating: ${combo} (unique types)"
    python scripts/codefuse-hqcm/generate_tangled.py \
        --dataset-path "${DATASET_PATH}" \
        --types "${combo}" \
        --concerns 3 \
        --num-cases "${NUM_CASES}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
done

echo
echo "✅ Dataset generation completed!"
echo "📂 Generated files are in: ${OUTPUT_DIR}"
echo "🔍 Use the following command to see all generated files:"
echo "    find ${OUTPUT_DIR} -name '*.json' -type f | sort" 