# Tangled CSS Dataset: Token-Based Diff Size Analysis

## Purpose
Analysis of git diff sizes using **token count** (via tiktoken) for tangled commits to understand LLM input complexity.

## Overall Statistics

- **Total Commits**: 1050 tangled commits
- **Mean**: 1,847 tokens
- **Median**: 1,069 tokens
- **Standard Deviation**: 2,060
- **Range**: 60 - 11,864 tokens
- **Quartiles**: Q25=511, Q75=2,440
- **Percentiles**: 90th=4,426, 95th=6,635, 99th=10,343

## Token Range Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| ≤1K | 500 | 47.6% |
| 1K-2K | 237 | 22.6% |
| 2K-3K | 109 | 10.4% |
| 3K-4K | 84 | 8.0% |
| 4K-5K | 32 | 3.0% |
| >5K | 88 | 8.4% |

## LLM Input Assessment

### Token Count Summary
- **≤ 2K tokens**: 737 commits (70.2%) - Excellent for LLM input
- **2K-4K tokens**: 193 commits (18.4%) - Good for LLM input
- **> 5K tokens**: 88 commits (8.4%) - Challenging for LLM input

### Overall Assessment: ✅ **Good** - Most commits (95%) are under 8K tokens

### Detailed Distribution
- **≤1K**: 500 commits (47.6%)
- **1K-2K**: 237 commits (22.6%)
- **2K-3K**: 109 commits (10.4%)
- **3K-4K**: 84 commits (8.0%)
- **4K-5K**: 32 commits (3.0%)
- **>5K**: 88 commits (8.4%)
