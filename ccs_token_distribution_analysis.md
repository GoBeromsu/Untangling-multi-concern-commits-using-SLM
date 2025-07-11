# CCS Dataset: Token-Based Diff Size Analysis by Commit Type

## Purpose
Analysis of git diff sizes using **token count** (via tiktoken) by commit type to identify types with smaller diffs suitable for LLM input.

## Summary: Types Ranked by Median Token Count (Smallest to Largest)

| Rank | Type | Count | Median Tokens | Mean Tokens | 95th Percentile |
|------|------|-------|---------------|-------------|----------------|
| 1 | **chore** | 140 | 248 | 5,402 | 8,224 |
| 2 | **ci** | 140 | 262 | 484 | 1,402 |
| 3 | **docs** | 140 | 326 | 2,691 | 7,772 |
| 4 | **fix** | 140 | 432 | 26,318 | 3,371 |
| 5 | **build** | 140 | 478 | 25,693 | 82,314 |
| 6 | **style** | 140 | 550 | 9,257 | 37,504 |
| 7 | **test** | 140 | 680 | 5,979 | 7,553 |
| 8 | **refactor** | 140 | 1,128 | 2,712 | 8,700 |
| 9 | **perf** | 140 | 1,400 | 3,278 | 12,090 |
| 10 | **feat** | 140 | 1,602 | 2,939 | 8,160 |

## Token Range Distribution by Type

Distribution of diffs across token count ranges for each commit type:

| Type | ≤1K | 1K-2K | 2K-3K | 3K-4K | 4K-5K | >5K |
|------|-----|-------|-------|-------|-------|-----|
| **chore** | 114 (81.4%) | 8 (5.7%) | 5 (3.6%) | 1 (0.7%) | 3 (2.1%) | 9 (6.4%) |
| **ci** | 130 (92.9%) | 7 (5.0%) | 1 (0.7%) | 1 (0.7%) | 0 (0.0%) | 1 (0.7%) |
| **docs** | 106 (75.7%) | 8 (5.7%) | 4 (2.9%) | 3 (2.1%) | 3 (2.1%) | 16 (11.4%) |
| **fix** | 109 (77.9%) | 17 (12.1%) | 2 (1.4%) | 6 (4.3%) | 0 (0.0%) | 6 (4.3%) |
| **build** | 96 (68.6%) | 6 (4.3%) | 6 (4.3%) | 3 (2.1%) | 1 (0.7%) | 28 (20.0%) |
| **style** | 82 (58.6%) | 21 (15.0%) | 6 (4.3%) | 5 (3.6%) | 2 (1.4%) | 24 (17.1%) |
| **test** | 90 (64.3%) | 30 (21.4%) | 10 (7.1%) | 1 (0.7%) | 0 (0.0%) | 9 (6.4%) |
| **refactor** | 68 (48.6%) | 27 (19.3%) | 13 (9.3%) | 6 (4.3%) | 7 (5.0%) | 19 (13.6%) |
| **perf** | 61 (43.6%) | 29 (20.7%) | 15 (10.7%) | 4 (2.9%) | 9 (6.4%) | 22 (15.7%) |
| **feat** | 49 (35.0%) | 33 (23.6%) | 20 (14.3%) | 12 (8.6%) | 4 (2.9%) | 22 (15.7%) |

## Recommendations for LLM Input (Token-Based)

### 🌟 **Excellent for LLM Input** (95th percentile < 2,000 tokens)
- **ci**: median 262 tokens, 95th percentile 1,402 tokens, 97.9% ≤2K tokens

### ✅ **Good for LLM Input** (95th percentile 2K-4K tokens)
- **fix**: median 432 tokens, 95th percentile 3,371 tokens, 91.4% ≤3K tokens

### ⚠️ **Moderate for LLM Input** (95th percentile 4K-8K tokens)
- **docs**: median 326 tokens, 95th percentile 7,772 tokens, 11.4% >5K tokens
- **test**: median 680 tokens, 95th percentile 7,553 tokens, 6.4% >5K tokens

### ❌ **Challenging for LLM Input** (95th percentile ≥ 8K tokens)
- **chore**: median 248 tokens, 95th percentile 8,224 tokens, 6.4% >5K tokens
- **build**: median 478 tokens, 95th percentile 82,314 tokens, 20.0% >5K tokens
- **style**: median 550 tokens, 95th percentile 37,504 tokens, 17.1% >5K tokens
- **refactor**: median 1,128 tokens, 95th percentile 8,700 tokens, 13.6% >5K tokens
- **perf**: median 1,400 tokens, 95th percentile 12,090 tokens, 15.7% >5K tokens
- **feat**: median 1,602 tokens, 95th percentile 8,160 tokens, 15.7% >5K tokens

## Detailed Statistics by Type

### chore
- **Total Count**: 140 samples
- **Mean**: 5,402 tokens
- **Median**: 248 tokens
- **Standard Deviation**: 30,805
- **Range**: 64 - 298,892 tokens
- **Quartiles**: Q25=140, Q75=638
- **Percentiles**: 90th=2,702, 95th=8,224, 99th=141,970

**Token Range Distribution:**
  - ≤1K: 114 samples (81.4%)
  - 1K-2K: 8 samples (5.7%)
  - 2K-3K: 5 samples (3.6%)
  - 3K-4K: 1 samples (0.7%)
  - 4K-5K: 3 samples (2.1%)
  - >5K: 9 samples (6.4%)

### ci
- **Total Count**: 140 samples
- **Mean**: 484 tokens
- **Median**: 262 tokens
- **Standard Deviation**: 803
- **Range**: 77 - 8,150 tokens
- **Quartiles**: Q25=155, Q75=556
- **Percentiles**: 90th=895, 95th=1,402, 99th=3,144

**Token Range Distribution:**
  - ≤1K: 130 samples (92.9%)
  - 1K-2K: 7 samples (5.0%)
  - 2K-3K: 1 samples (0.7%)
  - 3K-4K: 1 samples (0.7%)
  - 4K-5K: 0 samples (0.0%)
  - >5K: 1 samples (0.7%)

### docs
- **Total Count**: 140 samples
- **Mean**: 2,691 tokens
- **Median**: 326 tokens
- **Standard Deviation**: 11,416
- **Range**: 60 - 127,463 tokens
- **Quartiles**: Q25=190, Q75=963
- **Percentiles**: 90th=5,506, 95th=7,772, 99th=29,368

**Token Range Distribution:**
  - ≤1K: 106 samples (75.7%)
  - 1K-2K: 8 samples (5.7%)
  - 2K-3K: 4 samples (2.9%)
  - 3K-4K: 3 samples (2.1%)
  - 4K-5K: 3 samples (2.1%)
  - >5K: 16 samples (11.4%)

### fix
- **Total Count**: 140 samples
- **Mean**: 26,318 tokens
- **Median**: 432 tokens
- **Standard Deviation**: 296,328
- **Range**: 102 - 3,519,774 tokens
- **Quartiles**: Q25=207, Q75=901
- **Percentiles**: 90th=1,848, 95th=3,371, 99th=24,323

**Token Range Distribution:**
  - ≤1K: 109 samples (77.9%)
  - 1K-2K: 17 samples (12.1%)
  - 2K-3K: 2 samples (1.4%)
  - 3K-4K: 6 samples (4.3%)
  - 4K-5K: 0 samples (0.0%)
  - >5K: 6 samples (4.3%)

### build
- **Total Count**: 140 samples
- **Mean**: 25,693 tokens
- **Median**: 478 tokens
- **Standard Deviation**: 109,310
- **Range**: 91 - 999,428 tokens
- **Quartiles**: Q25=202, Q75=2,179
- **Percentiles**: 90th=34,130, 95th=82,314, 99th=495,455

**Token Range Distribution:**
  - ≤1K: 96 samples (68.6%)
  - 1K-2K: 6 samples (4.3%)
  - 2K-3K: 6 samples (4.3%)
  - 3K-4K: 3 samples (2.1%)
  - 4K-5K: 1 samples (0.7%)
  - >5K: 28 samples (20.0%)

### style
- **Total Count**: 140 samples
- **Mean**: 9,257 tokens
- **Median**: 550 tokens
- **Standard Deviation**: 37,613
- **Range**: 80 - 385,136 tokens
- **Quartiles**: Q25=257, Q75=2,280
- **Percentiles**: 90th=18,776, 95th=37,504, 99th=141,972

**Token Range Distribution:**
  - ≤1K: 82 samples (58.6%)
  - 1K-2K: 21 samples (15.0%)
  - 2K-3K: 6 samples (4.3%)
  - 3K-4K: 5 samples (3.6%)
  - 4K-5K: 2 samples (1.4%)
  - >5K: 24 samples (17.1%)

### test
- **Total Count**: 140 samples
- **Mean**: 5,979 tokens
- **Median**: 680 tokens
- **Standard Deviation**: 45,976
- **Range**: 122 - 532,187 tokens
- **Quartiles**: Q25=312, Q75=1,570
- **Percentiles**: 90th=2,560, 95th=7,553, 99th=83,910

**Token Range Distribution:**
  - ≤1K: 90 samples (64.3%)
  - 1K-2K: 30 samples (21.4%)
  - 2K-3K: 10 samples (7.1%)
  - 3K-4K: 1 samples (0.7%)
  - 4K-5K: 0 samples (0.0%)
  - >5K: 9 samples (6.4%)

### refactor
- **Total Count**: 140 samples
- **Mean**: 2,712 tokens
- **Median**: 1,128 tokens
- **Standard Deviation**: 5,173
- **Range**: 106 - 40,530 tokens
- **Quartiles**: Q25=378, Q75=2,475
- **Percentiles**: 90th=6,187, 95th=8,700, 99th=24,419

**Token Range Distribution:**
  - ≤1K: 68 samples (48.6%)
  - 1K-2K: 27 samples (19.3%)
  - 2K-3K: 13 samples (9.3%)
  - 3K-4K: 6 samples (4.3%)
  - 4K-5K: 7 samples (5.0%)
  - >5K: 19 samples (13.6%)

### perf
- **Total Count**: 140 samples
- **Mean**: 3,278 tokens
- **Median**: 1,400 tokens
- **Standard Deviation**: 6,127
- **Range**: 128 - 43,253 tokens
- **Quartiles**: Q25=413, Q75=2,906
- **Percentiles**: 90th=8,271, 95th=12,090, 99th=34,441

**Token Range Distribution:**
  - ≤1K: 61 samples (43.6%)
  - 1K-2K: 29 samples (20.7%)
  - 2K-3K: 15 samples (10.7%)
  - 3K-4K: 4 samples (2.9%)
  - 4K-5K: 9 samples (6.4%)
  - >5K: 22 samples (15.7%)

### feat
- **Total Count**: 140 samples
- **Mean**: 2,939 tokens
- **Median**: 1,602 tokens
- **Standard Deviation**: 4,649
- **Range**: 92 - 35,015 tokens
- **Quartiles**: Q25=647, Q75=3,114
- **Percentiles**: 90th=6,287, 95th=8,160, 99th=25,413

**Token Range Distribution:**
  - ≤1K: 49 samples (35.0%)
  - 1K-2K: 33 samples (23.6%)
  - 2K-3K: 20 samples (14.3%)
  - 3K-4K: 12 samples (8.6%)
  - 4K-5K: 4 samples (2.9%)
  - >5K: 22 samples (15.7%)

