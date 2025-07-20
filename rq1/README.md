# RQ0: Performance Gap Analysis

This experiment analyzes the performance gap between different models on the commit untangling task.

## Description

Compares the performance of GPT-4-turbo and microsoft/phi-4 models on untangling tangled commits into separate concerns.

## Execution

```bash
cd rq0
python main.py
```

## Output

Results will be saved to `../results/rq0/`:

- `predictions.csv`: Model predictions for each sample
- `metrics.json`: F1, precision, and recall metrics
