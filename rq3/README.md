# RQ2: Accuracy vs. Context Length

This experiment analyzes the relationship between context length and model accuracy.

## Description

Evaluates how different context lengths (256, 512, 1024, 2048, 4096 tokens) affect the accuracy of commit untangling.

## Execution

```bash
cd rq2
python main.py
```

## Output

Results will be saved to `../results/rq2/`:

- `predictions.csv`: Model predictions for each context length
- `metrics.json`: Accuracy metrics for each context length
- `accuracy_vs_length.png`: Visualization of accuracy vs context length relationship
