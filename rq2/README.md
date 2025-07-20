# RQ1: Context Ablation Study

This experiment studies the impact of commit message context on model performance.

## Description

Evaluates how removing commit messages affects the ability to untangle commits, using only the diff information.

## Execution

```bash
cd rq1
python main.py
```

## Output

Results will be saved to `../results/rq1/`:

- `predictions.csv`: Model predictions without commit message context
- `metrics.json`: Performance metrics compared to RQ0 baseline
