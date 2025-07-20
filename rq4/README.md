# RQ3: Latency vs. Context Length

This experiment analyzes the relationship between context length and model inference latency.

## Description

Evaluates how different context lengths (256, 512, 1024, 2048, 4096 tokens) affect the inference latency of commit untangling.

## Execution

```bash
cd rq3
python main.py
```

## Output

Results will be saved to `../results/rq3/`:

- `predictions.csv`: Model predictions with latency measurements for each context length
- `metrics.json`: Latency statistics for each context length
- `latency_vs_length.png`: Visualization of latency vs context length relationship
