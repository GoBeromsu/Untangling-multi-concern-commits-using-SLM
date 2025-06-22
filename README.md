# Concern is All You Need : Semantic Untangling with Small Language Model

## Dataset

### Purpose

The CCC Tangled Dataset (Variant A) is designed for training and evaluating concern separation models. This dataset artificially combines multiple commit concerns into single tangled samples, simulating real-world scenarios where developers need to untangle mixed concerns in code changes.

### Data Source

The dataset is generated from the [Conventional Commit Classification (CCC) dataset](https://huggingface.co/datasets/0x404/ccs_dataset) available on HuggingFace, which contains 1,400 labeled commit samples across 10 conventional commit types.

### Dataset Structure

#### `tangled.csv`

Input samples for Small Language Models (SLM) training and inference.

| Field           | Type   | Description                                                         |
| --------------- | ------ | ------------------------------------------------------------------- |
| `sample_id`     | int    | Unique identifier for each tangled sample                           |
| `concern_count` | int    | Number of individual concerns combined in this sample (typically 3) |
| `tangled_diff`  | string | Combined git diff containing multiple shuffled concerns             |

#### `ground_truth.csv`

Ground truth data for evaluation and concern separation validation.

| Field           | Type   | Description                                                                                                   |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------------- |
| `sample_id`     | int    | References the corresponding tangled sample                                                                   |
| `concern_index` | int    | Sequential index of the concern within the sample (0, 1, 2...)                                                |
| `concern_type`  | string | Conventional commit type (`feat`, `fix`, `refactor`, `style`, `test`, `docs`, `chore`, `perf`, `ci`, `build`) |
| `diff`          | string | Individual git diff for this specific concern                                                                 |

### Generation

The dataset is generated using `scripts/generate_variant_a.py`, which:

1. Loads the original CCC dataset from HuggingFace
2. Randomly selects distinct commit types for each sample
3. Combines and shuffles git diffs to create tangled versions
4. Maintains ground truth mappings for evaluation
