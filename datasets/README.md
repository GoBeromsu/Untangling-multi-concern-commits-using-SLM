---
license: mit
task_categories:
  - text-generation
  - text-classification
language:
  - en
tags:
  - code
  - git
  - commits
  - software-engineering
  - concern-separation
size_categories:
  - 1K<n<10K
---

# Untangling Multi-Concern Commits with Small Language Models

This dataset contains commit data for training and evaluating models on software engineering tasks, specifically focusing on identifying and separating concerns in multi-concern commits.

## Dataset Description

This dataset consists of two main configurations:

### 1. Sampled Dataset (`sampled`)

- **File**: `data/sampled_css_dataset.csv`
- **Description**: Individual atomic commits with single concerns
- **Features**:
  - `annotated_type`: The type of concern/change in the commit
  - `masked_commit_message`: Commit message with sensitive information masked
  - `git_diff`: The actual code changes in diff format
  - `sha`: Git commit SHA hash

### 2. Tangled Dataset (`tangled`)

- **File**: `data/tangled_css_dataset.csv`
- **Description**: Multi-concern commits that combine multiple atomic commits
- **Features**:
  - `description`: Combined description of all concerns
  - `diff`: Combined diff of all changes
  - `concern_count`: Number of individual concerns combined
  - `shas`: JSON string containing array of original commit SHAs
  - `types`: JSON string containing array of concern types

## Dataset Statistics

- **Sampled Dataset**: ~1.3MB, individual atomic commits
- **Tangled Dataset**: ~7.1MB, artificially combined multi-concern commits

## Use Cases

1. **Commit Message Generation**: Generate appropriate commit messages for code changes
2. **Concern Classification**: Classify the type of concern addressed in a commit
3. **Commit Decomposition**: Break down multi-concern commits into individual concerns
4. **Code Change Analysis**: Understand the relationship between code changes and their descriptions

## Data Collection and Processing

The dataset was created by:

1. Collecting atomic commits from software repositories
2. Sampling and filtering commits based on quality criteria
3. Artificially combining atomic commits to create tangled multi-concern examples
4. Masking sensitive information while preserving semantic content

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{css_commits_dataset,
  title={Untangling Multi-Concern Commits with Small Language Models},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/Untangling-Multi-Concern-Commits-with-Small-Language-Models}
}
```

## Scripts and Tools

This dataset includes several Python scripts for data processing and analysis:

- `sample_css_dataset.py`: Script for sampling and filtering commits
- `generate_tangled.py`: Script for creating tangled multi-concern commits
- `clean_ccs_dataset.py`: Data cleaning and preprocessing utilities
- `show_sampled_diffs.py`: Visualization of sampled commit diffs
- `show_tokens_distribution.py`: Analysis of token distribution in the dataset

## License

This dataset is released under the MIT License. See the LICENSE file for details.

## Dataset Loading

You can load this dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the sampled dataset
sampled_data = load_dataset("Untangling-Multi-Concern-Commits-with-Small-Language-Models", "sampled")

# Load the tangled dataset
tangled_data = load_dataset("Untangling-Multi-Concern-Commits-with-Small-Language-Models", "tangled")
```
