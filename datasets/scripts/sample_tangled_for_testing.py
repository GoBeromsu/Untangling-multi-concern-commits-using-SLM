#!/usr/bin/env python3
"""Randomly sample 10 items from tangled dataset for testing purposes."""


import pandas as pd
import numpy as np
from pathlib import Path


def load_tangled_dataset(csv_file: Path) -> pd.DataFrame:
    """Load the tangled dataset from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} commits from tangled dataset")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")


def sample_random_commits(
    df: pd.DataFrame, sample_size: int, random_seed: int = 42
) -> pd.DataFrame:
    """Sample random commits from the dataset."""
    if len(df) < sample_size:
        raise ValueError(
            f"Dataset has only {len(df)} commits, cannot sample {sample_size}"
        )

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Random sampling
    sampled_df = df.sample(n=sample_size, random_state=random_seed)
    return sampled_df.reset_index(drop=True)


def save_sampled_dataset(df: pd.DataFrame, output_file: Path) -> None:
    """Save sampled dataset to CSV file."""
    try:
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} sampled commits to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error saving CSV file: {e}")


def display_sample_summary(df: pd.DataFrame) -> None:
    """Display summary of sampled dataset."""
    print("\n" + "=" * 60)
    print("SAMPLE SUMMARY")
    print("=" * 60)
    print(f"Total sampled commits: {len(df)}")

    if "concern_count" in df.columns:
        print(
            f"Concern count range: {df['concern_count'].min()} - {df['concern_count'].max()}"
        )
        print(f"Average concern count: {df['concern_count'].mean():.2f}")

    if "types" in df.columns:
        print(f"Sample includes types column: Yes")

    if "shas" in df.columns:
        print(f"Sample includes SHAs column: Yes")

    print("=" * 60)


def main():
    """Main function to sample 10 random commits from tangled dataset."""
    print("Tangled Dataset Random Sampler - Extracting 10 commits for testing...")

    # Configuration
    csv_file = Path("../data/tangled_ccs_dataset.csv")
    output_file = Path("../data/tangled_test_sample.csv")
    sample_size = 10

    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        return

    # Load dataset
    df = load_tangled_dataset(csv_file)

    # Sample random commits
    sampled_df = sample_random_commits(df, sample_size)

    # Save sampled dataset
    save_sampled_dataset(sampled_df, output_file)

    # Display summary
    display_sample_summary(sampled_df)

    print(f"\nRandom sample saved to: {output_file}")
    print("Process completed! 🎉")


if __name__ == "__main__":
    main()
