#!/usr/bin/env python3
"""Tangled Dataset Generator for Concern Separation (Variant A).

This script generates a tangled dataset from the conventional commit classification
dataset by combining multiple concerns (commit types) into single samples.
Outputs both tangled samples and ground truth for evaluation.
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import logging
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TangledDatasetGenerator:
    """Generator for tangled datasets with multiple concerns.
    
    This class handles the creation of tangled datasets where multiple commit
    concerns are combined into single samples, with ground truth separation
    maintained for evaluation purposes.
    
    Attributes:
        samples_count: Number of tangled samples to generate
        concerns_per_sample: Number of distinct concerns per sample
        output_path: Directory to save generated datasets
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use
        type_samples: Dictionary organizing samples by commit type
        types_available: List of available commit types
    """

    def __init__(
        self,
        samples_count: int = 500,  # Max from 2000 available samples
        concerns_per_sample: int = 3,
        output_path: str = "./datasets/tangled/variant_a",
        dataset_name: str = "0x404/ccs_dataset",
        split: str = "train",
    ) -> None:
        """Initialize the tangled dataset generator.
        
        Args:
            samples_count: Number of tangled samples to generate
            concerns_per_sample: Number of distinct concerns per sample (2-5 recommended)
            output_path: Output directory for generated files
            dataset_name: HuggingFace dataset name
            split: Dataset split to use ('train', 'test', 'validation')
        """
        self.samples_count = samples_count
        self.concerns_per_sample = concerns_per_sample
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.split = split
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Data structures
        self.type_samples: Dict[str, List[str]] = defaultdict(list)
        self.types_available: List[str] = []
        
        logger.info(f"Initialized TangledDatasetGenerator")
        logger.info(f"  Samples: {self.samples_count}")
        logger.info(f"  Concerns per sample: {self.concerns_per_sample}")
        logger.info(f"  Output directory: {self.output_path}")

    def load_data(self) -> None:
        """Load dataset from HuggingFace and organize by commit type.
        
        Raises:
            RuntimeError: If dataset loading or processing fails.
        """
        logger.info(f"Loading dataset: {self.dataset_name} (split: {self.split})")
        
        try:
            dataset_raw: Dataset = load_dataset(self.dataset_name, split=self.split)
            logger.info(f"Loaded {len(dataset_raw)} samples")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}") from e

        # Organize data by commit type
        samples_valid = 0
        for row in tqdm(dataset_raw, desc="Processing samples"):
            commit_type = row.get("type")
            git_diff = row.get("git_diff")
            
            if commit_type and git_diff and len(git_diff.strip()) > 0:
                self.type_samples[commit_type].append(git_diff)
                samples_valid += 1

        if not self.type_samples:
            raise RuntimeError("No valid data found in dataset")
            
        self.types_available = list(self.type_samples.keys())
        logger.info(f"Found {samples_valid} valid samples across {len(self.types_available)} types")
        logger.info(f"Available types: {self.types_available}")
        
        # Log type distribution and calculate max possible samples
        for commit_type, diffs in self.type_samples.items():
            logger.info(f"  {commit_type}: {len(diffs)} samples")
        
        # Calculate maximum possible samples based on minimum type count
        min_type_count = min(len(diffs) for diffs in self.type_samples.values())
        max_possible = min_type_count * len(self.types_available) // self.concerns_per_sample
        logger.info(f"Maximum possible tangled samples: {max_possible}")
        
        if self.samples_count > max_possible:
            logger.warning(f"Requested {self.samples_count} exceeds maximum {max_possible}")
            logger.warning("Consider reducing samples_count or concerns_per_sample")

    def generate_samples(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate the complete tangled dataset.
        
        Returns:
            Tuple containing:
                - tangled_data: DataFrame with tangled samples
                - truth_data: DataFrame with individual concerns
                
        Raises:
            RuntimeError: If dataset generation fails.
        """
        if not self.type_samples:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        if len(self.types_available) < self.concerns_per_sample:
            raise RuntimeError(
                f"Not enough commit types ({len(self.types_available)}) "
                f"for {self.concerns_per_sample} concerns"
            )

        logger.info(f"Generating {self.samples_count} tangled samples")

        # Data containers
        tangled_records = []
        truth_records = []

        try:
            for sample_id in tqdm(range(self.samples_count), desc="Generating samples"):
                try:
                    # Select distinct commit types
                    types_selected = random.sample(self.types_available, k=self.concerns_per_sample)
                    
                    # Generate concerns and collect diffs
                    concerns_current = []
                    diffs_collected = []
                    
                    for concern_index, commit_type in enumerate(types_selected):
                        if not self.type_samples[commit_type]:
                            logger.warning(f"No samples available for type: {commit_type}")
                            continue
                            
                        diff_selected = random.choice(self.type_samples[commit_type])
                        
                        # Add to ground truth
                        concerns_current.append({
                            "sample_id": sample_id,
                            "concern_index": concern_index,
                            "concern_type": commit_type,
                            "diff": diff_selected
                        })
                        diffs_collected.append(diff_selected)
                    
                    if len(concerns_current) < self.concerns_per_sample:
                        logger.warning(f"Insufficient concerns for sample {sample_id}, skipping")
                        continue
                    
                    # Shuffle diffs to create tangled version
                    random.shuffle(diffs_collected)
                    diff_tangled = "\n".join(diffs_collected)
                    
                    # Add to tangled data
                    tangled_records.append({
                        "sample_id": sample_id,
                        "concern_count": len(diffs_collected),
                        "tangled_diff": diff_tangled
                    })
                    
                    # Add all concerns to ground truth
                    truth_records.extend(concerns_current)
                    
                except Exception as e:
                    logger.error(f"Error generating sample {sample_id}: {e}")
                    continue

        except Exception as e:
            raise RuntimeError(f"Failed to generate dataset: {e}") from e

        # Create DataFrames
        tangled_data = pd.DataFrame(tangled_records)
        truth_data = pd.DataFrame(truth_records)

        logger.info(f"Generated {len(tangled_data)} tangled samples")
        logger.info(f"Generated {len(truth_data)} ground truth entries")

        return tangled_data, truth_data

    def save_data(self, tangled_data: pd.DataFrame, truth_data: pd.DataFrame) -> None:
        """Save datasets to files.
        
        Args:
            tangled_data: DataFrame with tangled samples
            truth_data: DataFrame with ground truth data
            
        Raises:
            IOError: If file saving fails.
        """
        tangled_file = self.output_path / "tangled.csv"
        truth_file = self.output_path / "ground_truth.csv"

        try:
            # Save files
            tangled_data.to_csv(tangled_file, index=False, encoding="utf-8")
            truth_data.to_csv(truth_file, index=False, encoding="utf-8")
            
            logger.info(f"Saved tangled dataset: {tangled_file}")
            logger.info(f"Saved ground truth dataset: {truth_file}")
            
            # Log file sizes
            size_tangled = tangled_file.stat().st_size / 1024 / 1024  # MB
            size_truth = truth_file.stat().st_size / 1024 / 1024  # MB
            
            logger.info(f"File sizes: tangled={size_tangled:.2f}MB, ground_truth={size_truth:.2f}MB")
            
        except Exception as e:
            raise IOError(f"Failed to save files: {e}") from e

    def validate_output(self) -> bool:
        """Validate the generated files.
        
        Returns:
            True if validation passes, False otherwise.
        """
        tangled_file = self.output_path / "tangled.csv"
        truth_file = self.output_path / "ground_truth.csv"

        if not tangled_file.exists() or not truth_file.exists():
            logger.error("Output files do not exist")
            return False

        try:
            # Load and validate DataFrames
            tangled_data = pd.read_csv(tangled_file)
            truth_data = pd.read_csv(truth_file)
            
            # Basic validation
            entries_expected = len(tangled_data) * self.concerns_per_sample
            entries_actual = len(truth_data)
            
            if entries_actual != entries_expected:
                logger.warning(
                    f"Ground truth entries mismatch: expected {entries_expected}, "
                    f"got {entries_actual}"
                )
            
            # Check required columns
            columns_tangled_required = {"sample_id", "concern_count", "tangled_diff"}
            columns_truth_required = {"sample_id", "concern_index", "concern_type", "diff"}
            
            if not columns_tangled_required.issubset(tangled_data.columns):
                logger.error(f"Missing tangled columns: {columns_tangled_required - set(tangled_data.columns)}")
                return False
                
            if not columns_truth_required.issubset(truth_data.columns):
                logger.error(f"Missing ground truth columns: {columns_truth_required - set(truth_data.columns)}")
                return False

            logger.info(f"✅ Validation passed!")
            logger.info(f"  Tangled samples: {len(tangled_data)}")
            logger.info(f"  Ground truth entries: {len(truth_data)}")
            logger.info(f"  Average concerns per sample: {len(truth_data) / len(tangled_data):.1f}")
            
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def show_sample(self) -> None:
        """Display sample data for inspection."""
        tangled_file = self.output_path / "tangled.csv"
        truth_file = self.output_path / "ground_truth.csv"
        
        if not tangled_file.exists() or not truth_file.exists():
            logger.error("Dataset files do not exist")
            return
            
        try:
            tangled_data = pd.read_csv(tangled_file)
            truth_data = pd.read_csv(truth_file)
            
            logger.info("\n" + "="*60)
            logger.info("SAMPLE DATA PREVIEW")
            logger.info("="*60)
            
            # Show tangled sample
            sample_id = 0
            sample_tangled = tangled_data[tangled_data['sample_id'] == sample_id].iloc[0]
            
            logger.info(f"\n📝 Tangled Sample {sample_id}:")
            logger.info(f"  Sample ID: {sample_tangled['sample_id']}")
            logger.info(f"  Concern Count: {sample_tangled['concern_count']}")
            logger.info(f"  Diff Preview: {sample_tangled['tangled_diff'][:200]}...")
            
            # Show ground truth
            sample_truth = truth_data[truth_data['sample_id'] == sample_id]
            
            logger.info(f"\n🎯 Ground Truth for Sample {sample_id}:")
            for _, row in sample_truth.iterrows():
                logger.info(f"  Concern {row['concern_index']}: {row['concern_type']}")
                logger.info(f"    Diff Preview: {row['diff'][:100]}...")
                
        except Exception as e:
            logger.error(f"Failed to show sample data: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Variant A tangled dataset")
    parser.add_argument(
        "--samples-count",
        type=int,
        default=500,
        help="Number of tangled samples to generate"
    )
    parser.add_argument(
        "--concerns-per-sample",
        type=int,
        default=3,
        help="Number of concerns per sample"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./datasets/tangled/variant_a",
        help="Output directory path"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="0x404/ccs_dataset",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split", 
        type=str,
        default="train",
        help="Dataset split to use"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the tangled dataset generation."""
    args = parse_args()
    
    # Configuration from command line
    config = {
        "samples_count": args.samples_count,
        "concerns_per_sample": args.concerns_per_sample,
        "output_path": args.output_path,
        "dataset_name": args.dataset_name,
        "split": args.split
    }

    try:
        # Initialize generator
        generator = TangledDatasetGenerator(**config)
        
        # Load and prepare data
        generator.load_data()
        
        # Generate datasets
        tangled_data, truth_data = generator.generate_samples()
        
        # Save to files
        generator.save_data(tangled_data, truth_data)
        
        # Validate output
        if generator.validate_output():
            logger.info("✅ Dataset generation successful!")
            
            # Show sample data
            generator.show_sample()
        else:
            logger.error("❌ Dataset validation failed!")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main() 