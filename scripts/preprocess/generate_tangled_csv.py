#!/usr/bin/env python3
"""CSV-based Tangled Code Changes Generator.

This script generates tangled code changes from preprocessed CSV files by
combining multiple concerns (commit types) within each repository into
single samples. The script reads CSV files with repository_name, type, and
git_diff columns and outputs tangled samples.
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CsvTangledGenerator:
    """Generator for tangled code changes from CSV files.
    
    This class handles the creation of tangled code changes where multiple
    commit concerns from the same repository are combined into single samples.
    Each repository is processed independently to maintain consistency.
    
    Attributes:
        input_csv_path: Path to input CSV file
        concerns_per_sample: Number of distinct concerns per tangled sample
        allow_duplicate_concern_types: Whether to allow same type multiple times
        total_samples: Total number of tangled samples to generate
        output_dir: Directory to save generated files
        repo_type_samples: Dictionary organizing samples by repository and type
        repositories_available: List of available repositories
    """

    def __init__(
        self,
        input_csv_path: str,
        concerns_per_sample: int = 2,
        allow_duplicate_concern_types: bool = False,
        total_samples: int = 10,
        output_dir: str = None,
    ) -> None:
        """Initialize the CSV tangled generator.
        
        Args:
            input_csv_path: Path to input CSV file
            concerns_per_sample: Number of concerns per tangled sample (2-10 recommended)
            allow_duplicate_concern_types: Allow selecting same type multiple times
            total_samples: Total number of tangled samples to generate
            output_dir: Output directory (defaults to input file directory)
        """
        self.input_csv_path = Path(input_csv_path)
        self.concerns_per_sample = concerns_per_sample
        self.allow_duplicate_concern_types = allow_duplicate_concern_types
        self.total_samples = total_samples
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.input_csv_path.parent
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data structures: repo_name -> {type -> [diffs]}
        self.repo_type_samples: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.repositories_available: List[str] = []
        
        logger.info(f"Initialized CsvTangledGenerator")
        logger.info(f"  Input file: {self.input_csv_path}")
        logger.info(f"  Concerns per sample: {self.concerns_per_sample}")
        logger.info(f"  Allow duplicate types: {self.allow_duplicate_concern_types}")
        logger.info(f"  Total samples: {self.total_samples}")
        logger.info(f"  Output directory: {self.output_dir}")

    def load_csv_data(self) -> None:
        """Load CSV data and organize by repository and type.
        
        Raises:
            FileNotFoundError: If input CSV file doesn't exist
            ValueError: If CSV has missing required columns or invalid data
        """
        if not self.input_csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {self.input_csv_path}")
        
        logger.info(f"Loading CSV data from: {self.input_csv_path}")
        
        try:
            df = pd.read_csv(self.input_csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}") from e
        
        # Validate required columns
        required_columns = {"repository_name", "type", "git_diff"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Process and organize data
        samples_valid = 0
        samples_skipped = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV rows"):
            repo_name = row.get("repository_name")
            commit_type = row.get("type")
            git_diff = row.get("git_diff")
            
            # Validate row data
            if pd.isna(repo_name) or pd.isna(commit_type) or pd.isna(git_diff):
                samples_skipped += 1
                continue
                
            git_diff_str = str(git_diff).strip()
            if not git_diff_str or len(git_diff_str) == 0:
                samples_skipped += 1
                continue
            
            # Store valid data
            self.repo_type_samples[str(repo_name)][str(commit_type)].append(git_diff_str)
            samples_valid += 1
        
        if not self.repo_type_samples:
            raise ValueError("No valid data found in CSV file")
        
        self.repositories_available = list(self.repo_type_samples.keys())
        
        logger.info(f"Found {samples_valid} valid samples, skipped {samples_skipped}")
        logger.info(f"Found {len(self.repositories_available)} repositories")

    def generate_tangled_samples(self) -> pd.DataFrame:
        """Generate tangled samples for all repositories.
        
        Returns:
            DataFrame with columns: repository_name, types, type_count, diff
            
        Raises:
            RuntimeError: If data generation fails
        """
        if not self.repo_type_samples:
            raise RuntimeError("CSV data not loaded. Call load_csv_data() first.")
        
        logger.info(f"Generating tangled samples for {len(self.repositories_available)} repositories")
        
        tangled_records = []
        total_samples_generated = 0
        
        try:
            for repo_name in tqdm(self.repositories_available, desc="Processing repositories"):
                repo_data = self.repo_type_samples[repo_name]
                types_available = list(repo_data.keys())
                
                # Check if repository has enough types for non-duplicate mode
                if not self.allow_duplicate_concern_types and len(types_available) < self.concerns_per_sample:
                    logger.warning(
                        f"Repository '{repo_name}' has only {len(types_available)} types "
                        f"but needs {self.concerns_per_sample} (duplicate types disabled). Skipping."
                    )
                    continue
                
                # Generate samples for this repository
                repo_samples_generated = 0
                for sample_id in range(self.total_samples):
                    try:
                        # Select concern types
                        if self.allow_duplicate_concern_types:
                            types_selected = random.choices(types_available, k=self.concerns_per_sample)
                        else:
                            if len(types_available) >= self.concerns_per_sample:
                                types_selected = random.sample(types_available, k=self.concerns_per_sample)
                            else:
                                continue  # Skip if not enough unique types
                        
                        # Collect diffs for selected types
                        diffs_collected = []
                        for concern_type in types_selected:
                            available_diffs = repo_data[concern_type]
                            if not available_diffs:
                                logger.warning(f"No diffs available for {repo_name}:{concern_type}")
                                continue
                            
                            selected_diff = random.choice(available_diffs)
                            diffs_collected.append(selected_diff)
                        
                        if len(diffs_collected) != self.concerns_per_sample:
                            continue  # Skip if couldn't collect enough diffs
                        
                        # Combine diffs into tangled diff
                        tangled_diff = "\n".join(diffs_collected)
                        
                        # Create record
                        tangled_records.append({
                            "repository_name": repo_name,
                            "types": ",".join(types_selected),
                            "type_count": len(types_selected),
                            "diff": tangled_diff
                        })
                        
                        repo_samples_generated += 1
                        total_samples_generated += 1
                        
                    except Exception as e:
                        logger.error(f"Error generating sample for {repo_name}: {e}")
                        continue
                
                logger.info(f"Generated {repo_samples_generated} samples for {repo_name}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate tangled samples: {e}") from e
        
        # Create DataFrame
        tangled_data = pd.DataFrame(tangled_records)
        logger.info(f"Total generated samples: {total_samples_generated}")
        
        return tangled_data

    def save_results(self, tangled_data: pd.DataFrame) -> Path:
        """Save tangled data to CSV file.
        
        Args:
            tangled_data: DataFrame with tangled samples
            
        Returns:
            Path to saved file
            
        Raises:
            IOError: If file saving fails
        """
        # Generate output filename: tangled_{original_name}.csv
        input_stem = self.input_csv_path.stem  # filename without extension
        output_filename = f"tangled_{input_stem}.csv"
        output_path = self.output_dir / output_filename
        
        try:
            tangled_data.to_csv(output_path, index=False, encoding="utf-8")
            
            logger.info(f"Saved tangled dataset: {output_path}")
            
            # Log file size
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"File size: {file_size_mb:.2f}MB")
            
            return output_path
            
        except Exception as e:
            raise IOError(f"Failed to save results: {e}") from e




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate tangled code changes from CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV file with repository_name, type, git_diff columns"
    )
    parser.add_argument(
        "--concerns-per-sample",
        type=int,
        default=2,
        help="Number of concerns to combine per tangled sample"
    )
    parser.add_argument(
        "--allow-duplicate-types",
        action="store_true",
        help="Allow selecting the same concern type multiple times"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=10,
        help="Total number of tangled samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to input file directory)"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the CSV tangled generator."""
    args = parse_args()
    
    # Configuration from command line
    config = {
        "input_csv_path": args.input_csv,
        "concerns_per_sample": args.concerns_per_sample,
        "allow_duplicate_concern_types": args.allow_duplicate_types,
        "total_samples": args.total_samples,
        "output_dir": args.output_dir
    }
    
    try:
        # Initialize generator
        generator = CsvTangledGenerator(**config)
        
        # Load and process CSV data
        generator.load_csv_data()
        
        # Generate tangled samples
        tangled_data = generator.generate_tangled_samples()
        
        if tangled_data.empty:
            logger.error("❌ No tangled samples were generated!")
            return
        
        # Save results
        output_path = generator.save_results(tangled_data)
        logger.info("✅ Tangled dataset generation successful!")
    
    except Exception as e:
        logger.error(f"Tangled dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main() 