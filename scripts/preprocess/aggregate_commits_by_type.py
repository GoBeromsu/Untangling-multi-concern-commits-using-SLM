#!/usr/bin/env python3
"""Dataset preprocessing script.

This script preprocesses commit data by:
1. Extracting repository names from commit URLs
2. Grouping commits by repository and type
3. Aggregating git diffs within each repository-type combination
4. Outputting a single, cleaned CSV for downstream pipeline usage.

Refactored for simplicity and core functionality.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ============================================================================
# Configuration - Project Root Based Paths
# ============================================================================

# Project root directory (assumes script is in scripts/ subdirectory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input CSV file path - conventional commit classification dataset
INPUT_CSV_PATH = PROJECT_ROOT / "datasets/candidates/conventional-commit-classification/Dataset/annotated_dataset.csv"

# Output directory for preprocessed data
OUTPUT_DIR = PROJECT_ROOT / "datasets/tangled"

# Processing configuration
AGGREGATE_CONCERNS = True
CHUNK_SIZE = 10000  # Process data in chunks for memory efficiency

# Required input columns
REQUIRED_INPUT_COLUMNS = {"type", "commit_url", "git_diff"}

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging() -> logging.Logger:
    """Set up simple, stream-only logging.
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger


# ============================================================================
# Data Processing Functions
# ============================================================================

def extract_repo_names(commit_urls: pd.Series) -> pd.Series:
    """Extract repository names from GitHub commit URLs using vectorized operations.
    
    Args:
        commit_urls: Series of GitHub commit URLs
        
    Returns:
        Series of repository names in format 'owner/repo'.
        Rows that cannot be parsed will contain NaN.
    """
    logger.info(f"Extracting repository names from {len(commit_urls)} URLs...")
    
    # GitHub commit URL pattern: https://github.com/owner/repo/commit/sha
    pattern = r"https://github\.com/([^/]+)/([^/]+)/commit/.+"
    
    # Extract using pandas string operations (vectorized)
    extracted = commit_urls.str.extract(pattern, expand=True)
    
    # Combine owner and repo with '/'
    repo_names = extracted[0] + '/' + extracted[1]
    
    # This will return NaN for rows where extraction failed
    return repo_names


def prepare_data_for_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Extract repository names and prepare data for aggregation.
    
    Args:
        df: Raw commit data DataFrame
        
    Returns:
        DataFrame with type, repository_name, git_diff columns
    """
    logger.info("Preparing data for aggregation...")
    
    # Create working copy
    df_work = df[list(REQUIRED_INPUT_COLUMNS)].copy()
    
    # Extract repository names
    df_work['repository_name'] = extract_repo_names(df_work['commit_url'])
    
    # Handle failed extractions
    original_count = len(df_work)
    df_work.dropna(subset=['repository_name'], inplace=True)
    failed_count = original_count - len(df_work)
    if failed_count > 0:
        logger.warning(f"Dropped {failed_count} rows due to URL parsing failure.")

    # Select final columns
    df_processed = df_work[['type', 'repository_name', 'git_diff']].copy()
    
    logger.info(f"✓ Data prepared with {len(df_processed)} records.")
    return df_processed


def aggregate_concerns_by_repo_and_type(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate git diffs by repository and commit type.
    
    Args:
        df: DataFrame with type, repository_name, git_diff columns
        
    Returns:
        DataFrame with aggregated git diffs
    """
    logger.info("Aggregating concerns by repository and type...")
    
    # Efficient aggregation using groupby
    df_aggregated = (
        df.groupby(['repository_name', 'type'], as_index=False)
        .agg({'git_diff': lambda x: '\n'.join(x.astype(str))})
    )
    
    logger.info(f"✓ Aggregation completed. {len(df_aggregated)} records created.")
    return df_aggregated


# ============================================================================
# I/O Functions
# ============================================================================

def load_data_from_csv(csv_path: Path, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """Load CSV data with memory-efficient chunked reading for large files.
    
    Args:
        csv_path: Path to CSV file
        chunk_size: Size of chunks for reading (None for full load)
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading CSV data from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        # Memory-efficient chunked reading if file is large
        use_chunks = chunk_size and csv_path.stat().st_size > 50 * 1024 * 1024
        
        if use_chunks:
            logger.info(f"Using chunked reading with chunk_size={chunk_size}")
            chunks = [chunk for chunk in pd.read_csv(csv_path, chunksize=chunk_size)]
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_path)
        
        logger.info(f"✓ CSV loaded successfully with {len(df):,} rows.")
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}") from e


def save_processed_data(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save preprocessed data to a CSV file.
    
    Args:
        df: Preprocessed DataFrame to save
        output_dir: Directory to save files
        
    Returns:
        Path to the saved data file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "preprocessed.csv"
    
    logger.info(f"Saving {len(df)} records to: {data_file}")
    df.to_csv(data_file, index=False, encoding="utf-8")
    
    logger.info(f"✓ Data saved successfully.")
    return data_file


def log_summary_statistics(df_prepared: pd.DataFrame, df_aggregated: pd.DataFrame) -> None:
    """Log simple summary statistics to the console.
    
    Args:
        df_prepared: DataFrame before aggregation, used for original commit counts.
        df_aggregated: The final, aggregated DataFrame.
    """
    repo_count = df_aggregated['repository_name'].nunique()
    type_distribution = df_aggregated['type'].value_counts().to_dict()
    commit_counts_per_repo = df_prepared['repository_name'].value_counts()
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total unique repositories: {repo_count}")
    
    logger.info("\nConcern type distribution (processed records):")
    for concern_type, count in type_distribution.items():
        logger.info(f"  - {concern_type}: {count} records")

    logger.info("\nCommit counts per repository (original commits):")
    for repo_name, count in commit_counts_per_repo.items():
        logger.info(f"  - {repo_name}: {count} commits")

    logger.info("=" * 60)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline() -> None:
    """Execute the complete preprocessing pipeline."""
    pipeline_start = time.time()
    
    logger.info("=" * 60)
    logger.info("STARTING DATASET PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load data
        df_original = load_data_from_csv(INPUT_CSV_PATH, chunk_size=CHUNK_SIZE)
        
        # Step 2: Prepare data for aggregation
        df_prepared = prepare_data_for_aggregation(df_original)
        
        # Step 3: Aggregate concerns
        if AGGREGATE_CONCERNS:
            df_final = aggregate_concerns_by_repo_and_type(df_prepared)
        else:
            df_final = df_prepared
            logger.info("Skipping aggregation as per configuration.")
        
        # Step 4: Save results
        save_processed_data(df_final, OUTPUT_DIR)
        
        # Step 5: Log summary
        log_summary_statistics(df_prepared, df_final)
        
        total_time = time.time() - pipeline_start
        logger.info(f"Total processing time: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        exit(1)


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Main function to run the preprocessing pipeline."""
    global logger
    logger = setup_logging()
    
    run_pipeline()
    logger.info(f"🎉 Preprocessing completed successfully!")


if __name__ == "__main__":
    main() 