#!/usr/bin/env python3
"""Upload dataset to HuggingFace Hub with overwrite functionality."""

import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from datasets import load_dataset
from huggingface_hub import HfApi, login, create_repo, delete_repo

# Load environment variables from .env file
load_dotenv()

DATASET_REPO_ID = os.getenv(
    "DATASET_REPO_ID",
    "Berom0227/Untangling-Multi-Concern-Commits-with-Small-Language-Models",
)
DATASETS_PATH = Path(__file__).parent.parent
DATA_PATH = DATASETS_PATH / "data"
SCRIPTS_PATH = DATASETS_PATH / "scripts"

REQUIRED_FILES = [
    DATA_PATH / "tangled_css_dataset_train.csv",
    DATA_PATH / "tangled_css_dataset_test.csv",
]

UPLOAD_FILES = [
    ("README.md", DATASETS_PATH / "README.md"),
    ("dataset_info.yaml", DATASETS_PATH / "dataset_info.yaml"),
    ("data/tangled_css_dataset_train.csv", DATA_PATH / "tangled_css_dataset_train.csv"),
    ("data/tangled_css_dataset_test.csv", DATA_PATH / "tangled_css_dataset_test.csv"),
    ("data/excluded_commits.csv", DATA_PATH / "excluded_commits.csv"),
    ("scripts/clean_ccs_dataset.py", SCRIPTS_PATH / "clean_ccs_dataset.py"),
    (
        "scripts/generate_tangled_commites.py",
        SCRIPTS_PATH / "generate_tangled_commites.py",
    ),
    ("scripts/sample_atomic_commites.py", SCRIPTS_PATH / "sample_atomic_commites.py"),
    ("scripts/show_sampled_commites.py", SCRIPTS_PATH / "show_sampled_commites.py"),
    (
        "scripts/show_tokens_distribution.py",
        SCRIPTS_PATH / "show_tokens_distribution.py",
    ),
    ("scripts/upload_to_huggingface.py", SCRIPTS_PATH / "upload_to_huggingface.py"),
    (".env.example", DATASETS_PATH / ".env.example"),
]


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from command line args or environment."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.getenv("HUGGINGFACE_HUB_TOKEN")


def authenticate_huggingface(token: Optional[str] = None) -> None:
    """Authenticate with HuggingFace Hub."""
    if not token:
        token = get_hf_token()

    if not token:
        print("✗ No HuggingFace token provided")
        print("Usage: python upload_to_huggingface.py <token>")
        print("Or set HUGGINGFACE_HUB_TOKEN in .env file")
        sys.exit(1)

    try:
        login(token=token)
        print("✓ Successfully authenticated with HuggingFace Hub")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        sys.exit(1)


def create_or_overwrite_repo(repo_id: str, overwrite: bool = True) -> None:
    """Create or overwrite HuggingFace repository."""
    api = HfApi()

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        if overwrite:
            print(f"Repository {repo_id} exists. Deleting for overwrite...")
            delete_repo(repo_id=repo_id, repo_type="dataset")
            print("✓ Repository deleted successfully")
        else:
            print(
                f"Repository {repo_id} already exists. Use overwrite=True to replace it."
            )
            return
    except Exception as e:
        print(f"Repository {repo_id} doesn't exist or error checking: {e}")

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        print(f"✓ Created new repository: {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise


def upload_dataset_files(repo_id: str) -> None:
    """Upload dataset files to HuggingFace Hub."""
    api = HfApi()

    print("Uploading files to HuggingFace Hub...")
    for repo_path, local_path in UPLOAD_FILES:
        if local_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"✓ Uploaded {repo_path}")
            except Exception as e:
                print(f"✗ Failed to upload {repo_path}: {e}")
        else:
            print(f"⚠ File not found: {local_path}")


def verify_upload(repo_id: str) -> None:
    """Verify dataset upload by loading both configurations."""
    print("\nVerifying dataset upload...")

    try:
        train_dataset = load_dataset(repo_id, "train", split="train")
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        print(f"  Columns: {train_dataset.column_names}")

        test_dataset = load_dataset(repo_id, "test", split="train")
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
        print(f"  Columns: {test_dataset.column_names}")

        print("\n✓ Dataset upload verification successful!")

    except Exception as e:
        print(f"✗ Dataset verification failed: {e}")
        print("Dataset may still be processing. Try again in a few minutes.")


def check_required_files() -> None:
    """Check if required dataset files exist."""
    for file_path in REQUIRED_FILES:
        if not file_path.exists():
            print(f"✗ Required file not found: {file_path}")
            sys.exit(1)
    print("✓ All required files found")


def main() -> None:
    """Main execution function."""
    print("🚀 Starting HuggingFace dataset upload...")
    print(f"Repository: {DATASET_REPO_ID}")
    print(f"Dataset path: {DATASETS_PATH}")

    check_required_files()

    try:
        authenticate_huggingface()
        create_or_overwrite_repo(DATASET_REPO_ID, overwrite=True)
        upload_dataset_files(DATASET_REPO_ID)
        verify_upload(DATASET_REPO_ID)

        print(
            f"\n🎉 Dataset successfully uploaded to: https://huggingface.co/datasets/{DATASET_REPO_ID}"
        )

    except Exception as e:
        print(f"✗ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
