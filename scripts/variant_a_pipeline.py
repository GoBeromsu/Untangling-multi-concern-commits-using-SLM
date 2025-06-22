#!/usr/bin/env python3
"""Complete Variant A pipeline using OpenAI Batch API for efficient processing."""

import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Config
SAMPLES = 100
CONCERNS_PER_SAMPLE = 3
DATA_DIR = Path("./datasets/tangled/variant_a")
DATASET_NAME = "0x404/ccs_dataset"
MODEL = "gpt-4.1-mini-2025-04-14"

# Classification types
COMMIT_TYPES = ["feat", "fix", "perf", "style", "refactor", "docs", "test", "ci", "build", "chore"]
SYSTEM_PROMPT = """Extract commit concerns and classify each with conventional commit types.

Types:
- feat: Code changes that introduce new functionality, including internal or user-facing features. This includes additions that enhance the capabilities of the software system.
- fix: Code changes that resolve faults or bugs. These modifications address errors that affect correct behaviour.
- perf: Code changes that optimise performance, such as improvements in execution speed or memory efficiency.
- style: Code changes that improve readability or adhere to formatting standards, without affecting the logic or meaning. Includes naming, indentation, or linting adjustments.
- refactor: Code changes that restructure code to improve maintainability, modularity, or scalability without changing its external behaviour. This excludes “perf” and “style” changes. Examples: code cleanup, exception handling improvements, deprecated code removal.
- docs: Code changes that affect documentation. Includes comment updates, typo corrections, and documentation file changes.
- test: Code changes that modify test files, including test additions or updates.
- ci: Code changes to Continuous Integration configuration or workflow scripts (e.g., .travis.yml, .github/workflows).
- build: Code changes to the build system or dependencies. Includes build configuration files, dependency upgrades, or build scripts.
- chore: Code changes that do not fit into other categories. Includes auxiliary or maintenance tasks.

Return JSON only.
"""

RESPONSE_SCHEMA = {
    "type": "object", 
    "properties": {
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": COMMIT_TYPES},
                    "diff": {"type": "string"}
                },
                "required": ["type", "diff"],
                "additionalProperties": False
            }
        }
    },
    "required": ["concerns"],
    "additionalProperties": False
}


class VariantAPipeline:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        
        # OpenAI API configuration
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate API key
        self._validate_api_key()
    
    def _validate_api_key(self):
        """Validate OpenAI API key by making a test request."""
        try:
            logger.info("Validating API key...")
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            # Check if we have proper access
            models_data = response.json()
            if "data" in models_data and len(models_data["data"]) > 0:
                logger.info("✓ API key validated successfully")
                logger.info(f"✓ Access to {len(models_data['data'])} models confirmed")
            else:
                logger.warning("⚠ API key valid but no models accessible")
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ API key validation failed with HTTP error: {e}")
            if e.response.status_code == 401:
                raise ValueError("Invalid OpenAI API key - authentication failed")
            elif e.response.status_code == 403:
                raise ValueError("OpenAI API key lacks necessary permissions")
            else:
                raise ValueError(f"API key validation failed: HTTP {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ API key validation failed with network error: {e}")
            raise ValueError(f"Network error during API key validation: {e}")
        
    def generate_dataset(self):
        """Generate tangled dataset from CCS data."""
        logger.info("Loading CCS dataset...")
        dataset = load_dataset(DATASET_NAME, split="train")
        
        # Group by type
        type_samples = defaultdict(list)
        for row in tqdm(dataset, desc="Processing samples"):
            if row.get("type") and row.get("git_diff"):
                type_samples[row["type"]].append(row["git_diff"])
        
        types_available = list(type_samples.keys())
        logger.info(f"Found {len(types_available)} types: {types_available}")
        
        # Generate tangled samples
        tangled_records = []
        truth_records = []
        
        for sample_id in tqdm(range(SAMPLES), desc="Generating samples"):
            selected_types = random.sample(types_available, k=CONCERNS_PER_SAMPLE)
            
            concerns = []
            diffs = []
            
            for idx, commit_type in enumerate(selected_types):
                if type_samples[commit_type]:
                    diff = random.choice(type_samples[commit_type])
                    concerns.append({
                        "sample_id": sample_id,
                        "concern_index": idx,
                        "concern_type": commit_type,
                        "diff": diff
                    })
                    diffs.append(diff)
            
            if len(concerns) == CONCERNS_PER_SAMPLE:
                random.shuffle(diffs)
                tangled_records.append({
                    "sample_id": sample_id,
                    "concern_count": len(diffs),
                    "tangled_diff": "\n".join(diffs)
                })
                truth_records.extend(concerns)
        
        # Save data
        tangled_df = pd.DataFrame(tangled_records)
        truth_df = pd.DataFrame(truth_records)
        
        tangled_df.to_csv(self.data_dir / "tangled.csv", index=False)
        truth_df.to_csv(self.data_dir / "ground_truth.csv", index=False)
        
        logger.info(f"Generated {len(tangled_df)} tangled samples")
        return tangled_df
    
    def create_batch_requests(self, tangled_df):
        """Create batch request file."""
        batch_file = self.data_dir / "batch_requests.jsonl"
        
        tasks = []
        for _, row in tangled_df.iterrows():
            task = {
                "custom_id": f"sample-{row.sample_id}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "temperature": 1.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Tangled code changes diff:\n{row.tangled_diff}"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "untangle_response",
                            "schema": RESPONSE_SCHEMA,
                            "strict": True
                        }
                    }
                }
            }
            tasks.append(task)
        
        with open(batch_file, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
        
        logger.info(f"Created batch file with {len(tasks)} requests")
        return batch_file
    
    def submit_batch(self, batch_file):
        """Submit batch job to OpenAI using REST API."""
        # Upload file
        file_upload_headers = {
            "Authorization": f"Bearer {self.api_key}"
            # Note: Content-Type is automatically set by requests for multipart/form-data
        }
        
        with open(batch_file, "rb") as f:
            files = {
                "file": f,
                "purpose": (None, "batch")
            }
            
            response = requests.post(
                f"{self.base_url}/files",
                headers=file_upload_headers,
                files=files,
                timeout=60
            )
            response.raise_for_status()
            file_obj = response.json()
        
        logger.info(f"File uploaded: {file_obj['id']}")
        
        # Create batch job
        batch_data = {
            "input_file_id": file_obj["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        
        response = requests.post(
            f"{self.base_url}/batches",
            headers=self.headers,
            json=batch_data,
            timeout=30
        )
        response.raise_for_status()
        batch_job = response.json()
        
        logger.info(f"Submitted batch job: {batch_job['id']}")
        return batch_job["id"]
    
    def wait_for_batch(self, batch_id):
        """Wait for batch completion using REST API."""
        logger.info("Waiting for batch completion...")
        
        while True:
            response = requests.get(
                f"{self.base_url}/batches/{batch_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            batch = response.json()
            
            status = batch["status"]
            
            if status == "completed":
                logger.info("Batch completed successfully")
                return batch["output_file_id"]
            elif status == "failed":
                logger.error("Batch failed")
                return None
            
            logger.info(f"Status: {status}")
            time.sleep(30)
    
    def download_results(self, output_file_id):
        """Download and preprocess batch results using REST API - focus on structured output only."""
        response = requests.get(
            f"{self.base_url}/files/{output_file_id}/content",
            headers=self.headers,
            timeout=60
        )
        response.raise_for_status()
        content = response.content
        
        results_file = self.data_dir / "batch_results.jsonl"
        with open(results_file, 'wb') as f:
            f.write(content)
        
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                sample_id = int(result["custom_id"].split("-")[1])
                
                try:
                    response = result["response"]["body"]["choices"][0]["message"]
                    structured_output = json.loads(response["content"])
                    concerns = structured_output.get("concerns", [])
                    
                    processed_concerns = []
                    for concern in concerns:
                        processed_concern = {
                            "type": concern.get("type", ""),
                            "diff": concern.get("diff", "")
                        }
                        processed_concerns.append(processed_concern)
                    
                    processed_result = {
                        "sample_id": sample_id,
                        "predicted_count": len(processed_concerns),
                        "predicted_types": [c["type"] for c in processed_concerns],
                        "concerns": processed_concerns
                    }
                    results.append(processed_result)
                    
                except Exception as e:
                    logger.error(f"Error parsing structured output for sample {sample_id}: {e}")
        
        pred_file = self.data_dir / "predictions.jsonl"
        with open(pred_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, separators=(',', ':')) + '\n')
        
        logger.info(f"Preprocessed and saved {len(results)} structured outputs")
        return results
    
    def evaluate(self, predictions):
        """Evaluate preprocessed structured outputs against ground truth with comprehensive metrics."""
        gt_df = pd.read_csv(self.data_dir / "ground_truth.csv")
        
        pred_dict = {}
        for pred in predictions:
            pred_dict[pred["sample_id"]] = pred["predicted_types"]
        
        results = []
        tp_total = fp_total = fn_total = 0
        
        for sample_id, group in gt_df.groupby("sample_id"):
            if sample_id not in pred_dict:
                continue
                
            gt_types = set(group["concern_type"].tolist())
            pred_types = set(pred_dict[sample_id])
            
            gt_count = len(gt_types)
            pred_count = len(pred_types)
            
            exact_match = gt_types == pred_types
            count_match = gt_count == pred_count
            
            tp = len(gt_types & pred_types)
            fp = len(pred_types - gt_types)
            fn = len(gt_types - pred_types)
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            precision_s = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_s = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_s = (2 * precision_s * recall_s / (precision_s + recall_s) 
                    if (precision_s + recall_s) > 0 else 0.0)
            
            union = gt_types | pred_types
            jaccard_s = len(gt_types & pred_types) / len(union) if union else 0.0
            
            results.append({
                "sample_id": sample_id,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "gt_types": sorted(list(gt_types)),
                "pred_types": sorted(list(pred_types)),
                "exact_match": exact_match,
                "count_match": count_match,
                "precision_s": precision_s,
                "recall_s": recall_s,
                "f1_s": f1_s,
                "jaccard_s": jaccard_s,
                "tp": tp,
                "fp": fp,
                "fn": fn
            })
        
        results_df = pd.DataFrame(results)
        
        total = len(results_df)
        if total == 0:
            logger.warning("No samples to evaluate")
            return {}
        
        exact_match_rate = results_df["exact_match"].sum() / total
        count_match_rate = results_df["count_match"].sum() / total
        
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall) 
                    if (micro_precision + micro_recall) > 0 else 0.0)
        
        macro_precision = results_df["precision_s"].mean()
        macro_recall = results_df["recall_s"].mean()
        macro_f1 = results_df["f1_s"].mean()
        
        mean_jaccard = results_df["jaccard_s"].mean()
        mae_count = (results_df["pred_count"] - results_df["gt_count"]).abs().mean()
        
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Evaluated samples: {total}")
        logger.info("")
        logger.info("EXACT MATCHING:")
        logger.info(f"  • Exact match rate: {exact_match_rate:.3f} - perfect type set match")
        logger.info(f"  • Count match rate: {count_match_rate:.3f} - correct number of types only")
        logger.info("")
        logger.info("MICRO METRICS - aggregated across all types:")
        logger.info(f"  • Precision: {micro_precision:.3f} - ratio of correct predictions")
        logger.info(f"  • Recall: {micro_recall:.3f} - ratio of ground truth types found")
        logger.info(f"  • F1-Score: {micro_f1:.3f} - harmonic mean of precision and recall")
        logger.info("")
        logger.info("MACRO METRICS - averaged across samples:")
        logger.info(f"  • Precision: {macro_precision:.3f}")
        logger.info(f"  • Recall: {macro_recall:.3f}")
        logger.info(f"  • F1-Score: {macro_f1:.3f}")
        logger.info("")
        logger.info("ADDITIONAL METRICS:")
        logger.info(f"  • Mean Jaccard: {mean_jaccard:.3f} - set similarity measure")
        logger.info(f"  • MAE Count: {mae_count:.3f} - mean absolute error in count prediction")
        logger.info("=" * 60)
        
        results_df.to_csv(self.data_dir / "evaluation.csv", index=False)
        
        summary = {
            "evaluated_samples": total,
            "exact_match_rate": exact_match_rate,
            "count_match_rate": count_match_rate,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "mean_jaccard": mean_jaccard,
            "mae_count": mae_count,
            "total_tp": int(tp_total),
            "total_fp": int(fp_total),
            "total_fn": int(fn_total)
        }
        
        with open(self.data_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run(self):
        logger.info("Starting pipeline")
        
        # Check if data exists
        tangled_file = self.data_dir / "tangled.csv"
        if tangled_file.exists():
            logger.info("Loading existing dataset")
            tangled_df = pd.read_csv(tangled_file)
        else:
            tangled_df = self.generate_dataset()
        
        # Create batch requests
        batch_file = self.create_batch_requests(tangled_df)
        
        # Submit batch
        batch_id = self.submit_batch(batch_file)
        
        # Wait for completion
        output_file_id = self.wait_for_batch(batch_id)
        if not output_file_id:
            logger.error("Batch processing failed")
            return
        
        # Download results
        predictions = self.download_results(output_file_id)
        
        # Evaluate
        summary = self.evaluate(predictions)
        
        logger.info("Completed")
        return summary


if __name__ == "__main__":
    pipeline = VariantAPipeline()
    pipeline.run() 