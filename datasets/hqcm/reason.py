#!/usr/bin/env python3

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from openai import AsyncOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-4.1-2025-04-14"
MAX_CONCURRENT_REQUESTS = 10  # Control concurrency to avoid rate limits

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}},
    "required": ["reasoning"],
    "additionalProperties": False,
}

STRUCTURED_OUTPUT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "reasoning_response",
        "schema": RESPONSE_SCHEMA,
        "strict": True,
    },
}


def load_tangled_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_csv(file_path)


def get_system_prompt() -> str:
    """Get system prompt for reasoning generation"""
    return """
You are a software engineer splitting a commit into atomic changes, labelling each one using Conventional Commit types.
# Instructions
- Always classify each atomic change using a single <type> from the Conventional Commit taxonomy.
- Prioritise **why** the change was made (purpose) before **what** was changed (object).
- Think step by step with Conventional Commit taxonomy before deciding the final <type>.
    1. Identify the intent of the change (Purpose)
    2. Determine the scope of change (Object)
    3. Analyse behavioural impact
    4. Map to Conventional Commit type using purpose-object priority

# Conventional Commit Taxonomy
## Purpose (Why the Change Was made)
- **feat**: Code changes that adds new features to the codebase, encompassing both internal and user-oriented features.
- **fix**:  Code change that patches a bug in the codebase
- **refactor**: Code changes that aim to restructure the program without changing its behavior, aiming to improve maintainability. 
## Object (What Kind of Entity Was Changed)
- **docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.
- **test**: Code changes that modify test files or test directories (e.g., files named with `*Test.java`, `test_*.py`, `__tests__/`).
- **cicd**: Code changes that modify CI/CD pipelines, scripts, or config files (e.g., `.github/workflows`, `.gitlab-ci.yml`, Jenkinsfile).
- **build**: Code changes that modify build tooling, dependencies, or build configuration files (e.g., `build.gradle`, `pom.xml`, `Makefile`, `Dockerfile`, or scripts affecting the build process).
# Example
<commit_diff id="example-1">
diff --git a/services/java/com/android/server/updates/SELinuxPolicyInstallReceiver.java b/services/java/com/android/server/updates/SELinuxPolicyInstallReceiver.java
index e8337f6..0ab86e4 100644
--- a/services/java/com/android/server/updates/SELinuxPolicyInstallReceiver.java
+++ b/services/java/com/android/server/updates/SELinuxPolicyInstallReceiver.java
@@ -122,9 +122,16 @@ public class SELinuxPolicyInstallReceiver extends ConfigUpdateInstallReceiver {
     }
 
     private void setEnforcingMode(Context context) {
-        boolean mode = Settings.Global.getInt(context.getContentResolver(),
-            Settings.Global.SELINUX_STATUS, 0) == 1;
-        SELinux.setSELinuxEnforce(mode);
+        String mode = Settings.Global.getString(context.getContentResolver(),
+            Settings.Global.SELINUX_STATUS);
+        if (mode.equals("1")) {
+            Slog.i(TAG, "Setting enforcing mode");
+            SystemProperties.set("persist.selinux.enforcing", mode);
+        } else if (mode.equals("0")) {
+            Slog.i(TAG, "Tried to set permissive mode, ignoring");
+        } else {
+            Slog.e(TAG, "Got invalid enforcing mode: " + mode);
+        }
     }
 
     @Override

diff --git a/common/buildcraft/api/recipes/AssemblyRecipe.java b/common/buildcraft/api/recipes/AssemblyRecipe.java
index a384f7125..573db2827 100644
--- a/common/buildcraft/api/recipes/AssemblyRecipe.java
+++ b/common/buildcraft/api/recipes/AssemblyRecipe.java
@@ -1,8 +1,6 @@
 package buildcraft.api.recipes;
 
 import java.util.LinkedList;
-
-import buildcraft.core.inventory.StackHelper;
 import net.minecraft.item.ItemStack;
 
 public class AssemblyRecipe {
</commit_diff>

<reasoning id="example-1">
Step 1: What does the change aim to achieve? → Enhances SELinux mode handling with better validation and logging, and cleans up unused import.  
Step 2: What kind of file/entity changed? → Security policy receiver class and recipe API class.  
Step 3: Purpose analysis: SELinux changes add new logging capabilities and improved error handling, while import removal is code cleanup.  
Step 4: `feat` for enhanced SELinux functionality with new logging, `refactor` for removing unused import.
</reasoning>
<commit_label id="example-1">
["feat", "refactor"]
</commit_label>

<commit_diff id="example-2">
diff --git a/src/biz/bokhorst/xprivacy/Util.java b/src/biz/bokhorst/xprivacy/Util.java
@@ -94,6 +94,8 @@ public class Util {
 		else if (ex instanceof ActivityShare.ServerException)
 			priority = Log.WARN;
+		else if (ex instanceof NoClassDefFoundError)
+			priority = Log.WARN;
 		else
 			priority = Log.ERROR;
</commit_diff>

<reasoning id="example-2">
Step 1: The intent is to handle a new error type (`NoClassDefFoundError`) with a specific log level (`WARN`).  
Step 2: The change affects the logging-level assignment logic in `Util.java`.  
Step 3: The behavioural impact is that the system will now handle a previously unhandled error type differently (lower severity).  
Step 4: Since this expands the system's ability to respond gracefully to a new case, it introduces new behaviour → feat.
</reasoning>

<commit_label id="example-2">
["feat"]
</commit_label>
"""


def create_user_prompt(diff: str, correct_types: str) -> str:
    """Create user prompt for reasoning generation"""
    context = f"""
You are given a commit diff and the assigned Conventional Commit types.

Your task is to produce a reasoning that justifies the given type(s), by following the step-by-step process below:
1. Identify the intent of the change (Purpose)
2. Determine the scope of change (Object)
3. Analyse behavioural impact
4. Map to Conventional Commit type using purpose-object priority

Use one combined reasoning block that includes all assigned types. Write clearly and concisely.
<commit_diff>
{diff}
</commit_diff>
<commit_label>
{correct_types}
</commit_label>
<reasoning>
"""
    return f"{context}"


async def process_single_row(
    client: AsyncOpenAI,
    row_data: Dict[str, Any],
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single row and return complete result with input and reasoning"""
    async with semaphore:
        try:
            user_prompt = create_user_prompt(row_data["changes"], row_data["types"])

            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=STRUCTURED_OUTPUT_FORMAT,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")

            response_data = json.loads(content)
            reasoning = response_data.get("reasoning", "No reasoning provided")

            return {
                "changes": row_data["changes"],
                "types": row_data["types"],
                "count": row_data["count"],
                "reasoning": reasoning,
                "success": True,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {
                "changes": row_data["changes"],
                "types": row_data["types"],
                "count": row_data["count"],
                "reasoning": f"JSON decode error: {str(e)}",
                "success": False,
            }
        except Exception as e:
            logger.error(f"API error: {e}")
            return {
                "changes": row_data["changes"],
                "types": row_data["types"],
                "count": row_data["count"],
                "reasoning": f"API error: {str(e)}",
                "success": False,
            }


async def process_dataset_rows(
    rows: List[Dict[str, Any]],
    api_key: str,
    system_prompt: str,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
) -> List[Dict[str, Any]]:
    """Process multiple rows concurrently and return complete results"""
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = [process_single_row(client, row, system_prompt, semaphore) for row in rows]

    logger.info(
        f"Processing {len(tasks)} rows with {max_concurrent_requests} concurrent requests"
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            # Skip failed tasks completely - they won't be in final results
            continue
        else:
            processed_results.append(result)

    await client.close()
    return processed_results


def prepare_row_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare row data for processing"""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "changes": row["changes"],
                "types": row["types"],
                "count": row["count"],
            }
        )
    return rows


def create_results_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create DataFrame directly from results"""
    # Log success/failure statistics
    successful_count = sum(1 for result in results if result["success"])
    failed_count = len(results) - successful_count
    logger.info(
        f"Successfully processed {successful_count} rows, {failed_count} failed"
    )

    # Create DataFrame from results
    df_results = pd.DataFrame(results)

    # Remove success flag column as it's not needed in final output
    if "success" in df_results.columns:
        df_results = df_results.drop("success", axis=1)

    return df_results


async def process_dataset_async(
    input_file: Path,
    output_file: Path,
    api_key: str,
    sample_size: int = None,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
) -> None:
    """Process dataset with async API calls"""

    # Load dataset
    df = load_tangled_dataset(input_file)
    logger.info(f"Loaded dataset with {len(df)} rows")

    # Apply sampling if specified
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info(f"Using sample of {len(df)} rows")

    # Prepare data for processing
    system_prompt = get_system_prompt()
    rows = prepare_row_data(df)

    # Process rows concurrently
    results = await process_dataset_rows(
        rows, api_key, system_prompt, max_concurrent_requests
    )

    # Create results DataFrame and save
    df_results = create_results_dataframe(results)
    df_results.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df_results)} results to {output_file}")


def process_dataset(
    input_file: Path,
    output_file: Path,
    api_key: str,
    sample_size: int = None,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
) -> None:
    """Sync wrapper for async processing"""
    asyncio.run(
        process_dataset_async(
            input_file, output_file, api_key, sample_size, max_concurrent_requests
        )
    )


def main() -> None:
    """Main entry point"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    input_file = Path("datasets/tangled_dataset.csv")
    output_file = Path("datasets/tangled_dataset_with_reasoning.csv")

    sample_size = None
    max_concurrent_requests = 10

    process_dataset(
        input_file, output_file, api_key, sample_size, max_concurrent_requests
    )


if __name__ == "__main__":
    main()
