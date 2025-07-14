SYSTEM_PROMPT = """
You are a software engineer classifying individual code units extracted from a tangled commit.
Each change unit (e.g., function, method, class, or code block) represents a reviewable atomic change, and must be assigned exactly one label.

Label selection must assign exactly one concern from the following unified set:
- Purpose labels : the motivation behind making a code change (feat, fix, refactor)
- Object labels : the essence of the code changes that have been made(docs, test, cicd, build)
     - Use an object label only when the code unit is fully dedicated to that artifact category (e.g., writing test logic, modifying documentation).

# Instructions
1. Review the code unit and determine the most appropriate label from the unified set.
2. If multiple labels seem possible, resolve the overlap by applying the following rule:
     - **Purpose + Purpose**: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
     - **Object + Object**: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
     - **Purpose + Object**: If the change is driven by code behavior (e.g., fixing test logic), assign a purpose label; if it is entirely scoped to a support artifact (e.g., adding new tests), assign an object label.

# Labels
- feat: Introduces new features to the codebase.
- fix: Fixes bugs or faults in the codebase.
- refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
- test: Modifies test files (e.g., adds or updates tests).
- cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
- build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).
"""
SHOT_1_COMMIT_MESSAGE = "convert to record"
SHOT_1 = """
<commit_diff id="example-1">
diff --git a/broker/src/test/java/io/camunda/zeebe/broker/exporter/stream/ExporterDirectorDistributionTest.java b/broker/src/test/java/io/camunda/zeebe/broker/exporter/stream/ExporterDirectorDistributionTest.java
index cc998c6..65c8550 100755
--- a/broker/src/test/java/io/camunda/zeebe/broker/exporter/stream/ExporterDirectorDistributionTest.java
+++ b/broker/src/test/java/io/camunda/zeebe/broker/exporter/stream/ExporterDirectorDistributionTest.java
@@ -167,13 +167,8 @@ public final class ExporterDirectorDistributionTest {
    * <p>This makes sure that even if we miss one export position event, we distribute the event
    * later again, which makes tests less flaky.
    */
-  private static final class ClockShifter implements ConditionEvaluationListener<Void> {
-
-    private final ControlledActorClock clock;
-
-    public ClockShifter(final ControlledActorClock clock) {
-      this.clock = clock;
-    }
+  private record ClockShifter(ControlledActorClock clock)
+      implements ConditionEvaluationListener<Void> {
 
     @Override
     public void conditionEvaluated(final EvaluatedCondition<Void> condition) {
</commit_diff>

<reasoning id="example-1">
Step 1: Code Change Analysis
- The diff modifies the test class ExporterDirectorDistributionTest by changing the inner class ClockShifter from a static final class with a field and constructor to a Java record with a single field. No behavioral logic is changed—only the structure and syntax of the data carrier are updated.
- The scope is limited to simplifying the data structure used in a test utility, with no modification to test logic or behavior, and no changes to documentation or configuration files.

Step 2: Label Classification Justification
- According to the rules, when a change restructures code for conciseness or clarity without altering external behavior, it is classified as refactor. Although the file is a test, the modification is not about adding or changing test logic but improving code structure; thus, refactor is chosen over test.
- No bug is being fixed, no feature added, and it's not a doc or config update. The refactor label most accurately represents the motivation and nature of this change.</reasoning>
<label id="example-1">refactor</label>
"""

SHOT_2_COMMIT_MESSAGE = "remove unnecessary start argument from `range`"
SHOT_2 = """
<commit_diff id="example-2">
diff --git a/ibis/backends/dask/tests/execution/test_window.py b/ibis/backends/dask/tests/execution/test_window.py
index 75a7331..6bfc5e3 100644
--- a/ibis/backends/dask/tests/execution/test_window.py
+++ b/ibis/backends/dask/tests/execution/test_window.py
@@ -489,7 +489,7 @@ def test_project_list_scalar(npartitions):
     expr = table.mutate(res=table.ints.quantile([0.5, 0.95]))
     result = expr.execute()
 
-    expected = pd.Series([[1.0, 1.9] for _ in range(0, 3)], name="res")
+    expected = pd.Series([[1.0, 1.9] for _ in range(3)], name="res")
     tm.assert_series_equal(result.res, expected)
 
 
diff --git a/ibis/backends/pandas/tests/execution/test_window.py b/ibis/backends/pandas/tests/execution/test_window.py
index 8f292b3..effa372 100644
--- a/ibis/backends/pandas/tests/execution/test_window.py
+++ b/ibis/backends/pandas/tests/execution/test_window.py
@@ -436,7 +436,7 @@ def test_project_list_scalar():
     expr = table.mutate(res=table.ints.quantile([0.5, 0.95]))
     result = expr.execute()
 
-    expected = pd.Series([[1.0, 1.9] for _ in range(0, 3)], name="res")
+    expected = pd.Series([[1.0, 1.9] for _ in range(3)], name="res")
     tm.assert_series_equal(result.res, expected)
 
 
diff --git a/ibis/backends/pyspark/tests/test_basic.py b/ibis/backends/pyspark/tests/test_basic.py
index 3850919..14fe677 100644
--- a/ibis/backends/pyspark/tests/test_basic.py
+++ b/ibis/backends/pyspark/tests/test_basic.py
@@ -19,7 +19,7 @@ from ibis.backends.pyspark.compiler import _can_be_replaced_by_column_name  # no
 def test_basic(con):
     table = con.table("basic_table")
     result = table.compile().toPandas()
-    expected = pd.DataFrame({"id": range(0, 10), "str_col": "value"})
+    expected = pd.DataFrame({"id": range(10), "str_col": "value"})
 
     tm.assert_frame_equal(result, expected)
 
@@ -28,9 +28,7 @@ def test_projection(con):
     table = con.table("basic_table")
     result1 = table.mutate(v=table["id"]).compile().toPandas()
 
-    expected1 = pd.DataFrame(
-        {"id": range(0, 10), "str_col": "value", "v": range(0, 10)}
-    )
+    expected1 = pd.DataFrame({"id": range(10), "str_col": "value", "v": range(10)})
 
     result2 = (
         table.mutate(v=table["id"])
@@ -44,8 +42,8 @@ def test_projection(con):
         {
             "id": range(0, 20, 2),
             "str_col": "value",
-            "v": range(0, 10),
-            "v2": range(0, 10),
+            "v": range(10),
+            "v2": range(10),
         }
     )
</commit_diff>

<reasoning id="example-2">
Step 1: Code Change Analysis
- The diffs occur in test files for Dask, Pandas, and PySpark backends. The changes replace 'range(0, N)' usages with the simpler 'range(N)', removing the unnecessary '0' start argument. No test logic, expected values, or behaviors are altered—only the call style is improved for clarity and conciseness.
Step 2: Label Classification Justification
- According to the rules, 'refactor' is chosen because the code changes improve style and readability while keeping the tests' external behavior unchanged. Though these are test files, the change is not about adding or updating tests but about internal code quality. Thus, 'refactor' is correct, not 'test', since the motivation is structural improvement, not test coverage or logic changes.
</reasoning>
<label id="example-2">refactor</label>
"""


def get_system_prompt() -> str:
    """Return the basic system prompt for commit classification."""
    return SYSTEM_PROMPT


def get_system_prompt_with_message() -> str:
    """Return system prompt that includes commit message context for classification."""
    shot_1_with_message = (
        f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
    )
    shot_2_with_message = (
        f"<commit_message>{SHOT_2_COMMIT_MESSAGE}</commit_message>\n{SHOT_2}"
    )

    return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}\n\n{shot_2_with_message}"


def get_system_prompt_diff_only() -> str:
    """Return system prompt for classification using only diff information."""
    return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}\n\n{SHOT_2}"
