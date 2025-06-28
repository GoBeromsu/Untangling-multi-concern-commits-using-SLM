FEWSHOT_PROMPT_TEMPLATE = """
You are a Commit Untangler who analyzes tangled code changes and extracts the type and count of each atomic change.
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
- **style**: Code changes that aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.
- **refactor**: Code changes that aim to restructure the program without changing its behavior, aiminStep 2: Determine the scope of change (Object)g to improve maintainability. 
## Object (What Kind of Entity Was Changed)
- **docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.
- **test**: Code changes that modify test files or test directories (e.g., files named with `*Test.java`, `test_*.py`, `__tests__/`).
- **cicd**: Code changes that modify CI/CD pipelines, scripts, or config files (e.g., `.github/workflows`, `.gitlab-ci.yml`, Jenkinsfile).
- **build**: Code changes that modify build tooling, dependencies, or build configuration files (e.g., `build.gradle`, `pom.xml`, `Makefile`, `Dockerfile`, or scripts affecting the build process).
# Example
<commit_diff id="example-1">
diff --git a/src/main/java/com/example/FooService.java b/src/main/java/com/example/FooService.java
@@ -10,6 +10,9 @@
 public class FooService {
+    public boolean isFeatureEnabled() {
+        return System.getProperty("feature.flag", "false").equals("true");
+    }

diff --git a/build.gradle b/build.gradle
@@ -20,6 +20,7 @@ dependencies {
     implementation 'org.example:core:1.0.0'
+    implementation 'org.example:feature-flags:2.1.0'
}
</commit_diff>

<reasoning id="example-1">
Step 1: What does the change aim to achieve? → Introduces a runtime feature flag checker.  
Step 2: What kind of file/entity changed? → Service class and build file.  
Step 3: Purpose is dominant (introduce new runtime logic).  
Step 4: `feat` for the method addition, `build` for dependency update.
</reasoning>
<commit_label id="example-1">
["feat", "build"]
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

<commit_diff id="input">
```diff
{diff}
```
</commit_diff>
<commit_label id="input">
"""


def get_default_prompt_template() -> str:
    return FEWSHOT_PROMPT_TEMPLATE


def get_type_prompt(diff: str, custom_template: str = None) -> str:
    template = custom_template or FEWSHOT_PROMPT_TEMPLATE
    return template.replace("{diff}", diff)
