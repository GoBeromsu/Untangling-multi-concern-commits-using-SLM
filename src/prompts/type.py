FEWSHOT_PROMPT_TEMPLATE = """\
A git commit can typically be classified into specific categories by examining its code changes. These categories include:

- "style": Changes that solely improve the code's formatting and appearance without affecting functionality (e.g., adjusting whitespace, fixing indentation, cleaning up code formatting).
- "docs": Updates or improvements to documentation, which may include inline code comments, README files, or any other type of documentation associated with the project.
- "test": Modifications exclusively related to test code, like the addition of new tests or the correction and improvement of existing tests.
- "build": Changes that affect the build system or tools (like Gulp, Broccoli, NPM) or alterations to external dependencies (e.g., library or package updates).
- "cicd": Tweaks to configuration files or scripts used in Continuous Integration/Continuous Deployment (CI/CD) systems, such as Travis CI or CircleCI configurations.
- "fix": Code amendments that focus on rectifying errors, fixing bugs, or patching security vulnerabilities.
- "feat": Commits that introduce new features or capabilities to the project, such as new classes, functions, or methods.
- "refactor": Changes that reorganize and clean up the codebase without modifying its external behavior or outputs, improving readability and maintainability.

For a given git commit, we can inspect its code difference (diff)to determine its type.

Diff: ```diff
diff --git a/util/src/com/intellij/util/containers/SLRUMap.java b/util/src/com/intellij/util/containers/SLRUMap.java
index 7f3d09c..635dfab 100644
--- a/util/src/com/intellij/util/containers/SLRUMap.java
+++ b/util/src/com/intellij/util/containers/SLRUMap.java
@@ -69,12 +69,12 @@ public class SLRUMap<K,V> {{
   public void put(K key, V value) {{
     V oldValue = myProtectedQueue.remove(key);
     if (oldValue != null) {{
-      onDropFromCache(key, value);
+      onDropFromCache(key, oldValue);
     }}
 
     oldValue = myProbationalQueue.put(getStableKey(key), value);
     if (oldValue != null) {{
-      onDropFromCache(key, value);
+      onDropFromCache(key, oldValue);
     }}
   }}
```
Message: Corrected parameter error in onDropFromCache() function call
Types: fix
Reason: The git commit is a "fix" commit as it rectified a parameter error where `oldValue` should be passed as the argument of `onDropFromCache` rather than `value`.

Diff: ```diff
{diff}
```
Types: """


def get_default_prompt_template() -> str:
    return FEWSHOT_PROMPT_TEMPLATE


def get_type_prompt(diff: str, custom_template: str = None) -> str:
    template = custom_template or FEWSHOT_PROMPT_TEMPLATE
    return template.format(diff=diff)
