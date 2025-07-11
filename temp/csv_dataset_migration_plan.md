# CSV Dataset Migration Plan

## 1. Overview of Current State

- Batch evaluation expects **JSON** test datasets searched under `datasets/**/*.json`.
- `load_concern_test_dataset()` reads a JSON object and returns `(cases, metadata)`.
- Each test case is expected to contain:
  - `tangleChange`: full diff string
  - `atomicChanges`: list of objects with a `label` key (concern type)
- UI and evaluation logic rely on these JSON structures.

## 2. Overview of Final State

- Batch evaluation can consume **CSV** datasets located at `datasets/ccs/css_tangled_dataset.csv`.
- CSV columns: `description`, `diff`, `concern_count`, `shas`, `types`.
- Only `diff`, `concern_count`, and `types` are required for evaluation; `description` and `shas` are ignored for now.
- Evaluation pipeline (UI → loader → metrics) works unchanged from the user’s point of view.
- Back-compatibility with the original JSON format is preserved (for now).

## 3. Files to Change

1. **`src/app.py`**
   - Add `load_concern_test_dataset_csv()` that reads the CSV file into a list of test-case dicts.
   - Update `render_batch_evaluation_interface()` to detect/select `.csv` files alongside `.json`.
2. **`src/ui/patterns.py`**
   - Extend `extract_test_case_data()` to handle the CSV row structure (fallback to existing logic when keys differ).
3. **`README.md`** (optional)
   - Brief usage note for CSV dataset support.

## 4. Task Checklist

- [x] Detect CSV files in dataset picker (app.py)
  - [x] Include `datasets/**/*.csv` in the search pattern
- [x] Implement CSV dataset loader (app.py)
  - [x] Parse `types` column as JSON list
  - [x] Build case dicts compatible with `extract_test_case_data`
- [x] Update `extract_test_case_data` (patterns.py)
  - [x] Support both JSON and CSV case schemas (existing function works!)
- [x] Remove/guard metadata handling when absent
- [x] Smoke-test batch evaluation with `css_tangled_dataset.csv`
  - [x] Verify CSV loading (630 cases loaded successfully)
  - [x] Verify data structure compatibility
- [ ] Update README with CSV dataset instructions (optional)
