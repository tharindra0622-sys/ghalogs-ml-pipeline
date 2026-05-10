"""
verify_output.py
================
Checks that data/runs_200_2.csv (parsed from runs.json.gz) exactly matches
the expected values from the original sample dataset — row by row, column by column.

Run locally:
    python verify_output.py

Run in GitHub Actions:
    python verify_output.py  (added as a step after fetch_data.py)

Exit code 0 = all checks passed  |  Exit code 1 = mismatch found
"""

import sys
import math
import pandas as pd

# ── Expected values from original sample CSV (first 5 rows) ───────────────────
# These are the ground-truth values extracted directly from runs_200_2.csv.
# Every field must match exactly after fetch_data.py parses runs.json.gz.

EXPECTED = [
    # ── Row 0 ──────────────────────────────────────────────────────────────
    {
        "_id":                     "pytables/pytables_.github/workflows/wheels.yml_200_1",
        "repository_name":         "pytables/pytables",
        "workflow_path":           ".github/workflows/wheels.yml",
        "run_number":              200,
        "run_attempt":             1,
        "total_logs_size":         5547709.0,
        "log_num_jobs":            18,
        "log_total_steps":         90,
        "log_total_duration_sec":  26155.301228,
        "log_shell_steps":         43,
        "log_action_steps":        47,
        "log_error_steps":         0,
        "log_total_lines":         44625,
        "log_has_linux":           1,
        "log_has_macos":           1,
        "log_has_windows":         1,
        "log_num_os_types":        3,
        "log_early3_total_dur":    8.926717,
        "log_early3_max_dur":      5.290709,
        "log_early3_min_dur":      0.21581,
        "log_early3_shell_count":  1,
        "log_early3_action_count": 2,
        "log_early3_error_count":  0,
        "log_early3_avg_dur":      2.975572333333333,
        "log_error_rate":          0.0,
        "log_shell_ratio":         0.4777777777777778,
        "log_avg_step_dur":        290.6144580888889,
        "log_max_step_dur":        10977.500073,
        "metadata_conclusion":     "success",
        "metadata_event":          "push",
        "metadata_status":         "completed",
        "metadata_actor_login":    "ivilata",
        "metadata_created_at":     "2023-09-21T12:55:26Z",
        "metadata_repository_full_name": "PyTables/PyTables",
    },
    # ── Row 1 ──────────────────────────────────────────────────────────────
    {
        "_id":                     "pytables/pytables_.github/workflows/wheels.yml_201_1",
        "repository_name":         "pytables/pytables",
        "workflow_path":           ".github/workflows/wheels.yml",
        "run_number":              201,
        "run_attempt":             1,
        "total_logs_size":         5554835.0,
        "log_num_jobs":            18,
        "log_total_steps":         90,
        "log_total_duration_sec":  29746.845546,
        "log_shell_steps":         43,
        "log_action_steps":        47,
        "log_error_steps":         0,
        "log_total_lines":         44760,
        "log_has_linux":           1,
        "log_has_macos":           1,
        "log_has_windows":         1,
        "log_num_os_types":        3,
        "log_early3_total_dur":    20.075384,
        "log_early3_max_dur":      10.715605,
        "log_early3_min_dur":      0.262531,
        "log_early3_shell_count":  1,
        "log_early3_action_count": 2,
        "log_early3_error_count":  0,
        "log_early3_avg_dur":      6.691794666666667,
        "log_error_rate":          0.0,
        "log_shell_ratio":         0.4777777777777778,
        "log_avg_step_dur":        330.52050606666666,
        "log_max_step_dur":        13188.428275,
        "metadata_conclusion":     "success",
        "metadata_event":          "push",
        "metadata_status":         "completed",
        "metadata_actor_login":    "ivilata",
        "metadata_created_at":     "2023-09-25T09:38:43Z",
        "metadata_repository_full_name": "PyTables/PyTables",
    },
    # ── Row 2 ──────────────────────────────────────────────────────────────
    {
        "_id":                     "pytables/pytables_.github/workflows/wheels.yml_203_1",
        "repository_name":         "pytables/pytables",
        "workflow_path":           ".github/workflows/wheels.yml",
        "run_number":              203,
        "run_attempt":             1,
        "total_logs_size":         2734797.0,
        "log_num_jobs":            5,
        "log_total_steps":         27,
        "log_total_duration_sec":  2349.4265,
        "log_shell_steps":         10,
        "log_action_steps":        17,
        "log_error_steps":         0,
        "log_total_lines":         30623,
        "log_has_linux":           1,
        "log_has_macos":           1,
        "log_has_windows":         1,
        "log_num_os_types":        3,
        "log_early3_total_dur":    16.44417,
        "log_early3_max_dur":      13.631518,
        "log_early3_min_dur":      0.874104,
        "log_early3_shell_count":  0,
        "log_early3_action_count": 3,
        "log_early3_error_count":  0,
        "log_early3_avg_dur":      5.48139,
        "log_error_rate":          0.0,
        "log_shell_ratio":         0.3703703703703703,
        "log_avg_step_dur":        87.0157962962963,
        "log_max_step_dur":        489.451196,
        "metadata_conclusion":     "failure",
        "metadata_event":          "push",
        "metadata_status":         "completed",
        "metadata_actor_login":    "ivilata",
        "metadata_created_at":     "2023-09-26T12:15:50Z",
        "metadata_repository_full_name": "PyTables/PyTables",
    },
    # ── Row 3 ──────────────────────────────────────────────────────────────
    {
        "_id":                     "pytables/pytables_.github/workflows/wheels.yml_204_1",
        "repository_name":         "pytables/pytables",
        "workflow_path":           ".github/workflows/wheels.yml",
        "run_number":              204,
        "run_attempt":             1,
        "total_logs_size":         5389593.0,
        "log_num_jobs":            15,
        "log_total_steps":         78,
        "log_total_duration_sec":  27100.06335,
        "log_shell_steps":         37,
        "log_action_steps":        41,
        "log_error_steps":         0,
        "log_total_lines":         42247,
        "log_has_linux":           1,
        "log_has_macos":           1,
        "log_has_windows":         1,
        "log_num_os_types":        3,
        "log_early3_total_dur":    15.075727,
        "log_early3_max_dur":      8.506311,
        "log_early3_min_dur":      0.238914,
        "log_early3_shell_count":  1,
        "log_early3_action_count": 2,
        "log_early3_error_count":  0,
        "log_early3_avg_dur":      5.025242333333334,
        "log_error_rate":          0.0,
        "log_shell_ratio":         0.4743589743589743,
        "log_avg_step_dur":        347.43670961538464,
        "log_max_step_dur":        13972.402237,
        "metadata_conclusion":     "failure",
        "metadata_event":          "push",
        "metadata_status":         "completed",
        "metadata_actor_login":    "ivilata",
        "metadata_created_at":     "2023-09-26T14:20:14Z",
        "metadata_repository_full_name": "PyTables/PyTables",
    },
    # ── Row 4 ──────────────────────────────────────────────────────────────
    {
        "_id":                     "pytables/pytables_.github/workflows/wheels.yml_205_1",
        "repository_name":         "pytables/pytables",
        "workflow_path":           ".github/workflows/wheels.yml",
        "run_number":              205,
        "run_attempt":             1,
        "total_logs_size":         5347954.0,
        "log_num_jobs":            15,
        "log_total_steps":         78,
        "log_total_duration_sec":  26420.734518,
        "log_shell_steps":         37,
        "log_action_steps":        41,
        "log_error_steps":         0,
        "log_total_lines":         42180,
        "log_has_linux":           1,
        "log_has_macos":           1,
        "log_has_windows":         1,
        "log_num_os_types":        3,
        "log_early3_total_dur":    14.021396,
        "log_early3_max_dur":      7.46219,
        "log_early3_min_dur":      0.204022,
        "log_early3_shell_count":  1,
        "log_early3_action_count": 2,
        "log_early3_error_count":  0,
        "log_early3_avg_dur":      4.673798666666666,
        "log_error_rate":          0.0,
        "log_shell_ratio":         0.4743589743589743,
        "log_avg_step_dur":        338.72736561538466,
        "log_max_step_dur":        12166.338749,
        "metadata_conclusion":     "success",
        "metadata_event":          "push",
        "metadata_status":         "completed",
        "metadata_actor_login":    "ivilata",
        "metadata_created_at":     "2023-09-26T15:08:50Z",
        "metadata_repository_full_name": "PyTables/PyTables",
    },
]

FLOAT_TOLERANCE = 1e-4   # allow tiny floating-point differences
INPUT_CSV = "data/runs_200_2.csv"

# ─────────────────────────────────────────────────────────────────────────────

def vals_match(col, actual, expected):
    """Return True if actual == expected (with float tolerance)."""
    # Both missing
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    # Float comparison
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=FLOAT_TOLERANCE)
        except (TypeError, ValueError):
            return False
    # String / int comparison
    return str(actual).strip() == str(expected).strip()


def run_checks():
    print("=" * 65)
    print("  verify_output.py — Checking first 5 rows vs. expected values")
    print("=" * 65)

    # Load parsed CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n[FAIL] {INPUT_CSV} not found. Run fetch_data.py first.\n")
        sys.exit(1)

    print(f"\n  Loaded: {INPUT_CSV}  ({len(df):,} rows × {df.shape[1]} cols)\n")

    total_checks = 0
    total_pass   = 0
    total_fail   = 0
    failed_details = []

    for row_idx, expected_row in enumerate(EXPECTED):
        actual_row = df.iloc[row_idx]
        row_pass = 0
        row_fail = 0

        for col, exp_val in expected_row.items():
            total_checks += 1
            act_val = actual_row.get(col, None) if col in df.columns else None

            if vals_match(col, act_val, exp_val):
                row_pass += 1
                total_pass += 1
            else:
                row_fail += 1
                total_fail += 1
                failed_details.append({
                    "row": row_idx,
                    "col": col,
                    "expected": exp_val,
                    "actual":   act_val,
                })

        status = "✓ PASS" if row_fail == 0 else f"✗ FAIL ({row_fail} mismatches)"
        _id = expected_row["_id"]
        print(f"  Row {row_idx}  {status}")
        print(f"    _id      : {_id}")
        print(f"    repo     : {expected_row['repository_name']}")
        print(f"    workflow : {expected_row['workflow_path']}")
        print(f"    run      : {expected_row['run_number']} / attempt {expected_row['run_attempt']}")
        print(f"    log_size : {expected_row['total_logs_size']:,.0f} bytes")
        print(f"    jobs/steps: {expected_row['log_num_jobs']} jobs, {expected_row['log_total_steps']} steps")
        print(f"    duration : {expected_row['log_total_duration_sec']:.2f} sec")
        print(f"    error_rate: {expected_row['log_error_rate']}")
        print(f"    conclusion: {expected_row['metadata_conclusion']}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("-" * 65)
    print(f"  TOTAL CHECKS : {total_checks}")
    print(f"  PASSED       : {total_pass}  ✓")
    print(f"  FAILED       : {total_fail}  {'✗' if total_fail else '✓'}")
    print("-" * 65)

    if failed_details:
        print("\n  MISMATCHES DETAIL:")
        for f in failed_details:
            print(f"\n    Row {f['row']}  |  column: {f['col']}")
            print(f"      expected : {repr(f['expected'])}")
            print(f"      actual   : {repr(f['actual'])}")
        print()
        print("  [RESULT] ✗  Data does NOT match — check fetch_data.py flatten logic.")
        print("=" * 65)
        sys.exit(1)
    else:
        print("\n  [RESULT] ✓  All checks passed — parsed data matches sample CSV exactly!")
        print("=" * 65)
        sys.exit(0)


if __name__ == "__main__":
    run_checks()
