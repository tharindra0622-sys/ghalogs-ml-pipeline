import os
import gzip
import json
import argparse
import urllib.request
import pandas as pd
from datetime import datetime, timezone

ZENODO_URL = "https://zenodo.org/records/14796970/files/runs.json.gz?download=1"
LOCAL_GZ   = "data/runs.json.gz"
OUTPUT_CSV = "data/runs_200_2.csv"

KEEP_COLUMNS = [
    "_id", "repository_name", "workflow_path", "run_number", "run_attempt",
    "metadata_id", "metadata_name", "metadata_node_id", "metadata_head_branch",
    "metadata_head_sha", "metadata_path", "metadata_display_title",
    "metadata_run_number", "metadata_event", "metadata_status",
    "metadata_conclusion", "metadata_workflow_id", "metadata_check_suite_id",
    "metadata_check_suite_node_id", "metadata_pull_requests",
    "metadata_created_at", "metadata_updated_at", "metadata_actor_login",
    "metadata_actor_id", "metadata_actor_node_id", "metadata_actor_gravatar_id",
    "metadata_actor_type", "metadata_actor_site_admin", "metadata_run_attempt",
    "metadata_referenced_workflows", "metadata_run_started_at",
    "metadata_triggering_actor_login", "metadata_triggering_actor_id",
    "metadata_triggering_actor_node_id", "metadata_triggering_actor_gravatar_id",
    "metadata_triggering_actor_type", "metadata_triggering_actor_site_admin",
    "metadata_head_commit_id", "metadata_head_commit_tree_id",
    "metadata_head_commit_message", "metadata_head_commit_timestamp",
    "metadata_head_commit_author_name", "metadata_head_commit_author_email",
    "metadata_head_commit_committer_name", "metadata_head_commit_committer_email",
    "metadata_repository_id", "metadata_repository_node_id",
    "metadata_repository_name", "metadata_repository_full_name",
    "metadata_repository_private", "metadata_repository_owner_login",
    "metadata_repository_owner_id", "metadata_repository_owner_node_id",
    "metadata_repository_owner_gravatar_id", "metadata_repository_owner_type",
    "metadata_repository_owner_site_admin", "metadata_repository_description",
    "metadata_repository_fork", "metadata_head_repository_id",
    "metadata_head_repository_node_id", "metadata_head_repository_name",
    "metadata_head_repository_full_name", "metadata_head_repository_private",
    "metadata_head_repository_owner_login", "metadata_head_repository_owner_id",
    "metadata_head_repository_owner_node_id",
    "metadata_head_repository_owner_gravatar_id",
    "metadata_head_repository_owner_type",
    "metadata_head_repository_owner_site_admin",
    "metadata_head_repository_description", "metadata_head_repository_fork",
    "logs_archive_path", "total_logs_size",
    "log_num_jobs", "log_total_steps", "log_total_duration_sec",
    "log_shell_steps", "log_action_steps", "log_error_steps",
    "log_total_lines", "log_has_linux", "log_has_macos", "log_has_windows",
    "log_num_os_types", "log_early3_total_dur", "log_early3_max_dur",
    "log_early3_min_dur", "log_early3_shell_count", "log_early3_action_count",
    "log_early3_error_count", "log_early3_avg_dur", "log_error_rate",
    "log_shell_ratio", "log_avg_step_dur", "log_max_step_dur",
]

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def download_file():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LOCAL_GZ):
        size_mb = os.path.getsize(LOCAL_GZ) / 1024 / 1024
        log(f"runs.json.gz already exists ({size_mb:.1f} MB) — skipping download.")
        return
    log("Downloading runs.json.gz from Zenodo (~1.1 GB)...")
    def reporthook(block_num, block_size, total_size):
        mb = block_num * block_size / 1024 / 1024
        if block_num % 1000 == 0:
            if total_size > 0:
                pct = block_num * block_size / total_size * 100
                print(f"\r  {mb:.1f} MB / {total_size/1024/1024:.1f} MB ({pct:.1f}%)", end="", flush=True)
    urllib.request.urlretrieve(ZENODO_URL, LOCAL_GZ, reporthook)
    print()
    log("Download complete.")

def flatten_record(record):
    flat = {}
    flat["_id"]               = record.get("_id", "")
    flat["repository_name"]   = record.get("repository_name", "")
    flat["workflow_path"]     = record.get("workflow_path", "")
    flat["run_number"]        = record.get("run_number", 0)
    flat["run_attempt"]       = record.get("run_attempt", 0)
    flat["logs_archive_path"] = record.get("logs_archive_path", "")
    flat["total_logs_size"]   = record.get("total_logs_size", None)

    flat["log_num_jobs"]            = record.get("log_num_jobs", 0)
    flat["log_total_steps"]         = record.get("log_total_steps", 0)
    flat["log_total_duration_sec"]  = record.get("log_total_duration_sec", 0.0)
    flat["log_shell_steps"]         = record.get("log_shell_steps", 0)
    flat["log_action_steps"]        = record.get("log_action_steps", 0)
    flat["log_error_steps"]         = record.get("log_error_steps", 0)
    flat["log_total_lines"]         = record.get("log_total_lines", 0)
    flat["log_has_linux"]           = int(record.get("log_has_linux", 0))
    flat["log_has_macos"]           = int(record.get("log_has_macos", 0))
    flat["log_has_windows"]         = int(record.get("log_has_windows", 0))
    flat["log_num_os_types"]        = record.get("log_num_os_types", 0)
    flat["log_early3_total_dur"]    = record.get("log_early3_total_dur", 0.0)
    flat["log_early3_max_dur"]      = record.get("log_early3_max_dur", 0.0)
    flat["log_early3_min_dur"]      = record.get("log_early3_min_dur", 0.0)
    flat["log_early3_shell_count"]  = record.get("log_early3_shell_count", 0)
    flat["log_early3_action_count"] = record.get("log_early3_action_count", 0)
    flat["log_early3_error_count"]  = record.get("log_early3_error_count", 0)
    flat["log_early3_avg_dur"]      = record.get("log_early3_avg_dur", 0.0)
    flat["log_error_rate"]          = record.get("log_error_rate", 0.0)
    flat["log_shell_ratio"]         = record.get("log_shell_ratio", 0.0)
    flat["log_avg_step_dur"]        = record.get("log_avg_step_dur", 0.0)
    flat["log_max_step_dur"]        = record.get("log_max_step_dur", 0.0)

    meta = record.get("metadata", {}) or {}
    flat["metadata_id"]               = meta.get("id", None)
    flat["metadata_name"]             = meta.get("name", "")
    flat["metadata_node_id"]          = meta.get("node_id", "")
    flat["metadata_head_branch"]      = meta.get("head_branch", "")
    flat["metadata_head_sha"]         = meta.get("head_sha", "")
    flat["metadata_path"]             = meta.get("path", "")
    flat["metadata_display_title"]    = meta.get("display_title", "")
    flat["metadata_run_number"]       = meta.get("run_number", None)
    flat["metadata_event"]            = meta.get("event", "")
    flat["metadata_status"]           = meta.get("status", "")
    flat["metadata_conclusion"]       = meta.get("conclusion", "")
    flat["metadata_workflow_id"]      = meta.get("workflow_id", None)
    flat["metadata_check_suite_id"]   = meta.get("check_suite_id", None)
    flat["metadata_check_suite_node_id"] = meta.get("check_suite_node_id", "")
    flat["metadata_pull_requests"]    = str(meta.get("pull_requests", []))
    flat["metadata_created_at"]       = meta.get("created_at", "")
    flat["metadata_updated_at"]       = meta.get("updated_at", "")
    flat["metadata_run_attempt"]      = meta.get("run_attempt", None)
    flat["metadata_referenced_workflows"] = str(meta.get("referenced_workflows", []))
    flat["metadata_run_started_at"]   = meta.get("run_started_at", "")

    actor = meta.get("actor", {}) or {}
    flat["metadata_actor_login"]       = actor.get("login", "")
    flat["metadata_actor_id"]          = actor.get("id", None)
    flat["metadata_actor_node_id"]     = actor.get("node_id", "")
    flat["metadata_actor_gravatar_id"] = actor.get("gravatar_id", None)
    flat["metadata_actor_type"]        = actor.get("type", "")
    flat["metadata_actor_site_admin"]  = actor.get("site_admin", False)

    ta = meta.get("triggering_actor", {}) or {}
    flat["metadata_triggering_actor_login"]       = ta.get("login", "")
    flat["metadata_triggering_actor_id"]          = ta.get("id", None)
    flat["metadata_triggering_actor_node_id"]     = ta.get("node_id", "")
    flat["metadata_triggering_actor_gravatar_id"] = ta.get("gravatar_id", None)
    flat["metadata_triggering_actor_type"]        = ta.get("type", "")
    flat["metadata_triggering_actor_site_admin"]  = ta.get("site_admin", False)

    hc = meta.get("head_commit", {}) or {}
    flat["metadata_head_commit_id"]        = hc.get("id", "")
    flat["metadata_head_commit_tree_id"]   = hc.get("tree_id", "")
    flat["metadata_head_commit_message"]   = hc.get("message", "")
    flat["metadata_head_commit_timestamp"] = hc.get("timestamp", "")
    author    = hc.get("author", {}) or {}
    committer = hc.get("committer", {}) or {}
    flat["metadata_head_commit_author_name"]     = author.get("name", "")
    flat["metadata_head_commit_author_email"]    = author.get("email", "")
    flat["metadata_head_commit_committer_name"]  = committer.get("name", "")
    flat["metadata_head_commit_committer_email"] = committer.get("email", "")

    repo   = meta.get("repository", {}) or {}
    rowner = repo.get("owner", {}) or {}
    flat["metadata_repository_id"]                = repo.get("id", None)
    flat["metadata_repository_node_id"]           = repo.get("node_id", "")
    flat["metadata_repository_name"]              = repo.get("name", "")
    flat["metadata_repository_full_name"]         = repo.get("full_name", "")
    flat["metadata_repository_private"]           = repo.get("private", False)
    flat["metadata_repository_description"]       = repo.get("description", "")
    flat["metadata_repository_fork"]              = repo.get("fork", False)
    flat["metadata_repository_owner_login"]       = rowner.get("login", "")
    flat["metadata_repository_owner_id"]          = rowner.get("id", None)
    flat["metadata_repository_owner_node_id"]     = rowner.get("node_id", "")
    flat["metadata_repository_owner_gravatar_id"] = rowner.get("gravatar_id", None)
    flat["metadata_repository_owner_type"]        = rowner.get("type", "")
    flat["metadata_repository_owner_site_admin"]  = rowner.get("site_admin", False)

    hrepo  = meta.get("head_repository", {}) or {}
    hrowner = hrepo.get("owner", {}) or {}
    flat["metadata_head_repository_id"]                = hrepo.get("id", None)
    flat["metadata_head_repository_node_id"]           = hrepo.get("node_id", "")
    flat["metadata_head_repository_name"]              = hrepo.get("name", "")
    flat["metadata_head_repository_full_name"]         = hrepo.get("full_name", "")
    flat["metadata_head_repository_private"]           = hrepo.get("private", False)
    flat["metadata_head_repository_description"]       = hrepo.get("description", "")
    flat["metadata_head_repository_fork"]              = hrepo.get("fork", False)
    flat["metadata_head_repository_owner_login"]       = hrowner.get("login", "")
    flat["metadata_head_repository_owner_id"]          = hrowner.get("id", None)
    flat["metadata_head_repository_owner_node_id"]     = hrowner.get("node_id", "")
    flat["metadata_head_repository_owner_gravatar_id"] = hrowner.get("gravatar_id", None)
    flat["metadata_head_repository_owner_type"]        = hrowner.get("type", "")
    flat["metadata_head_repository_owner_site_admin"]  = hrowner.get("site_admin", False)

    return flat

def parse_json_gz(max_rows=None):
    log(f"Parsing {LOCAL_GZ} ...")
    rows  = []
    count = 0
    with gzip.open(LOCAL_GZ, "rt", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            total = len(data)
            log(f"  JSON array — {total:,} records")
            for record in data:
                rows.append(flatten_record(record))
                count += 1
                if count % 50000 == 0:
                    log(f"  Parsed {count:,} / {total:,}")
                if max_rows and count >= max_rows:
                    break
        else:
            log("  Detected JSON-Lines format")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(flatten_record(json.loads(line)))
                    count += 1
                    if count % 50000 == 0:
                        log(f"  Parsed {count:,} rows")
                    if max_rows and count >= max_rows:
                        break
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP_COLUMNS]
    log(f"Parsed {len(df):,} rows x {df.shape[1]} columns")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)
    if not args.skip_download:
        download_file()
    df = parse_json_gz(max_rows=args.sample)
    df.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved {len(df):,} rows to {OUTPUT_CSV}")
    log("fetch_data.py complete.")

if __name__ == "__main__":
    main()
