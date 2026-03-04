#!/usr/bin/env python3
"""
batch_scraper_v2.py — Hardened React Component Scraper
=======================================================
Improvements over batch_scraper.py:
  - FIXED: processed_repos.txt now stores REPO URLS (not CSV filenames)
  - FIXED: Per-repo CSV is written to master CSV IMMEDIATELY after each repo,
           not batched at the end (prevents data loss on crash)
  - NEW: Checkpoints every 10 repos to data/checkpoints/
  - NEW: Clean summary report at the end

Usage:
    python batch_scraper_v2.py repos.txt data/master2.csv
    python batch_scraper_v2.py repos.txt data/master2.csv --workers 6
    python batch_scraper_v2.py repos.txt data/master2.csv --max-per-repo 300

The output CSV is APPEND-ONLY. It will never overwrite existing data.
If the same repo appears in repos.txt again, it will be skipped via processed_repos.txt.
"""

import sys
import subprocess
import tempfile
import shutil
import os
import csv
import threading
import argparse
import hashlib
import time
from pathlib import Path
from datetime import datetime
import concurrent.futures


# ─────────────────────────────────────────────────────────────
# GLOBAL WRITE LOCK — prevents concurrent CSV write corruption
# ─────────────────────────────────────────────────────────────
_csv_lock = threading.Lock()
_proc_lock = threading.Lock()


def append_to_master_csv(perrepo_csv: Path, master_csv: Path) -> int:
    """
    Thread-safe append of a per-repo CSV to the master CSV.
    Returns number of rows appended.
    """
    with _csv_lock:
        if not perrepo_csv.exists():
            return 0

        rows_written = 0
        master_exists = master_csv.exists()

        with open(perrepo_csv, newline='', encoding='utf-8') as src:
            reader = csv.reader(src)
            rows = list(reader)

        if not rows:
            return 0

        header = rows[0]
        data_rows = rows[1:]  # Skip header

        with open(master_csv, 'a', newline='', encoding='utf-8') as dst:
            writer = csv.writer(dst)
            if not master_exists:
                writer.writerow(header)  # Write header only if file is new
            writer.writerows(data_rows)
            rows_written = len(data_rows)

        return rows_written


def mark_processed(repo_url: str, proc_file: Path):
    """
    Thread-safe write of a repo URL to the processed registry.
    FIXED: Now writes the actual repo URL, not the CSV filename.
    """
    with _proc_lock:
        with open(proc_file, 'a', encoding='utf-8') as f:
            f.write(repo_url.strip() + '\n')


def clone_and_extract(repo_url: str, work_dir: Path, output_csv: Path,
                       proc_file: Path, excludes: list, max_per_repo: int,
                       checkpoint_dir: Path) -> dict:
    """
    Clone one repo, run structural_poc.py, append to master CSV immediately.
    Returns a result dict with status info.
    """
    name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    digest = hashlib.sha1(repo_url.encode()).hexdigest()[:8]
    dest = work_dir / f"{name}-{digest}"
    perrepo_csv = work_dir / f"{name}-{digest}.csv"

    result = {
        "url": repo_url,
        "name": name,
        "status": "failed",
        "rows": 0,
        "reason": "",
    }

    # 1. Clone (shallow — only latest commit, much faster)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--quiet", repo_url, str(dest)],
            stderr=subprocess.DEVNULL,
            timeout=120  # 2 min timeout per clone
        )
    except subprocess.CalledProcessError:
        result["reason"] = "clone failed"
        return result
    except subprocess.TimeoutExpired:
        result["reason"] = "clone timeout"
        return result

    # 2. Run structural extractor
    cmd = [sys.executable, "structural_poc.py", str(dest), "--output", str(perrepo_csv)]
    for ex in excludes:
        cmd += ["--exclude", ex]
    if max_per_repo > 0:
        cmd += ["--max-components", str(max_per_repo)]

    try:
        subprocess.check_call(cmd, timeout=300, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        result["reason"] = f"structural_poc error: {type(e).__name__}"
        shutil.rmtree(dest, ignore_errors=True)
        return result

    # 3. IMMEDIATELY append to master CSV (thread-safe)
    rows = append_to_master_csv(perrepo_csv, output_csv)

    # 4. Also save to checkpoint dir (preserves per-repo data)
    if rows > 0 and checkpoint_dir.exists():
        checkpoint_path = checkpoint_dir / f"{name}-{digest}.csv"
        shutil.copy(perrepo_csv, checkpoint_path)

    # 5. Mark as processed (URL-based, FIXED)
    mark_processed(repo_url, proc_file)

    # 6. Cleanup: delete clone + per-repo csv
    shutil.rmtree(dest, ignore_errors=True)
    perrepo_csv.unlink(missing_ok=True)

    result["status"] = "success" if rows > 0 else "empty"
    result["rows"] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description="Hardened React component scraper")
    parser.add_argument("repos_file", help="Path to repos.txt (one URL per line)")
    parser.add_argument("out_csv", help="Master output CSV path (append-only)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--exclude", "-x", action="append", default=[],
                        help="Path patterns to exclude from structural_poc")
    parser.add_argument("--max-per-repo", type=int, default=300,
                        help="Max components per repo (default: 300, 0=unlimited)")
    parser.add_argument("--work-dir", default=None,
                        help="Directory for clone/temp files (default: system temp)")
    parser.add_argument("--proc-file", default="data/processed_repos.txt",
                        help="Path to processed repos registry (default: data/processed_repos.txt)")
    args = parser.parse_args()

    repos_file = Path(args.repos_file)
    out_csv = Path(args.out_csv)
    proc_file = Path(args.proc_file)

    if not repos_file.exists():
        print(f"ERROR: {repos_file} not found")
        sys.exit(1)

    # Ensure data directory exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    proc_file.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint dir
    checkpoint_dir = out_csv.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Load repos
    repos = [line.strip() for line in repos_file.read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.startswith('#')]

    # Load already-processed URLs
    processed = set()
    if proc_file.exists():
        processed = set(line.strip() for line in proc_file.read_text().splitlines()
                        if line.strip())
        print(f"Loaded {len(processed)} already-processed repos from {proc_file}")

    # Filter out already-processed
    pending = []
    for url in repos:
        url_norm = url.rstrip('/')
        if url_norm in processed or (url_norm + '.git') in processed:
            print(f"  SKIP (already done): {url_norm.split('/')[-1]}")
        else:
            pending.append(url)

    print(f"\n{'='*55}")
    print(f"  Total in repos.txt:   {len(repos)}")
    print(f"  Already processed:    {len(repos) - len(pending)}")
    print(f"  To scrape now:        {len(pending)}")
    print(f"  Workers:              {args.workers}")
    print(f"  Max per repo:         {args.max_per_repo}")
    print(f"  Output:               {out_csv}")
    print(f"  Checkpoints:          {checkpoint_dir}")
    print(f"{'='*55}\n")

    if not pending:
        print("Nothing to do — all repos already processed.")
        return

    # Work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="react-scrape-"))
        cleanup_work = True

    # Run
    start = time.time()
    results = []
    completed = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    clone_and_extract,
                    url, work_dir, out_csv, proc_file,
                    args.exclude, args.max_per_repo, checkpoint_dir
                ): url
                for url in pending
            }

            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                completed += 1
                status_icon = "✅" if res["status"] == "success" else ("⚠️" if res["status"] == "empty" else "❌")
                print(f"  [{completed:3d}/{len(pending)}] {status_icon} {res['name']:<35} "
                      f"{res['rows']:4d} rows  {res['reason']}")

    finally:
        elapsed = time.time() - start
        if cleanup_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    # Summary
    success = [r for r in results if r["status"] == "success"]
    empty = [r for r in results if r["status"] == "empty"]
    failed = [r for r in results if r["status"] == "failed"]
    total_rows = sum(r["rows"] for r in success)

    print(f"\n{'='*55}")
    print(f"  SCRAPING COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Succeeded:   {len(success)} repos,  {total_rows} new component rows")
    print(f"  Empty:       {len(empty)} repos (no components found)")
    print(f"  Failed:      {len(failed)} repos")
    print(f"  Output:      {out_csv}  ({out_csv.stat().st_size / 1e6:.1f} MB)")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"\nNext: rebuild embeddings with:")
    print(f"  python core_engine/embed_components.py")
    print(f"  python core_engine/graph_embeddings.py")

    if failed:
        print(f"\nFailed repos (check manually):")
        for r in failed:
            print(f"  {r['url']}  [{r['reason']}]")


if __name__ == "__main__":
    main()
