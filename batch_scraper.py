#!/usr/bin/env python3
"""
Clone a list of React (or Next) repos and run the structural extractor on each.
Append results to a single CSV for large-scale dataset creation.

Usage:
    python batch_scraper.py repos.txt master.csv

The first argument is a newline-separated list of git URLs. The second is the
output CSV path. Existing CSV will be appended to.

The script assumes `structural_poc.py` is in the same directory and the
Python environment has all required packages.
"""

import sys
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
import concurrent.futures
import hashlib
import time


def clone_and_run(repo_url, tmp_dir, output_csv, excludes, max_per_repo):
    name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    # make a deterministic dir name
    digest = hashlib.sha1(repo_url.encode()).hexdigest()[:8]
    dest = tmp_dir / f"{name}-{digest}"
    perrepo_csv = tmp_dir / f"{name}-{digest}-structural.csv"
    try:
        print(f"cloning {repo_url} into {dest}")
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])
    except subprocess.CalledProcessError:
        print(f"failed to clone {repo_url}, skipping")
        return None
    # run parser with exclude patterns and cap
    cmd = [sys.executable, "structural_poc.py", str(dest), "--output", str(perrepo_csv)]
    for ex in excludes:
        cmd += ["--exclude", ex]
    if max_per_repo:
        cmd += ["--max-components", str(max_per_repo)]
    print(f"processing {name}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(f"structural_poc failed for {name}")
        return None
    if perrepo_csv.exists():
        return perrepo_csv
    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("repos_file")
    p.add_argument("out_csv")
    p.add_argument("--workers", "-w", type=int, default=4)
    p.add_argument("--clones-dir", default=None)
    p.add_argument("--exclude", "-x", action="append", default=[],
                   help="path patterns to exclude (passed to structural_poc)")
    p.add_argument("--max-per-repo", type=int, default=0,
                   help="cap components per repo (0 = unlimited)")
    args = p.parse_args()

    repos_file = Path(args.repos_file)
    out_csv = args.out_csv
    if not repos_file.exists():
        print(f"repo list {repos_file} not found")
        sys.exit(1)
    with open(repos_file) as f:
        repos = [line.strip() for line in f if line.strip()]

    clones_root = Path(args.clones_dir) if args.clones_dir else Path(tempfile.mkdtemp(prefix="repo-clones-"))
    clones_root.mkdir(parents=True, exist_ok=True)
    processed = set()
    proc_file = Path("processed_repos.txt")
    if proc_file.exists():
        with open(proc_file) as f:
            processed = set(l.strip() for l in f if l.strip())

    futures = []
    perrepo_csvs = []
    start = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for url in repos:
                if url in processed:
                    print(f"skipping already processed {url}")
                    continue
                futures.append(ex.submit(clone_and_run, url, clones_root, out_csv, args.exclude, args.max_per_repo))
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    perrepo_csvs.append(Path(res))
    finally:
        print(f"worker run time {time.time()-start:.1f}s")

    # append all per-repo CSVs into out_csv
    first_written = False
    for pcsv in perrepo_csvs:
        if not pcsv.exists():
            continue
        if not first_written and not Path(out_csv).exists():
            # write header+content
            shutil.copy(pcsv, out_csv)
            first_written = True
        else:
            # append skipping header
            with open(pcsv) as src, open(out_csv, 'a', newline='') as dst:
                lines = src.readlines()[1:]
                dst.writelines(lines)
        # mark processed by repo URL hash in filename
        with open(proc_file, 'a') as f:
            f.write(pcsv.name + "\n")
        pcsv.unlink()

    if args.clones_dir is None:
        print(f"cleaning up {clones_root}")
        shutil.rmtree(clones_root)
    print(f"done. aggregated into {out_csv}")

if __name__ == "__main__":
    main()
