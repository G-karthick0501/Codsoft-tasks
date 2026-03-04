#!/usr/bin/env python3
"""
smart_scraper.py — GitHub API + regex-first React component extractor.

NO git cloning. No tree-sitter required. Pure requests + regex.
Uses ThreadPoolExecutor for concurrent file downloads.

Usage:
    python smart_scraper.py --token YOUR_GITHUB_TOKEN --output master2.csv

    # Custom repos:
    python smart_scraper.py --token TOKEN --repos shadcn-ui/ui radix-ui/primitives

    # Read from file:
    python smart_scraper.py --token TOKEN --repos-file repos.txt

Environment variable alternative (avoid putting token in shell history):
    set GITHUB_TOKEN=ghp_xxxxx
    python smart_scraper.py
"""

import os
import re
import csv
import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Default repos to scrape (UI component libraries — gold standard) ─────────
DEFAULT_REPOS = [
    "shadcn-ui/ui",
    "radix-ui/primitives",
    "mantinedev/mantine",
    "chakra-ui/chakra-ui",
    "ant-design/ant-design",
    "mui/material-ui",
    "adobe/react-spectrum",
    "tailwindlabs/headlessui",
    "ariakit/ariakit",
    "reach/reach-ui",
]

# ─── GitHub API session ───────────────────────────────────────────────────────
class GitHubSession:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        # Increase pool size to match our thread count
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3,
        )
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET with automatic rate-limit back-off and polite delay."""
        time.sleep(0.05)   # 50ms polite delay per request across all threads
        for attempt in range(5):
            r = self.session.get(url, timeout=20, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 1)
                log.warning(f"Rate limited — sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return r  # caller handles
            # Other error — brief back-off
            time.sleep(2 ** attempt)
        return r  # return last response

    def get_file_tree(self, repo: str, sha: str = "HEAD") -> list[dict]:
        """Return flat list of all blobs in repo tree (recursive)."""
        url = f"{self.BASE}/repos/{repo}/git/trees/{sha}?recursive=1"
        r = self.get(url)
        if r.status_code != 200:
            log.error(f"Failed to get tree for {repo}: {r.status_code}")
            return []
        data = r.json()
        if data.get("truncated"):
            log.warning(f"{repo}: tree truncated — very large repo, partial results")
        return data.get("tree", [])

    def get_default_branch(self, repo: str) -> str:
        r = self.get(f"{self.BASE}/repos/{repo}")
        if r.status_code != 200:
            return "main"
        return r.json().get("default_branch", "main")

    def get_file_content(self, repo: str, path: str, branch: str = "main") -> Optional[str]:
        """
        Download via raw.githubusercontent.com CDN — NO rate limit for public repos.
        Only tree fetch (2 API calls per repo) uses the rate-limited authenticated endpoint.
        """
        raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        try:
            r = requests.get(raw_url, timeout=15)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        return None


# ─── Path filters ─────────────────────────────────────────────────────────────
_EXCLUDE_PATTERNS = [
    "__tests__", "/__test__/", "/test/", "/tests/",
    ".stories.", ".story.", ".test.", ".spec.",
    "/examples/", "/example/", "/demo/", "/demos/",
    "/fixtures/", "/__fixtures__/", "/mocks/",
    "/node_modules/", "/.next/", "/dist/", "/build/",
    "/coverage/", "/storybook/",
]

_INCLUDE_DIRS = [
    "/src/", "/components/", "/ui/", "/lib/",
    "/packages/", "/modules/", "/shared/",
]


def should_include(path: str) -> bool:
    """Return True if this file path is worth downloading."""
    p = path.lower().replace("\\", "/")

    # Must be JSX/TSX
    if not (p.endswith(".tsx") or p.endswith(".jsx")):
        return False

    # Exclude known noise dirs
    for excl in _EXCLUDE_PATTERNS:
        if excl in p:
            return False

    return True


# ─── Regex-first feature extraction ──────────────────────────────────────────

# Hooks
_RE_ANY_HOOK   = re.compile(r'\buse[A-Z]\w*\b')
_NAMED_HOOKS   = ["useState", "useEffect", "useCallback", "useMemo",
                   "useContext", "useReducer", "useRef", "useLayoutEffect"]

# Structure
_RE_TERNARY    = re.compile(r'(?<![=!<>?])\?(?![?.:])')   # ternary ?
_RE_MAP        = re.compile(r'\.map\s*\(')
_RE_FILTER     = re.compile(r'\.filter\s*\(')
_RE_REDUCE     = re.compile(r'\.reduce\s*\(')

# Lexical / semantic
_RE_FETCH      = re.compile(r'\bfetch\s*\(|\buseQuery\b|\buseSWR\b|\baxios\b|\buseInfiniteQuery\b')
_RE_IMPORT     = re.compile(r'^import\s+', re.MULTILINE)
_RE_ON_PROP    = re.compile(r'\bon[A-Z]\w+')             # event handler props
_RE_BOOL_PROP  = re.compile(r'\b(?:is|has|show|can|should|enable|disable)\w+')

# Component name from signature
_COMP_PATTERNS = [
    re.compile(r'export\s+default\s+function\s+([A-Z]\w*)'),
    re.compile(r'export\s+(?:const|let)\s+([A-Z]\w*)\s*(?:=|\:)'),
    re.compile(r'function\s+([A-Z]\w*)\s*\('),
    re.compile(r'const\s+([A-Z]\w*)\s*=\s*(?:React\.)?(?:memo|forwardRef)'),
    re.compile(r'const\s+([A-Z]\w*)\s*=\s*\('),
]

def extract_component_name(code: str) -> str:
    for pat in _COMP_PATTERNS:
        m = pat.search(code)
        if m:
            return m.group(1)
    return ""


def extract_prop_names(code: str) -> list[str]:
    """Pull prop names from the component's first parameter destructuring."""
    # Look for ({ prop1, prop2, ... }) or function Foo({ prop1 })
    m = re.search(r'(?:function\s+\w+|=)\s*\(\s*\{([^}]{0,400})\}', code)
    if m:
        raw = m.group(1)
        # Extract identifiers (before : or = or ,)
        props = re.findall(r'\b([a-zA-Z_]\w*)(?:\s*[=:,\n]|\s*$)', raw)
        # Filter noise
        return [p for p in props if p not in ("", "true", "false", "null")]
    return []


def jsx_depth_stack(code: str) -> int:
    """
    Fast stack-based JSX depth counter.
    Much faster than tree-sitter for this specific metric.
    """
    depth = 0
    max_d = 0
    # Find JSX open/close tags (not self-closing, not comments)
    # Simple heuristic: count < and /> patterns
    i = 0
    n = len(code)
    while i < n:
        if code[i] == '<':
            # Skip comments <!-- ... -->
            if code[i:i+4] == '<!--':
                end = code.find('-->', i)
                i = end + 3 if end != -1 else n
                continue
            # Self-closing: <Tag ... />
            # Closing: </Tag>
            # Opening: <Tag ...>
            if i + 1 < n and code[i+1] == '/':
                # closing tag
                depth = max(depth - 1, 0)
            elif i + 1 < n and code[i+1].isupper():
                # JSX component open tag (capital letter)
                # Check if self-closing
                end = code.find('>', i)
                if end != -1 and end > 0 and code[end-1] == '/':
                    pass  # self-closing, no depth change
                else:
                    depth += 1
                    max_d = max(max_d, depth)
        i += 1
    return max_d


def extract_features(code: str, repo: str, path: str) -> Optional[dict]:
    """
    Full feature extraction: quality gate + all metrics.
    Returns None if the file fails quality checks.
    """
    # ── Quality gate ──────────────────────────────────────────────────────────
    lines = code.splitlines()
    loc = len(lines)

    # Too short or too long
    if loc < 15 or loc > 500:
        return None

    # Must look like a React component
    has_return_jsx = bool(re.search(r'return\s*\(?\s*<', code))
    if not has_return_jsx:
        return None

    # Must have at least one JSX element
    has_jsx_elem = bool(re.search(r'<[A-Z][a-zA-Z]*|<(?:div|span|button|input|form|ul|ol|li|a|p|h[1-6]|img|svg)', code))
    if not has_jsx_elem:
        return None

    # ── Hooks ─────────────────────────────────────────────────────────────────
    all_hooks = _RE_ANY_HOOK.findall(code)
    hooks_total = len(set(all_hooks))          # unique hook types used
    hook_counts = {h: all_hooks.count(h) for h in _NAMED_HOOKS}
    custom_hooks = sum(1 for h in set(all_hooks) if h not in _NAMED_HOOKS)

    # ── Props ─────────────────────────────────────────────────────────────────
    prop_names = extract_prop_names(code)
    props_count = len(prop_names)

    # Must have something interesting
    if hooks_total == 0 and props_count == 0:
        return None

    # ── JSX structure ─────────────────────────────────────────────────────────
    depth = jsx_depth_stack(code)
    if depth < 2:
        return None

    jsx_elems_raw = len(re.findall(r'<[A-Z][a-zA-Z]*|<(?:div|span|button|input|form)', code))

    # ── Behavioral signals ────────────────────────────────────────────────────
    # Strip string literals before counting ternaries to reduce false positives
    code_no_strings = re.sub(r'`[^`]*`|"[^"]*"|\'[^\']*\'', '""', code)
    conditionals  = len(_RE_TERNARY.findall(code_no_strings))
    map_calls     = len(_RE_MAP.findall(code))
    filter_calls  = len(_RE_FILTER.findall(code))
    reduce_calls  = len(_RE_REDUCE.findall(code))

    # ── Lexical signals ───────────────────────────────────────────────────────
    has_fetch    = 1 if _RE_FETCH.search(code) else 0
    num_imports  = len(_RE_IMPORT.findall(code))

    on_props     = [p for p in prop_names if re.match(r'^on[A-Z]', p)]
    bool_props   = [p for p in prop_names if re.match(r'^(?:is|has|show|can|should)', p)]
    has_children = 1 if "children" in prop_names else 0

    # ── Comment extraction ────────────────────────────────────────────────────
    # Grab the first JSDoc block or leading // comment block
    comment = ""
    jsdoc = re.search(r'/\*\*(.*?)\*/', code, re.DOTALL)
    if jsdoc:
        comment = re.sub(r'\s*\*\s?', ' ', jsdoc.group(1)).strip()[:200]
    else:
        # leading // comments before a function/const
        m = re.search(r'((?:^\s*//[^\n]*\n)+)\s*(?:export|const|function)', code, re.MULTILINE)
        if m:
            comment = re.sub(r'^\s*//\s?', '', m.group(1), flags=re.MULTILINE).strip()[:200]

    comp_name = extract_component_name(code)

    return {
        "repo": repo,
        "file": path,
        "component": comp_name,
        "loc": loc,
        "hooks_total": hooks_total,
        "useState":      hook_counts.get("useState", 0),
        "useEffect":     hook_counts.get("useEffect", 0),
        "useCallback":   hook_counts.get("useCallback", 0),
        "useMemo":       hook_counts.get("useMemo", 0),
        "useContext":    hook_counts.get("useContext", 0),
        "useReducer":    hook_counts.get("useReducer", 0),
        "useRef":        hook_counts.get("useRef", 0),
        "useCustom":     custom_hooks,
        "props":         props_count,
        "jsx_depth":     depth,
        "jsx_elems":     jsx_elems_raw,
        "conditionals":  conditionals,
        "map_calls":     map_calls,
        "filter_calls":  filter_calls,
        "reduce_calls":  reduce_calls,
        "has_fetch":     has_fetch,
        "num_imports":   num_imports,
        "event_handlers": len(on_props),
        "bool_props":    len(bool_props),
        "has_children":  has_children,
        "prop_names":    ";".join(prop_names[:20]),   # cap at 20 props
        "comment":       comment.replace("\n", " "),
    }


# ─── Per-repo scraping ────────────────────────────────────────────────────────
def scrape_repo(repo: str, gh: GitHubSession, max_files: int = 0) -> list[dict]:
    """
    Full pipeline for one repo:
    1. Get file tree via API
    2. Filter to JSX/TSX component files
    3. Download + extract in parallel
    """
    log.info(f"[{repo}] Getting default branch...")
    branch = gh.get_default_branch(repo)

    log.info(f"[{repo}] Fetching file tree (branch={branch})...")
    tree = gh.get_file_tree(repo, sha=branch)
    if not tree:
        log.warning(f"[{repo}] Empty tree, skipping")
        return []

    # Filter to component candidates
    candidates = [
        item["path"] for item in tree
        if item["type"] == "blob" and should_include(item["path"])
    ]

    if max_files and len(candidates) > max_files:
        # Prioritise shorter paths (more likely src/components/X.tsx)
        candidates.sort(key=lambda p: len(p))
        candidates = candidates[:max_files]

    log.info(f"[{repo}] {len(candidates)} candidate files to download")

    results: list[dict] = []
    failed = 0
    skipped = 0

    def download_and_extract(path: str) -> Optional[dict]:
        content = gh.get_file_content(repo, path, branch)
        if content is None:
            return None
        return extract_features(content, repo, path)

    # CDN downloads have no rate limit — use 8 threads safely
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_and_extract, p): p for p in candidates}

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                log.info(f"[{repo}] {done}/{len(candidates)} files processed, {len(results)} kept")
            try:
                result = future.result()
            except Exception as e:
                log.debug(f"Error processing {futures[future]}: {e}")
                failed += 1
                continue

            if result is None:
                skipped += 1
            else:
                results.append(result)

    log.info(f"[{repo}] Done — kept={len(results)}, skipped={skipped}, failed={failed}")
    return results


# ─── CSV writer ───────────────────────────────────────────────────────────────
COLUMNS = [
    "repo", "file", "component", "loc",
    "hooks_total", "useState", "useEffect", "useCallback",
    "useMemo", "useContext", "useReducer", "useRef", "useCustom",
    "props", "jsx_depth", "jsx_elems",
    "conditionals", "map_calls", "filter_calls", "reduce_calls",
    "has_fetch", "num_imports",
    "event_handlers", "bool_props", "has_children",
    "prop_names", "comment",
]


def append_to_csv(rows: list[dict], output_path: str):
    path = Path(output_path)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Smart React component scraper — GitHub API + regex, no git cloning."
    )
    parser.add_argument(
        "--token", "-t",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)"
    )
    parser.add_argument(
        "--output", "-o",
        default="master2.csv",
        help="Output CSV path (default: master2.csv)"
    )
    parser.add_argument(
        "--repos", "-r",
        nargs="+",
        help="Repo(s) to scrape in owner/name format"
    )
    parser.add_argument(
        "--repos-file",
        help="Text file with one repo per line (owner/name)"
    )
    parser.add_argument(
        "--max-files-per-repo",
        type=int, default=0,
        help="Cap file downloads per repo (0=unlimited)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=4,
        help="Number of repos to process in parallel (default: 4)"
    )
    args = parser.parse_args()

    # ── Token check ───────────────────────────────────────────────────────────
    if not args.token:
        print("ERROR: GitHub token required. Pass --token or set GITHUB_TOKEN env var.")
        print("Get one at: https://github.com/settings/tokens (no scopes needed for public repos)")
        sys.exit(1)

    # ── Build repo list ───────────────────────────────────────────────────────
    repos = list(args.repos or [])
    if args.repos_file:
        with open(args.repos_file) as f:
            repos += [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not repos:
        log.info("No repos specified — using default UI component libraries")
        repos = DEFAULT_REPOS

    log.info(f"Will scrape {len(repos)} repos → {args.output}")

    gh = GitHubSession(args.token)
    total = 0

    if args.workers == 1 or len(repos) == 1:
        # Simple sequential mode
        for repo in repos:
            rows = scrape_repo(repo, gh, max_files=args.max_files_per_repo)
            if rows:
                append_to_csv(rows, args.output)
                total += len(rows)
                log.info(f"Running total: {total} components")
    else:
        # Parallel repo processing
        # Note: each repo already uses 8 threads internally for file downloads,
        # so don't go too wild on outer parallelism — 3-4 is plenty.
        outer_workers = min(args.workers, 4)
        log.info(f"Processing {len(repos)} repos with {outer_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=outer_workers) as executor:
            futures = {
                executor.submit(scrape_repo, repo, gh, args.max_files_per_repo): repo
                for repo in repos
            }
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    rows = future.result()
                    if rows:
                        append_to_csv(rows, args.output)
                        total += len(rows)
                        log.info(f"[{repo}] Written {len(rows)} rows. Running total: {total}")
                except Exception as e:
                    log.error(f"[{repo}] Failed: {e}")

    log.info(f"DONE. {total} components written to {args.output}")
    log.info(f"Run `py analyze_csv.py` (update CSV_PATH to {args.output!r}) to inspect results.")


if __name__ == "__main__":
    main()
