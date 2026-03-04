#!/usr/bin/env python3
"""
repo_finder.py — Multi-Source React Repo Discovery
====================================================
Discovers high-quality React component repositories from 4 sources:
  1. GitHub Search API  (requires tokens for best results)
  2. Awesome-React-Components list (no auth, curated)
  3. Libraries.io npm search (no auth needed)
  4. A hand-curated priority list (always included)

Usage:
    # With tokens (recommended):
    python repo_finder.py --tokens GITHUB_TOKEN_1 GITHUB_TOKEN_2 --output repos.txt

    # Without tokens (uses public curated lists only):
    python repo_finder.py --output repos.txt

Output: repos.txt with one GitHub clone URL per line, deduplicated.
"""

import argparse
import time
import json
import re
import sys
from pathlib import Path
from itertools import cycle
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ─────────────────────────────────────────────────────────────
# HAND-CURATED PRIORITY LIST (Gold Standard, Always Included)
# These are the most structurally rich React component libraries.
# ─────────────────────────────────────────────────────────────
PRIORITY_REPOS = [
    "https://github.com/radix-ui/primitives",
    "https://github.com/shadcn-ui/ui",
    "https://github.com/mui/material-ui",
    "https://github.com/ant-design/ant-design",
    "https://github.com/chakra-ui/chakra-ui",
    "https://github.com/mantinedev/mantine",
    "https://github.com/react-bootstrap/react-bootstrap",
    "https://github.com/TanStack/table",
    "https://github.com/TanStack/router",
    "https://github.com/TanStack/form",
    "https://github.com/tailwindlabs/headlessui",
    "https://github.com/ariakit/ariakit",
    "https://github.com/adobe/react-spectrum",
    "https://github.com/nextui-org/nextui",
    "https://github.com/tremor-so/tremor",
    "https://github.com/horizon-ui/horizon-ui-chakra",
    "https://github.com/reduxjs/react-redux",
    "https://github.com/pmndrs/zustand",
    "https://github.com/recharts/recharts",
    "https://github.com/airbnb/visx",
    "https://github.com/react-spring/react-spring",
    "https://github.com/framer/motion",
    "https://github.com/react-hook-form/react-hook-form",
    "https://github.com/jaredpalmer/formik",
    "https://github.com/floating-ui/floating-ui",
    "https://github.com/downshift-js/downshift",
    "https://github.com/tannerlinsley/react-table",
    "https://github.com/react-dnd/react-dnd",
    "https://github.com/atlassian/react-beautiful-dnd",
    "https://github.com/bvaughn/react-window",
    "https://github.com/bvaughn/react-virtualized",
    "https://github.com/wojtekmaj/react-pdf",
    "https://github.com/react-dropzone/react-dropzone",
    "https://github.com/JedWatson/react-select",
    "https://github.com/TeamWertarbyte/material-ui-chip-input",
    "https://github.com/KyleAMathews/react-spinkit",
    "https://github.com/davidhu2000/react-spinners",
    "https://github.com/uuidjs/uuid",
    "https://github.com/Shopify/polaris",
    "https://github.com/grommet/grommet",
    "https://github.com/uber/baseweb",
    "https://github.com/kiwicom/orbit",
    "https://github.com/carbon-design-system/carbon",
    "https://github.com/primer/react",
    "https://github.com/elastic/eui",
    "https://github.com/patternfly/patternfly-react",
    "https://github.com/marmelab/react-admin",
    "https://github.com/refinedev/refine",
    "https://github.com/react-pdf-viewer/react-pdf-viewer",
    "https://github.com/dmtrKovalenko/date-io",
]

# ─────────────────────────────────────────────────────────────
# QUALITY FILTER CRITERIA
# ─────────────────────────────────────────────────────────────
EXCLUDE_PATTERNS = [
    "template", "starter", "boilerplate", "example", "demo", "tutorial",
    "learn", "awesome", "collection", "list", "hub", "portal", "landing",
    "todo", "clone", "playground", "showcase",
]
MIN_STARS = 80


def is_quality_repo(repo: dict) -> bool:
    """Filter function for GitHub API search results."""
    name = repo.get("name", "").lower()
    desc = (repo.get("description") or "").lower()
    stars = repo.get("stargazers_count", 0)

    if stars < MIN_STARS:
        return False
    if any(p in name for p in EXCLUDE_PATTERNS):
        return False
    # Must have TypeScript or JavaScript as primary language
    lang = (repo.get("language") or "").lower()
    if lang not in ("typescript", "javascript"):
        return False
    return True


# ─────────────────────────────────────────────────────────────
# SOURCE 1: GITHUB SEARCH API
# ─────────────────────────────────────────────────────────────
SEARCH_QUERIES = [
    "react component library language:TypeScript stars:>200",
    "react ui library language:TypeScript stars:>100",
    "react hooks library language:TypeScript stars:>150",
    "react data table component language:TypeScript stars:>100",
    "react form component language:TypeScript stars:>100",
    "react animation component language:TypeScript stars:>80",
    "react modal drawer component language:JavaScript stars:>200",
    "react virtualized list component language:TypeScript stars:>100",
    "react design system language:TypeScript stars:>200",
    "headless ui components react language:TypeScript stars:>100",
]


def github_search(tokens: list, output_set: set, verbose: bool = True) -> int:
    """Run GitHub search queries with token rotation. Returns count of new repos found."""
    if not tokens:
        print("  [GitHub Search] No tokens provided — skipping API search.")
        return 0

    token_cycle = cycle(tokens)
    found = 0

    for query in SEARCH_QUERIES:
        token = next(token_cycle)
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 30}
        url = "https://api.github.com/search/repositories"

        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)

            if r.status_code == 403:
                print(f"  [GitHub Search] Rate limited on query '{query[:40]}'. Waiting 60s...")
                time.sleep(60)
                r = requests.get(url, headers=headers, params=params, timeout=10)

            if r.status_code != 200:
                print(f"  [GitHub Search] Error {r.status_code} for query '{query[:40]}'")
                continue

            items = r.json().get("items", [])
            for item in items:
                if is_quality_repo(item):
                    clone_url = item["clone_url"]
                    if clone_url not in output_set:
                        output_set.add(clone_url)
                        found += 1

            if verbose:
                print(f"  [GitHub Search] '{query[:50]}' -> {len(items)} results, {found} total unique so far")

            # Respect secondary rate limit (1 search per 2 seconds)
            time.sleep(2)

        except Exception as e:
            print(f"  [GitHub Search] Request failed: {e}")
            time.sleep(5)

    return found


# ─────────────────────────────────────────────────────────────
# SOURCE 2: AWESOME-REACT-COMPONENTS (no auth, curated)
# ─────────────────────────────────────────────────────────────
AWESOME_RAW_URL = "https://raw.githubusercontent.com/brillout/awesome-react-components/master/README.md"


def scrape_awesome_list(output_set: set) -> int:
    """Parse the awesome-react-components README for GitHub URLs."""
    print("  [Awesome List] Fetching brilliant/awesome-react-components...")
    try:
        r = requests.get(AWESOME_RAW_URL, timeout=15)
        if r.status_code != 200:
            print(f"  [Awesome List] Failed: HTTP {r.status_code}")
            return 0

        # Find all GitHub repo URLs
        pattern = r"https://github\.com/([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)"
        matches = re.findall(pattern, r.text)

        found = 0
        for match in matches:
            # Skip org/profile pages, issues, etc.
            parts = match.strip("/").split("/")
            if len(parts) != 2:
                continue
            owner, repo = parts
            # Skip meta links
            if repo.lower() in ("awesome-react-components", "react", "awesome"):
                continue

            clone_url = f"https://github.com/{owner}/{repo}.git"
            if clone_url not in output_set:
                output_set.add(clone_url)
                found += 1

        print(f"  [Awesome List] Found {found} new repos from {len(matches)} links")
        return found

    except Exception as e:
        print(f"  [Awesome List] Error: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# SOURCE 3: LIBRARIES.IO (no auth for limited use)
# ─────────────────────────────────────────────────────────────
def scrape_libraries_io(output_set: set, api_key: Optional[str] = None) -> int:
    """Query Libraries.io for top React component npm packages."""
    print("  [Libraries.io] Searching top react component packages...")
    search_terms = ["react-component", "react-ui", "react-hook", "react-table", "react-form"]
    found = 0

    for term in search_terms:
        params = {"q": term, "platforms": "npm", "sort": "dependents_count", "per_page": 30}
        if api_key:
            params["api_key"] = api_key

        try:
            r = requests.get("https://libraries.io/api/search", params=params, timeout=10)
            if r.status_code != 200:
                print(f"  [Libraries.io] HTTP {r.status_code} for '{term}' — skipping")
                continue

            packages = r.json()
            for pkg in packages:
                repo_url = pkg.get("repository_url", "")
                if "github.com" in repo_url:
                    # Normalize to clone URL
                    clean = repo_url.rstrip("/")
                    if not clean.endswith(".git"):
                        clean += ".git"
                    if clean not in output_set:
                        output_set.add(clean)
                        found += 1

            time.sleep(1)  # Be polite to libraries.io

        except Exception as e:
            print(f"  [Libraries.io] Error for '{term}': {e}")

    print(f"  [Libraries.io] Found {found} new repos")
    return found


# ─────────────────────────────────────────────────────────────
# SOURCE 4: PRIORITY LIST (always added)
# ─────────────────────────────────────────────────────────────
def add_priority_repos(output_set: set) -> int:
    found = 0
    for url in PRIORITY_REPOS:
        clone_url = url if url.endswith(".git") else url + ".git"
        # Also accept plain https format
        if clone_url not in output_set:
            # Try both with and without .git
            plain = url.rstrip("/")
            if plain not in output_set and (plain + ".git") not in output_set:
                output_set.add(clone_url)
                found += 1
    print(f"  [Priority List] Added {found} priority repos")
    return found


# ─────────────────────────────────────────────────────────────
# DEDUPLICATION: Normalize all URLs to consistent format
# ─────────────────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.endswith(".git"):
        url += ".git"
    return url


def deduplicate(urls: set) -> list:
    """Normalize and deduplicate, removing known non-component repos."""
    normalized = {}
    for url in urls:
        key = normalize_url(url).lower()
        normalized[key] = normalize_url(url)

    # Filter out obvious non-component repos
    filtered = []
    for key, url in normalized.items():
        parts = key.split("/")
        repo_name = parts[-1].replace(".git", "") if parts else ""
        if not any(p in repo_name for p in EXCLUDE_PATTERNS):
            filtered.append(url)

    return sorted(filtered)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Discover React component repos from multiple sources")
    parser.add_argument("--tokens", nargs="+", default=[], metavar="TOKEN",
                        help="GitHub Personal Access Tokens (classic, public_repo scope). Multiple tokens will be rotated.")
    parser.add_argument("--libraries-key", default=None, metavar="KEY",
                        help="Libraries.io API key (optional, improves rate limits)")
    parser.add_argument("--output", "-o", default="repos.txt",
                        help="Output file path (default: repos.txt)")
    parser.add_argument("--no-awesome", action="store_true",
                        help="Skip awesome-react-components list")
    parser.add_argument("--no-libraries", action="store_true",
                        help="Skip libraries.io search")
    parser.add_argument("--no-github-search", action="store_true",
                        help="Skip GitHub Search API (still uses priority list + awesome list)")
    args = parser.parse_args()

    # Load existing repos.txt to avoid overwriting existing work
    out_path = Path(args.output)
    repo_set: set = set()
    if out_path.exists():
        existing = [line.strip() for line in out_path.read_text().splitlines() if line.strip()]
        repo_set.update(existing)
        print(f"Loaded {len(repo_set)} existing repos from {out_path}")

    print("\n=== Repo Discovery Run ===")
    print(f"Tokens provided: {len(args.tokens)}")
    print(f"Output: {out_path}")
    print()

    # Source 1: Priority list (always)
    print("[1/4] Priority Hand-Curated List")
    add_priority_repos(repo_set)

    # Source 2: Awesome list
    if not args.no_awesome:
        print("\n[2/4] Awesome-React-Components")
        scrape_awesome_list(repo_set)

    # Source 3: Libraries.io
    if not args.no_libraries:
        print("\n[3/4] Libraries.io")
        scrape_libraries_io(repo_set, api_key=args.libraries_key)

    # Source 4: GitHub Search API
    if not args.no_github_search:
        print("\n[4/4] GitHub Search API")
        github_search(args.tokens, repo_set)

    # Deduplicate and write
    final_list = deduplicate(repo_set)
    out_path.write_text("\n".join(final_list) + "\n", encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"  Total unique repos discovered: {len(final_list)}")
    print(f"  Written to: {out_path}")
    print(f"\nNext step:")
    print(f"  python batch_scraper_v2.py {out_path} data/master2.csv --workers 4")


if __name__ == "__main__":
    main()
