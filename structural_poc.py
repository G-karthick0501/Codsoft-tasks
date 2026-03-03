#!/usr/bin/env python3
"""
Quick PoC for Phase‑0: walk a React codebase and extract simple
structural features using tree‑sitter.

Outputs a CSV with columns:
    repo, file, hooks, props, jsx_depth
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from tree_sitter import Language, Parser

# Instead of building grammars ourselves, rely on language-specific
# packages which expose precompiled parsers. These modules return a
# PyCapsule that can be wrapped by `tree_sitter.Language`.
import tree_sitter_javascript
import tree_sitter_typescript


def ensure_language():
    # convert the capsule functions into Language objects
    js_caps = tree_sitter_javascript.language()
    tsx_caps = tree_sitter_typescript.language_tsx()
    ts_caps = tree_sitter_typescript.language_typescript()
    # wrap in Language
    js_lang = Language(js_caps)
    ts_lang = Language(tsx_caps)
    ts_lang_alt = Language(ts_caps)
    # we return both variants in case we need plain ts files vs tsx
    return js_lang, ts_lang, ts_lang_alt

def parse_file(path, js_lang, tsx_lang, ts_lang):
    code = path.read_bytes()
    parser = Parser()
    # choose appropriate language
    if path.suffix in (".ts", ".tsx"):
        parser.language = tsx_lang if path.suffix == ".tsx" else ts_lang
    else:
        parser.language = js_lang
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    return tree

# Recognize any identifier starting with "use" followed by an upper-case letter,
# which covers all standard hooks (useState, useEffect, …) and most custom hooks.
# We'll still keep a small set for explicit checking if needed, but
# the regex below is the primary signal.
# known hooks we want to count individually; others count as custom
HOOK_NAMES = [
    "useState",
    "useEffect",
    "useCallback",
    "useMemo",
    "useContext",
    "useReducer",
]

import re
HOOK_PATTERN = re.compile(r"^use[A-Z].*")

def count_hooks(node, source):
    """Recursively count hook identifiers by type.

    Returns a dict with keys from HOOK_NAMES plus 'custom' and 'total'.
    """
    counts = {name: 0 for name in HOOK_NAMES}
    counts["custom"] = 0
    counts["total"] = 0

    def recurse(n):
        if n.type == "identifier":
            name = source[n.start_byte:n.end_byte].decode()
            if HOOK_PATTERN.match(name):
                counts["total"] += 1
                if name in HOOK_NAMES:
                    counts[name] += 1
                else:
                    counts["custom"] += 1
        for c in n.children:
            recurse(c)

    recurse(node)
    return counts

def count_props(node):
    """
    Look for function parameters (arrow or function declaration)
    and count identifiers / object patterns in the first param.
    """
    if node.type in ("function_declaration", "arrow_function", "function"):
        params = [c for c in node.children if c.type == "formal_parameters"]
        if params:
            p = params[0]
            # simple heuristic: count commas + identifiers
            text = p.text.decode()
            return text.count(",") + (1 if text.strip() else 0)
    total = 0
    for c in node.children:
        total += count_props(c)
    return total

def jsx_max_depth(node):
    """Return max nesting depth of JSX elements under this node."""
    if node.type in ("jsx_element", "jsx_self_closing_element"):
        depths = [jsx_max_depth(c) for c in node.children]
        return 1 + (max(depths) if depths else 0)
    else:
        return max((jsx_max_depth(c) for c in node.children), default=0)

def extract_lexical(src_bytes: bytes):
    """Quick regex-based heuristics to pull lexical features out of source.

    Returns (component_name, prop_list, comment_text).
    """
    text = src_bytes.decode('utf8', errors='ignore')
    lines = text.splitlines()

    # try to find preceding comment block
    comment = ''
    for i, line in enumerate(lines):
        if re.match(r"^\s*(export\s+)?(function|const|let|var)", line):
            # gather comment lines immediately above
            j = i - 1
            comments = []
            while j >= 0 and lines[j].strip().startswith('//'):
                comments.insert(0, lines[j].strip()[2:].strip())
                j -= 1
            comment = '\n'.join(comments)
            break

    name = None
    props = []
    # regex patterns to detect typical component signatures
    patterns = [
        r"export\s+default\s+function\s+(\w+)\s*\(([^)]*)\)",
        r"function\s+(\w+)\s*\(([^)]*)\)",
        r"const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>",
        r"const\s+(\w+)\s*=\s*([^=]+)\s*=>"  # catch destructuring outside parens
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group(1)
            params = m.group(2) if m.lastindex >= 2 else ''
            # split props by comma and strip braces
            params = params.strip()
            if params.startswith('{') and params.endswith('}'):
                params = params[1:-1]
            props = [p.strip().split('=')[0].strip() for p in params.split(',') if p.strip()]
            break
    return name, props, comment


def count_jsx_elements(node):
    cnt = 0
    if node.type in ("jsx_element", "jsx_self_closing_element"):
        cnt += 1
    for c in node.children:
        cnt += count_jsx_elements(c)
    return cnt


def count_conditionals(node):
    cnt = 1 if node.type == "conditional_expression" else 0
    for c in node.children:
        cnt += count_conditionals(c)
    return cnt


def count_map_calls(node):
    cnt = 0
    if node.type == "call_expression":
        # look for 'map' identifier in function child
        func = node.child_by_field_name("function")
        if func:
            # traverse func subtree looking for identifier 'map'
            stack = [func]
            while stack:
                cur = stack.pop()
                if cur.type == "identifier":
                    name = cur.text.decode()
                    if name == "map":
                        cnt += 1
                        break
                stack.extend(cur.children)
    for c in node.children:
        cnt += count_map_calls(c)
    return cnt


def analyze(path, js_lang, tsx_lang, ts_lang):
    tree = parse_file(path, js_lang, tsx_lang, ts_lang)
    if tree is None:
        return None
    src = path.read_bytes()
    root = tree.root_node
    hook_counts = count_hooks(root, src)
    props = count_props(root)
    depth = jsx_max_depth(root)
    jsxElems = count_jsx_elements(root)
    conds = count_conditionals(root)
    maps = count_map_calls(root)

    comp_name, prop_list, comment = extract_lexical(src)
    # prop-based features
    event_handlers = sum(1 for p in prop_list if p.startswith("on") and len(p) > 2 and p[2].isupper())
    bool_props = sum(1 for p in prop_list if p.startswith("is") or p.startswith("has") or p.startswith("show"))
    has_children = 1 if "children" in prop_list else 0

    return {
        "hooks": hook_counts,
        "props": props,
        "depth": depth,
        "jsx_elems": jsxElems,
        "conditionals": conds,
        "map_calls": maps,
        "component": comp_name,
        "prop_list": prop_list,
        "comment": comment,
        "event_handlers": event_handlers,
        "bool_props": bool_props,
        "has_children": has_children,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("root_dir")
    p.add_argument("--output", "-o", default="structural.csv")
    p.add_argument("--exclude", "-x", action="append", default=[],
                   help="glob-style path patterns to exclude (can be repeated)")
    p.add_argument("--max-components", type=int, default=0,
                   help="if >0, stop after this many components for the repo")
    args = p.parse_args()

    js_lang, tsx_lang, ts_lang = ensure_language()
    rows = []
    exclude_patterns = args.exclude
    max_comp = int(args.max_components or 0)

    def is_excluded(path: Path):
        s = str(path).replace('\\', '/')
        for pat in exclude_patterns:
            if pat in s:
                return True
            try:
                from fnmatch import fnmatch
                if fnmatch(s, pat):
                    return True
            except Exception:
                pass
        low = s.lower()
        tests = ["/test/", "/tests/", "__tests__", "/__fixtures__", "/fixtures/", "/examples/", "/example/", "/stories/"]
        if any(t in low for t in tests):
            return True
        return False

    for dirpath, _, filenames in os.walk(args.root_dir):
        for fn in filenames:
            if fn.endswith((".jsx", ".tsx", ".js", ".ts")):
                full = Path(dirpath) / fn
                if is_excluded(full):
                    continue
                res = analyze(full, js_lang, tsx_lang, ts_lang)
                if not res:
                    continue
                h = res["hooks"]
                props = res["props"]
                depth = res["depth"]
                # quality gates
                if depth <= 2:
                    continue
                if h["total"] == 0 and props == 0:
                    continue
                if props > 50:
                    props = 50
                rows.append([
                    args.root_dir, str(full),
                    h["total"], h.get("useState", 0), h.get("useEffect", 0), h.get("useCallback", 0),
                    h.get("useMemo", 0), h.get("useContext", 0), h.get("useReducer", 0), h.get("custom", 0),
                    props, depth,
                    res["jsx_elems"], res["conditionals"], res["map_calls"],
                    res["event_handlers"], res["bool_props"], res["has_children"],
                    res["component"] or "", ";".join(res["prop_list"]), res["comment"],
                ])
                if max_comp and len(rows) >= max_comp:
                    break
        if max_comp and len(rows) >= max_comp:
            break

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "repo", "file",
            "hooks_total", "useState", "useEffect", "useCallback",
            "useMemo", "useContext", "useReducer", "useCustom",
            "props", "jsx_depth",
            "jsx_elems", "conditionals", "map_calls",
            "event_handlers", "bool_props", "has_children",
            "component", "prop_names", "comment",
        ])
        writer.writerows(rows)
    print(f"wrote {len(rows)} records to {args.output}")

if __name__ == "__main__":
    main()