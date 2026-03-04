"""
Phase 0 Analysis: Validate structural signal quality in master.csv
Answers the key questions before moving to embeddings.

Works with both old master.csv (no filter_calls/has_fetch/num_imports)
and new CSVs generated after the structural_poc.py fix.
"""

import csv
import sys
import os
from collections import Counter

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\karth\OneDrive\Desktop\ml_on_code\Codsoft-tasks\master2.csv"
if not os.path.exists(CSV_PATH):
    fallback = r"c:\Users\karth\OneDrive\Desktop\ml_on_code\Codsoft-tasks\master.csv"
    if os.path.exists(fallback):
        print(f"[INFO] {CSV_PATH} not found — using {fallback}")
        CSV_PATH = fallback
    else:
        print(f"[ERROR] CSV not found: {CSV_PATH}"); sys.exit(1)
print(f"Analysing: {CSV_PATH}")

# ─── Load CSV ────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

total = len(rows)
print(f"\n{'='*60}")
print(f"  MASTER.CSV ANALYSIS REPORT")
print(f"{'='*60}")
print(f"Total rows: {total}")

# ─── Helper: safe int ────────────────────────────────────────────────────────
def ival(row, col):
    try:
        return int(row.get(col, 0) or 0)
    except ValueError:
        return 0

# ─── 1. TRIVIAL COMPONENT RATE ───────────────────────────────────────────────
print(f"\n{'─'*60}")
print("1. TRIVIAL vs NON-TRIVIAL COMPONENTS")
print("   (trivial = hooks=0 AND props<=1 AND jsx_depth<=3)")

trivial = [r for r in rows if ival(r,'hooks_total')==0 and ival(r,'props')<=1 and ival(r,'jsx_depth')<=3]
non_trivial = [r for r in rows if not (ival(r,'hooks_total')==0 and ival(r,'props')<=1 and ival(r,'jsx_depth')<=3)]

print(f"   Trivial components    : {len(trivial):>6}  ({100*len(trivial)/total:.1f}%)")
print(f"   Non-trivial           : {len(non_trivial):>6}  ({100*len(non_trivial)/total:.1f}%)")

# ─── 2. DISTRIBUTION STATS ───────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("2. FEATURE DISTRIBUTION (non-trivial only)")

metrics = ["hooks_total", "props", "jsx_depth", "jsx_elems",
           "event_handlers", "conditionals", "map_calls",
           "filter_calls", "reduce_calls", "has_fetch", "num_imports"]

for col in metrics:
    vals = [ival(r, col) for r in non_trivial]
    if not vals:
        continue
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / len(vals)
    sorted_v = sorted(vals)
    med = sorted_v[len(sorted_v)//2]
    zeros = sum(1 for v in vals if v == 0)
    print(f"   {col:<18} min={mn:>4}  max={mx:>5}  mean={mean:>6.1f}  median={med:>3}  zeros={100*zeros/len(vals):>5.1f}%")

# ─── 3. ZERO-HOOKS RATE ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("3. HOOK USAGE BREAKDOWN (all rows)")

hook_cols = ["useState","useEffect","useCallback","useMemo","useContext","useReducer","useRef","useCustom"]
for col in hook_cols:
    nonzero = sum(1 for r in rows if ival(r,col) > 0)
    print(f"   {col:<15}: {nonzero:>5} rows have it  ({100*nonzero/total:.1f}%)")

# ─── 4. COMMENT COVERAGE ─────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("4. LEXICAL COVERAGE (comments / prop names)")

has_comment  = sum(1 for r in rows if r.get("comment","").strip())
has_propname = sum(1 for r in rows if r.get("prop_names","").strip())
has_name     = sum(1 for r in rows if r.get("component","").strip())

print(f"   Has JSDoc comment : {has_comment:>6}  ({100*has_comment/total:.1f}%)")
print(f"   Has prop names    : {has_propname:>6}  ({100*has_propname/total:.1f}%)")
print(f"   Has component name: {has_name:>6}  ({100*has_name/total:.1f}%)")

# ─── 5. DIFFERENTIATION CHECK ────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("5. STRUCTURAL DIFFERENTIATION — do vectors separate component types?")

# Name-based grouping (rough heuristic)
groups = {
    "Modal/Dialog"  : ["modal","dialog","overlay","popup","drawer"],
    "Form/Input"    : ["form","input","field","textbox","checkbox","radio","select","dropdown"],
    "Button"        : ["button","btn","submit","cta"],
    "Navigation"    : ["nav","navbar","menu","sidebar","breadcrumb","tabs","tab"],
    "Table/List"    : ["table","list","grid","row","cell","column"],
    "Card/Layout"   : ["card","panel","container","wrapper","layout","header","footer"],
}

for group_name, keywords in groups.items():
    matched = [r for r in rows if any(kw in r.get("component","").lower() or kw in r.get("file","").lower()
                                       for kw in keywords)]
    if not matched:
        print(f"   {group_name:<18}: 0 found")
        continue
    avg_hooks = sum(ival(r,"hooks_total") for r in matched) / len(matched)
    avg_props = sum(ival(r,"props") for r in matched) / len(matched)
    avg_depth = sum(ival(r,"jsx_depth") for r in matched) / len(matched)
    print(f"   {group_name:<18}: n={len(matched):>4}  hooks={avg_hooks:>5.1f}  props={avg_props:>5.1f}  depth={avg_depth:>5.1f}")

# ─── 6. NAIVE QUERY TEST ─────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("6. NAIVE STRUCTURAL QUERIES (does filtering return sane results?)")

queries = [
    ("Complex interactive (hooks>=3, props>=3, depth>=4)",
     lambda r: ival(r,"hooks_total")>=3 and ival(r,"props")>=3 and ival(r,"jsx_depth")>=4),
    ("Form-like (event_handlers>=1, props>=2)",
     lambda r: ival(r,"event_handlers")>=1 and ival(r,"props")>=2),
    ("Stateful + conditional (useState>=1, conditionals>=1)",
     lambda r: ival(r,"useState")>=1 and ival(r,"conditionals")>=1),
    ("Deep tree, no hooks (depth>=6, hooks=0)",
     lambda r: ival(r,"jsx_depth")>=6 and ival(r,"hooks_total")==0),
    ("Data mapper (map_calls>=1, props>=1)",
     lambda r: ival(r,"map_calls")>=1 and ival(r,"props")>=1),
]

for label, fn in queries:
    matched = [r for r in rows if fn(r)]
    print(f"\n   Query: {label}")
    print(f"   Matches: {len(matched)}")
    # print top 5 examples
    for r in matched[:5]:
        name = r.get("component","?") or "?"
        fname = r.get("file","?").split("/")[-1]
        h = ival(r,"hooks_total"); p = ival(r,"props"); d = ival(r,"jsx_depth")
        print(f"      {name:<30} [{fname}]  hooks={h} props={p} depth={d}")

# ─── 7. DUPLICATE / NEAR-DUPLICATE DETECTION ─────────────────────────────────
print(f"\n{'─'*60}")
print("7. STRUCTURAL VECTOR DUPLICATES (identical hook+prop+depth tuple)")

vec_counts = Counter()
for r in non_trivial:
    vec = (ival(r,"hooks_total"), ival(r,"props"), ival(r,"jsx_depth"))
    vec_counts[vec] += 1

dup_vecs = {k:v for k,v in vec_counts.items() if v > 10}
print(f"   Unique vectors in non-trivial set : {len(vec_counts)}")
print(f"   Vectors appearing >10 times (hot spots):")
for vec, cnt in sorted(dup_vecs.items(), key=lambda x: -x[1])[:10]:
    print(f"      hooks={vec[0]} props={vec[1]} depth={vec[2]}  → {cnt} components")

# ─── 8. SUMMARY VERDICT ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY VERDICT")
print(f"{'='*60}")

trivial_pct = 100*len(trivial)/total
comment_pct = 100*has_comment/total

if trivial_pct > 60:
    print(f"  [WARN] {trivial_pct:.0f}% trivial components — filter needed before embeddings")
else:
    print(f"  [OK]   Only {trivial_pct:.0f}% trivial — dataset has meaningful signal")

if comment_pct < 20:
    print(f"  [WARN] Only {comment_pct:.0f}% have JSDoc — rely more on structural features")
else:
    print(f"  [OK]   {comment_pct:.0f}% have comments — lexical integration viable")

hook_users = sum(1 for r in rows if ival(r,"hooks_total") > 0)
if hook_users / total < 0.3:
    print(f"  [WARN] Only {100*hook_users/total:.0f}% use hooks — many non-interactive components in dataset")
else:
    print(f"  [OK]   {100*hook_users/total:.0f}% use hooks — good hook signal coverage")

print(f"\n  Total non-trivial components available for ML: {len(non_trivial)}")
print(f"  Recommendation: filter trivial, keep non_trivial set for Phase 1 embeddings")
print(f"{'='*60}\n")
