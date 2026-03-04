"""
weight_grid_search.py — Find optimal text/graph weight split
==============================================================
Tests all w_t/w_g combinations from 0.0 to 1.0 in 0.1 steps
using the structural benchmark v2 framework.

Run from project root:
    python analytics/weight_grid_search.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ─── Load data ─────────────────────────────────────────────
df = pd.read_pickle('data/vectors_reference.pkl')
df = df[df['component'].str.strip().ne('')].reset_index(drop=True)
graph_embeddings = np.load('data/graph_embeddings.npy')

# Load RAW magnitudes (pre-normalization) — these capture structural density
import os
if os.path.exists('data/graph_mags.npy'):
    graph_mags = np.load('data/graph_mags.npy')
else:
    graph_mags = np.linalg.norm(graph_embeddings, axis=1)

encoder = SentenceTransformer('all-MiniLM-L6-v2')
lex_feats = encoder.encode(df['combined_context'].tolist(), show_progress_bar=False, batch_size=128).astype('float32')

print(f"Data: {len(df)} components | Graph mags: min={graph_mags.min():.3f} max={graph_mags.max():.3f}")

# ─── Structural ground truth (same as benchmark_v2) ───────
def rel_complex_provider(row):
    s = 0
    if row['hooks_total'] >= 5: s += 1
    if row['hooks_total'] >= 8: s += 1
    if row.get('useContext', 0) > 0 or row.get('useReducer', 0) > 0: s += 1
    return min(s, 3)

def rel_data_fetching(row):
    s = 0
    if row.get('has_fetch', 0) == 1: s += 2
    if row.get('useState', 0) > 0:   s += 1
    if row.get('useEffect', 0) > 0:  s += 1
    return min(s, 3)

def rel_form_component(row):
    s = 0
    if row.get('event_handlers', 0) >= 2: s += 1
    if row.get('props', 0) >= 4:          s += 1
    if row.get('useState', 0) > 0:        s += 1
    if row.get('bool_props', 0) > 0:      s += 1
    return min(s, 3)

def rel_pure_presentational(row):
    s = 0
    if row['hooks_total'] == 0:        s += 2
    if row.get('has_fetch', 0) == 0:   s += 1
    if row.get('jsx_depth', 0) <= 4:   s += 1
    return min(s, 3)

def rel_animated(row):
    s = 0
    if row.get('useEffect', 0) > 0:    s += 1
    if row.get('useRef', 0) > 0:       s += 1
    if row.get('conditionals', 0) >= 3: s += 1
    if row.get('useCallback', 0) > 0:  s += 1
    return min(s, 3)

def rel_virtualized(row):
    s = 0
    if row.get('map_calls', 0) >= 2:   s += 1
    if row.get('useMemo', 0) > 0 or row.get('useCallback', 0) > 0: s += 1
    if row.get('loc', 0) >= 100:       s += 1
    if row.get('jsx_depth', 0) >= 5:   s += 1
    return min(s, 3)

def rel_global_state(row):
    s = 0
    if row.get('useContext', 0) > 0:   s += 2
    if row.get('useReducer', 0) > 0:   s += 2
    return min(s, 3)

def rel_dashboard(row):
    s = 0
    if row.get('has_fetch', 0) == 1:   s += 1
    if row.get('jsx_elems', 0) >= 10:  s += 1
    if row['hooks_total'] >= 4:        s += 1
    if row.get('loc', 0) >= 150:       s += 1
    return min(s, 3)

QUERIES = [
    ("complex_provider", "A stateful complex provider managing global context state with hooks", rel_complex_provider),
    ("data_fetching", "A component that fetches remote data and manages loading state", rel_data_fetching),
    ("form_component", "An interactive form component with many input handlers and validation", rel_form_component),
    ("presentational", "A simple pure presentational display component with no state", rel_pure_presentational),
    ("animated", "An animated interactive component with transitions and DOM refs", rel_animated),
    ("virtualized", "A high performance virtualized list with memoization", rel_virtualized),
    ("global_state", "A global state manager using context and reducer pattern", rel_global_state),
    ("dashboard", "A complex data dashboard that fetches and displays many elements", rel_dashboard),
]

# Pre-compute ground truth
gt = np.zeros((len(QUERIES), len(df)), dtype=np.float32)
for qi, (_, _, fn) in enumerate(QUERIES):
    for ci, (_, row) in enumerate(df.iterrows()):
        gt[qi, ci] = fn(row)

# Pre-compute text distances for all queries
print("Pre-computing query embeddings...")
q_texts = [q[1] for q in QUERIES]
q_embeddings = encoder.encode(q_texts, show_progress_bar=False).astype('float32')
text_dists = cdist(q_embeddings, lex_feats, metric='cosine')  # (8, N)

# ─── NDCG calculation ─────────────────────────────────────
def ndcg_at_k(ranked, gt_rel, k=10):
    dcg = sum((2**gt_rel[ranked[i]] - 1) / np.log2(i + 2) for i in range(min(k, len(ranked))))
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted(gt_rel, reverse=True)[:k]))
    return (dcg / idcg) if idcg > 0 else 0.0

# ─── Grid search ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  WEIGHT GRID SEARCH: w_t × w_g × recall_k × reward_scale")
print("=" * 60)

max_mag = graph_mags.max() or 1.0
best = {"ndcg": 0, "w_t": 0, "w_g": 0, "recall_k": 0, "scale": 0}

# Test grid
w_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
recall_values = [50, 100, 200]
scale_values = [0.1, 0.2, 0.3, 0.5]

total_configs = len(w_values) * len(recall_values) * len(scale_values)
tested = 0

for recall_k in recall_values:
    for scale in scale_values:
        for w_t in w_values:
            w_g = 1.0 - w_t
            ndcg_scores = []

            for qi in range(len(QUERIES)):
                # Recall pass
                recall_idx = np.argsort(text_dists[qi])[:recall_k]

                # Re-rank
                scores = np.full(len(df), np.inf)
                for idx in recall_idx:
                    struct_reward = graph_mags[idx] / max_mag
                    scores[idx] = (text_dists[qi][idx] * w_t) - (struct_reward * w_g * scale)

                ranked = np.argsort(scores)
                n = ndcg_at_k(ranked, gt[qi], k=10)
                ndcg_scores.append(n)

            mean_ndcg = np.mean(ndcg_scores)
            tested += 1

            if mean_ndcg > best['ndcg']:
                best = {"ndcg": mean_ndcg, "w_t": w_t, "w_g": w_g,
                        "recall_k": recall_k, "scale": scale}

            if tested % 33 == 0:
                print(f"  [{tested:3d}/{total_configs}] w_t={w_t:.1f} w_g={w_g:.1f} "
                      f"recall={recall_k} scale={scale:.1f} → NDCG={mean_ndcg:.4f}")

# Also test text-only and keyword baselines
# Text-only
text_ndcg = []
for qi in range(len(QUERIES)):
    ranked = np.argsort(text_dists[qi])
    text_ndcg.append(ndcg_at_k(ranked, gt[qi], k=10))

# Keyword
kw_ndcg = []
for qi in range(len(QUERIES)):
    query_terms = QUERIES[qi][1].lower().split()
    kw_scores = np.array([
        sum(1 for t in query_terms if t in str(df.iloc[i]['component']).lower())
        for i in range(len(df))
    ])
    ranked = np.argsort(-kw_scores)
    kw_ndcg.append(ndcg_at_k(ranked, gt[qi], k=10))

print(f"\n{'='*60}")
print(f"  GRID SEARCH RESULTS")
print(f"{'='*60}")
print(f"  {'Model':<35} {'NDCG@10':>8}")
print(f"  {'-'*45}")
print(f"  {'Keyword (baseline)':<35} {np.mean(kw_ndcg):>8.4f}")
print(f"  {'Text-only (MiniLM)':<35} {np.mean(text_ndcg):>8.4f}")
print(f"  {'Best Omnimodal config':<35} {best['ndcg']:>8.4f}")
print(f"\n  Optimal: w_t={best['w_t']:.1f}  w_g={best['w_g']:.1f}  "
      f"recall_k={best['recall_k']}  scale={best['scale']:.1f}")

gain_vs_text = ((best['ndcg'] - np.mean(text_ndcg)) / np.mean(text_ndcg) * 100)
gain_vs_kw   = ((best['ndcg'] - np.mean(kw_ndcg))   / np.mean(kw_ndcg)   * 100)
print(f"\n  Omnimodal gain vs Text-only:  {gain_vs_text:+.1f}%")
print(f"  Omnimodal gain vs Keyword:   {gain_vs_kw:+.1f}%")
print()
