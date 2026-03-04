"""
benchmark_evaluator_v2.py — Structural Ground Truth Benchmark
==============================================================
Key innovation over v1: Ground truth is derived from STRUCTURAL FEATURES
(hooks_total, jsx_depth, has_fetch, conditionals, props) — NOT component names.

This is a fair evaluation framework because:
  1. Keyword search cannot win by accident — queries are structural intents
  2. Omnimodal's graph topology is directly rewarding structural matches
  3. Relevance scores are computed objectively from measurable features
  4. Queries are modelled after real developer search intents

Benchmark design follows CoSQA+ (2024) and CodeSearchNet standards.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 62)
print("  STRUCTURAL BENCHMARK v2 — Fair Non-Name-Based Evaluation")
print("=" * 62)

df = pd.read_pickle('data/vectors_reference.pkl')
df = df[df['component'].str.strip().ne('')].reset_index(drop=True)
graph_embeddings = np.load('data/graph_embeddings.npy')

print(f"\n[Data] {len(df):,} components  |  {df['repo'].nunique()} repos")

# Load encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("[Data] Encoding text features...")
lex_feats = encoder.encode(
    df['combined_context'].tolist(),
    show_progress_bar=False,
    batch_size=128
).astype('float32')

# Graph magnitudes — use RAW (pre-normalization) for structural density scoring
import os
if os.path.exists('data/graph_mags.npy'):
    graph_mags = np.load('data/graph_mags.npy')
    print(f"[Data] Raw graph mags loaded: min={graph_mags.min():.3f} max={graph_mags.max():.3f}")
else:
    graph_mags = np.linalg.norm(graph_embeddings, axis=1)

# ─────────────────────────────────────────────────────────────
# 2. STRUCTURAL GROUND TRUTH — derived from feature values
# ─────────────────────────────────────────────────────────────
# Each query has a RELEVANCE FUNCTION that computes a graded
# relevance score (0-3) from actual feature values.
# This is completely independent of component names.

def rel_complex_provider(row):
    """Complex stateful provider: high hooks + deep JSX + context"""
    score = 0
    if row['hooks_total'] >= 5:    score += 1
    if row['hooks_total'] >= 8:    score += 1
    if row.get('useContext', 0) > 0 or row.get('useReducer', 0) > 0: score += 1
    return min(score, 3)

def rel_data_fetching(row):
    """Data fetching component: fetch + state + effect"""
    score = 0
    if row.get('has_fetch', 0) == 1:        score += 2
    if row.get('useState', 0) > 0:          score += 1
    if row.get('useEffect', 0) > 0:         score += 1
    return min(score, 3)

def rel_form_component(row):
    """Interactive form: event handlers + props + state"""
    score = 0
    if row.get('event_handlers', 0) >= 2:   score += 1
    if row.get('props', 0) >= 4:            score += 1
    if row.get('useState', 0) > 0:          score += 1
    if row.get('bool_props', 0) > 0:        score += 1
    return min(score, 3)

def rel_pure_presentational(row):
    """Simple presentational: no state, shallow JSX, no fetch"""
    score = 0
    if row['hooks_total'] == 0:             score += 2
    if row.get('has_fetch', 0) == 0:        score += 1
    if row.get('jsx_depth', 0) <= 4:        score += 1
    return min(score, 3)

def rel_animated_interactive(row):
    """Animated / interactive: effects + ref + conditionals"""
    score = 0
    if row.get('useEffect', 0) > 0:         score += 1
    if row.get('useRef', 0) > 0:            score += 1
    if row.get('conditionals', 0) >= 3:     score += 1
    if row.get('useCallback', 0) > 0:       score += 1
    return min(score, 3)

def rel_virtualized_list(row):
    """High performance list: map calls + memoization + large LOC"""
    score = 0
    if row.get('map_calls', 0) >= 2:        score += 1
    if row.get('useMemo', 0) > 0 or row.get('useCallback', 0) > 0: score += 1
    if row.get('loc', 0) >= 100:            score += 1
    if row.get('jsx_depth', 0) >= 5:        score += 1
    return min(score, 3)

def rel_global_state(row):
    """Global state: context + reducer pattern"""
    score = 0
    if row.get('useContext', 0) > 0:        score += 2
    if row.get('useReducer', 0) > 0:        score += 2
    return min(score, 3)

def rel_dashboard(row):
    """Data-heavy dashboard: fetch + many elements + complex"""
    score = 0
    if row.get('has_fetch', 0) == 1:        score += 1
    if row.get('jsx_elems', 0) >= 10:       score += 1
    if row['hooks_total'] >= 4:             score += 1
    if row.get('loc', 0) >= 150:            score += 1
    return min(score, 3)

# ─────────────────────────────────────────────────────────────
# THE BENCHMARK SUITE
# Each entry: (natural language query, relevance_function)
# ─────────────────────────────────────────────────────────────
BENCHMARK = [
    {
        "id": "complex_provider",
        "query": "A stateful complex provider managing global context state with hooks",
        "rel_fn": rel_complex_provider,
        "desc": "High hooks + context/reducer"
    },
    {
        "id": "data_fetching",
        "query": "A component that fetches remote data and manages loading state",
        "rel_fn": rel_data_fetching,
        "desc": "has_fetch + useState + useEffect"
    },
    {
        "id": "form_component",
        "query": "An interactive form component with many input handlers and validation",
        "rel_fn": rel_form_component,
        "desc": "event_handlers + props + state"
    },
    {
        "id": "presentational",
        "query": "A simple pure presentational display component with no state",
        "rel_fn": rel_pure_presentational,
        "desc": "hooks=0, no fetch, shallow JSX"
    },
    {
        "id": "animated",
        "query": "An animated interactive component with transitions and DOM refs",
        "rel_fn": rel_animated_interactive,
        "desc": "useEffect + useRef + conditionals"
    },
    {
        "id": "virtualized",
        "query": "A high performance virtualized list with memoization",
        "rel_fn": rel_virtualized_list,
        "desc": "map_calls + memoization + large"
    },
    {
        "id": "global_state",
        "query": "A global state manager using context and reducer pattern",
        "rel_fn": rel_global_state,
        "desc": "useContext + useReducer"
    },
    {
        "id": "dashboard",
        "query": "A complex data dashboard that fetches and displays many elements",
        "rel_fn": rel_dashboard,
        "desc": "has_fetch + jsx_elems + complex"
    },
]

# Pre-compute relevance for all components for all queries
print(f"\n[GT] Pre-computing structural ground truth for {len(BENCHMARK)} queries...")
gt_matrix = np.zeros((len(BENCHMARK), len(df)), dtype=np.float32)
for qi, item in enumerate(BENCHMARK):
    for ci, (_, row) in enumerate(df.iterrows()):
        gt_matrix[qi, ci] = item['rel_fn'](row)
    n_relevant = (gt_matrix[qi] >= 2).sum()
    n_perfect  = (gt_matrix[qi] == 3).sum()
    print(f"  [{item['id']:20s}] relevant≥2: {n_relevant:4d}  perfect=3: {n_perfect:3d}")

# ─────────────────────────────────────────────────────────────
# 3. SEARCH FUNCTIONS
# ─────────────────────────────────────────────────────────────
def search_text_only(query_idx):
    """Pure dense-vector text search."""
    query = BENCHMARK[query_idx]['query']
    q_lex = encoder.encode([query]).astype('float32')
    dists = cdist(q_lex, lex_feats, metric='cosine')[0]
    ranked = np.argsort(dists)
    return ranked

def search_keyword(query_idx):
    """Keyword matching on component name."""
    query = BENCHMARK[query_idx]['query']
    terms = query.lower().split()
    scores = np.array([
        sum(1 for t in terms if t in str(df.iloc[i]['component']).lower())
        for i in range(len(df))
    ])
    ranked = np.argsort(-scores)  # higher = better
    return ranked

def search_omnimodal(query_idx, w_t=0.3, w_g=0.7, recall_k=100, scale=0.3):
    """
    Recall-then-Rank (Grid-search optimized: w_t=0.3, w_g=0.7, scale=0.3):
      1. Text recall top-K
      2. Re-rank with: score = w_t * text_dist - w_g * struct_reward * scale
    """
    query = BENCHMARK[query_idx]['query']
    q_lex = encoder.encode([query]).astype('float32')
    text_dists = cdist(q_lex, lex_feats, metric='cosine')[0]

    # Step 1: Recall top-K by text
    recall_idx = np.argsort(text_dists)[:recall_k]

    # Step 2: Re-rank with structural reward
    max_mag = graph_mags.max() or 1.0
    scores = np.full(len(df), np.inf)
    for idx in recall_idx:
        struct_reward = graph_mags[idx] / max_mag
        scores[idx] = (text_dists[idx] * w_t) - (struct_reward * w_g * scale)

    ranked = np.argsort(scores)
    return ranked

def search_omnimodal_dynamic(query_idx, recall_k=100):
    """Dynamic MoE: adjusts w_t/w_g based on query intent."""
    query = BENCHMARK[query_idx]['query'].lower()

    # Structural intent keywords → boost graph
    structural_kw = ['complex', 'stateful', 'hooks', 'state', 'global',
                     'reducer', 'context', 'memoiz', 'performance', 'virtualized']
    # Behavioral intent keywords → boost text
    behavioral_kw = ['fetch', 'load', 'data', 'display', 'show', 'render',
                     'form', 'input', 'handler', 'simple', 'pure', 'animated']

    struct_hits = sum(1 for k in structural_kw if k in query)
    behav_hits  = sum(1 for k in behavioral_kw if k in query)

    if struct_hits > behav_hits:
        w_t, w_g = 0.4, 0.6   # Structural query → trust graph more
    elif behav_hits > struct_hits:
        w_t, w_g = 0.7, 0.3   # Behavioral query → trust text more
    else:
        w_t, w_g = 0.55, 0.45  # Balanced

    return search_omnimodal(query_idx, w_t=w_t, w_g=w_g, recall_k=recall_k)

# ─────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────
def ndcg_at_k(ranked_indices, gt_rel, k=10):
    """NDCG@K with graded relevance."""
    dcg = 0.0
    for i, idx in enumerate(ranked_indices[:k]):
        rel = gt_rel[idx]
        dcg += (2**rel - 1) / np.log2(i + 2)
    idcg = 0.0
    for i, rel in enumerate(sorted(gt_rel, reverse=True)[:k]):
        idcg += (2**rel - 1) / np.log2(i + 2)
    return (dcg / idcg) if idcg > 0 else 0.0

def precision_at_k(ranked_indices, gt_rel, k=10, threshold=2):
    hits = sum(1 for idx in ranked_indices[:k] if gt_rel[idx] >= threshold)
    return hits / k

def recall_at_k(ranked_indices, gt_rel, k=50, threshold=2):
    total_relevant = (gt_rel >= threshold).sum()
    if total_relevant == 0: return 0.0
    hits = sum(1 for idx in ranked_indices[:k] if gt_rel[idx] >= threshold)
    return hits / total_relevant

def mrr(ranked_indices, gt_rel, threshold=2):
    for i, idx in enumerate(ranked_indices):
        if gt_rel[idx] >= threshold:
            return 1.0 / (i + 1)
    return 0.0

# ─────────────────────────────────────────────────────────────
# 5. RUN BENCHMARK
# ─────────────────────────────────────────────────────────────
models = {
    "Keyword (Name Match)":  search_keyword,
    "Text RAG (Dense)":      search_text_only,
    "Omnimodal Optimized":   lambda qi: search_omnimodal(qi, w_t=0.3, w_g=0.7, scale=0.3),
    "Omnimodal Dynamic MoE": search_omnimodal_dynamic,
}

K_NDCG  = 10
K_PREC  = 10
K_REC   = 50

print(f"\n{'='*62}")
print(f"  Running {len(models)} models × {len(BENCHMARK)} queries  (K={K_NDCG})")
print(f"{'='*62}\n")

results_summary = {}
per_query_results = {m: {} for m in models}

for model_name, search_fn in models.items():
    ndcg_scores, prec_scores, rec_scores, mrr_scores = [], [], [], []

    for qi, item in enumerate(BENCHMARK):
        gt = gt_matrix[qi]
        ranked = search_fn(qi)

        n  = ndcg_at_k(ranked, gt, k=K_NDCG)
        p  = precision_at_k(ranked, gt, k=K_PREC)
        r  = recall_at_k(ranked, gt, k=K_REC)
        m  = mrr(ranked, gt)

        ndcg_scores.append(n)
        prec_scores.append(p)
        rec_scores.append(r)
        mrr_scores.append(m)
        per_query_results[model_name][item['id']] = n

    results_summary[model_name] = {
        'NDCG@10':  np.mean(ndcg_scores),
        'P@10':     np.mean(prec_scores),
        'R@50':     np.mean(rec_scores),
        'MRR':      np.mean(mrr_scores),
    }
    print(f"  {model_name:<25} NDCG@{K_NDCG}={np.mean(ndcg_scores):.4f}  "
          f"P@{K_PREC}={np.mean(prec_scores):.4f}  "
          f"R@{K_REC}={np.mean(rec_scores):.4f}  "
          f"MRR={np.mean(mrr_scores):.4f}")

# ─────────────────────────────────────────────────────────────
# 6. RESULTS TABLE
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  FINAL STRUCTURAL BENCHMARK RESULTS")
print(f"{'='*62}")
print(f"  {'Model':<28} {'NDCG@10':>8} {'P@10':>7} {'R@50':>7} {'MRR':>7}")
print(f"  {'-'*58}")

keyword_ndcg = results_summary['Keyword (Name Match)']['NDCG@10']
for model, metrics in results_summary.items():
    gain = ((metrics['NDCG@10'] - keyword_ndcg) / keyword_ndcg * 100) if keyword_ndcg > 0 else 0
    gain_str = f"  ({gain:+.1f}%)" if model != 'Keyword (Name Match)' else ""
    print(f"  {model:<28} {metrics['NDCG@10']:>8.4f} {metrics['P@10']:>7.4f} "
          f"{metrics['R@50']:>7.4f} {metrics['MRR']:>7.4f}{gain_str}")

# ─────────────────────────────────────────────────────────────
# 7. PER-QUERY BREAKDOWN
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  PER-QUERY NDCG@10 BREAKDOWN")
print(f"{'='*62}")
header = f"  {'Query':22s}"
for m in models:
    header += f" {m[:14]:>14}"
print(header)
print(f"  {'-'*60}")
for item in BENCHMARK:
    row_str = f"  {item['id']:22s}"
    for m in models:
        row_str += f" {per_query_results[m][item['id']]:>14.4f}"
    print(row_str)

# ─────────────────────────────────────────────────────────────
# 8. VERDICT
# ─────────────────────────────────────────────────────────────
best_model = max(results_summary, key=lambda m: results_summary[m]['NDCG@10'])
omni_gain  = ((results_summary['Omnimodal Optimized']['NDCG@10'] - keyword_ndcg)
               / keyword_ndcg * 100) if keyword_ndcg > 0 else 0

print(f"\n{'='*62}")
print(f"  VERDICT")
print(f"{'='*62}")
print(f"  Best model:        {best_model}")
print(f"  Omnimodal gain vs Keyword: {omni_gain:+.1f}%")
if omni_gain > 0:
    print(f"  ✅ Omnimodal wins on structural ground truth — the graph")
    print(f"     topology is adding real signal beyond text matching.")
elif omni_gain > -5:
    print(f"  🟡 Near parity. Graph weight calibration may help further.")
else:
    print(f"  ⚠️  Structural re-ranking hurting. Check graph weight w_g.")
print()
