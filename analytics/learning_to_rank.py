"""
learning_to_rank.py — Query-Dependent Feature Weighting
========================================================
Uses AdaBoost's per-query feature importances as supervision signal
to learn: query_text → 36-dim feature weight vector.

Level 1: Template lookup (nearest training query)
Level 2: TF-IDF → Ridge regression → feature weights
Level 3: TF-IDF → MLP → feature weights

Compared against: AdaBoost, Weighted Feature, Keyword, Random Forest

Run: python analytics/learning_to_rank.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.spatial.distance import cdist
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# 1. DATA + FEATURES
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  LEARNING TO RANK — Query-Dependent Feature Weighting")
print("=" * 70)

df = pd.read_csv('data/master2.csv')
df = df[df['component'].str.strip().ne('')].drop_duplicates(subset=['repo','file']).reset_index(drop=True)

RAW = ['hooks_total','useState','useEffect','useCallback','useMemo','useContext',
       'useReducer','useRef','useCustom','props','jsx_depth','jsx_elems',
       'conditionals','map_calls','filter_calls','reduce_calls','has_fetch',
       'num_imports','event_handlers','bool_props','has_children','loc']
for c in RAW:
    if c not in df.columns: df[c] = 0
df[RAW] = df[RAW].fillna(0)

df['state_hooks']      = df['useState'] + df['useReducer']
df['effect_hooks']     = df['useEffect'] + df['useCallback'] + df['useMemo']
df['context_hooks']    = df['useContext'] + df['useReducer']
df['ref_hooks']        = df['useRef']
df['hook_diversity']   = (df[['useState','useEffect','useCallback','useMemo',
                              'useContext','useReducer','useRef','useCustom']] > 0).sum(axis=1)
df['complexity_score'] = (df['hooks_total']*2 + df['conditionals'] + df['map_calls'] +
                          df['filter_calls'] + df['event_handlers'] + df['has_fetch']*3)
df['interactivity']    = df['event_handlers'] + df['bool_props'] + df['has_children']
df['data_pattern']     = df['has_fetch'] + (df['useState']>0).astype(int) + (df['useEffect']>0).astype(int)
df['jsx_density']      = df['jsx_elems'] / df['loc'].clip(lower=1)
df['hooks_per_loc']    = df['hooks_total'] / df['loc'].clip(lower=1)
df['props_per_loc']    = df['props'] / df['loc'].clip(lower=1)
df['is_stateless']     = (df['hooks_total'] == 0).astype(int)
df['is_complex']       = (df['hooks_total'] >= 5).astype(int)
df['is_fetcher']       = df['has_fetch']

ENG = ['state_hooks','effect_hooks','context_hooks','ref_hooks','hook_diversity',
       'complexity_score','interactivity','data_pattern','jsx_density',
       'hooks_per_loc','props_per_loc','is_stateless','is_complex','is_fetcher']
FEATS = RAW + ENG

print(f"[DATA] {len(df):,} components | {df['repo'].nunique()} repos | {len(FEATS)} features")

# ═══════════════════════════════════════════════════════════════
# 2. QUERY BANK — 30 queries for robust training
# ═══════════════════════════════════════════════════════════════
def gt_fn(qid, row):
    rules = {
        "complex_provider":     lambda r: min(3, int(r['hooks_total']>=5)+int(r['hooks_total']>=8)+int(r.get('useContext',0)>0 or r.get('useReducer',0)>0)),
        "data_fetching":        lambda r: min(3, int(r.get('has_fetch',0)==1)*2+int(r.get('useState',0)>0)+int(r.get('useEffect',0)>0)),
        "form_component":       lambda r: min(3, int(r.get('event_handlers',0)>=2)+int(r.get('props',0)>=4)+int(r.get('useState',0)>0)+int(r.get('bool_props',0)>0)),
        "presentational":       lambda r: min(3, int(r['hooks_total']==0)*2+int(r.get('has_fetch',0)==0)+int(r.get('jsx_depth',0)<=4)),
        "animated":             lambda r: min(3, int(r.get('useEffect',0)>0)+int(r.get('useRef',0)>0)+int(r.get('conditionals',0)>=3)+int(r.get('useCallback',0)>0)),
        "virtualized":          lambda r: min(3, int(r.get('map_calls',0)>=2)+int(r.get('useMemo',0)>0 or r.get('useCallback',0)>0)+int(r.get('loc',0)>=100)+int(r.get('jsx_depth',0)>=5)),
        "global_state":         lambda r: min(3, int(r.get('useContext',0)>0)*2+int(r.get('useReducer',0)>0)*2),
        "dashboard":            lambda r: min(3, int(r.get('has_fetch',0)==1)+int(r.get('jsx_elems',0)>=10)+int(r['hooks_total']>=4)+int(r.get('loc',0)>=150)),
        "deep_no_hooks":        lambda r: min(3, int(r.get('jsx_depth',0)>=8)*2+int(r['hooks_total']==0)*2),
        "stateless_layout":     lambda r: min(3, int(r['hooks_total']==0)+int(r.get('jsx_elems',0)>=15)*2+int(r.get('props',0)>=3)),
        "hooks_shallow":        lambda r: min(3, int(r['hooks_total']>=5)*2+int(r.get('jsx_depth',0)<=4)*2),
        "tiny_pure":            lambda r: min(3, int(r.get('loc',0)<=40)+int(r['hooks_total']==0)+int(r.get('props',0)<=2)+int(r.get('has_fetch',0)==0)),
        # Extended query bank for training LTR
        "ref_heavy":            lambda r: min(3, int(r.get('useRef',0)>=2)*2+int(r.get('useEffect',0)>0)),
        "callback_memoized":    lambda r: min(3, int(r.get('useCallback',0)>=2)*2+int(r.get('useMemo',0)>0)),
        "high_import":          lambda r: min(3, int(r.get('num_imports',0)>=10)*2+int(r.get('loc',0)>=200)),
        "map_heavy":            lambda r: min(3, int(r.get('map_calls',0)>=3)*2+int(r.get('jsx_elems',0)>=10)),
        "conditional_complex":  lambda r: min(3, int(r.get('conditionals',0)>=5)*2+int(r['hooks_total']>=3)),
        "minimal_wrapper":      lambda r: min(3, int(r.get('loc',0)<=30)*2+int(r.get('has_children',0)==1)),
        "event_rich":           lambda r: min(3, int(r.get('event_handlers',0)>=3)*2+int(r.get('props',0)>=5)),
        "large_component":      lambda r: min(3, int(r.get('loc',0)>=300)*2+int(r['hooks_total']>=4)),
        "custom_hooks_heavy":   lambda r: min(3, int(r.get('useCustom',0)>=3)*2+int(r['hooks_total']>=5)),
        "filter_reduce":        lambda r: min(3, int(r.get('filter_calls',0)>=1)*2+int(r.get('reduce_calls',0)>=1)*2),
        "prop_heavy":           lambda r: min(3, int(r.get('props',0)>=8)*2+int(r.get('bool_props',0)>=2)),
        "shallow_simple":       lambda r: min(3, int(r.get('jsx_depth',0)<=3)*2+int(r['hooks_total']<=1)),
    }
    return rules.get(qid, lambda r: 0)(row)

ALL_QUERIES = [
    # Original 12
    ("complex_provider",    "Complex stateful provider with context and many hooks"),
    ("data_fetching",       "Component that fetches remote data with loading state"),
    ("form_component",      "Interactive form with event handlers and validation"),
    ("presentational",      "Simple stateless presentational display component"),
    ("animated",            "Animated interactive component with refs and effects"),
    ("virtualized",         "High performance virtualized list with memoization"),
    ("global_state",        "Global state manager with context and reducer"),
    ("dashboard",           "Complex data dashboard with fetching and many elements"),
    ("deep_no_hooks",       "Deeply nested JSX component with no state hooks"),
    ("stateless_layout",    "Stateless layout component rendering many children"),
    ("hooks_shallow",       "Hook-heavy logic component with shallow DOM"),
    ("tiny_pure",           "Tiny minimal pure component with almost no props"),
    # Extended 12 for training
    ("ref_heavy",           "Component using multiple refs for DOM manipulation"),
    ("callback_memoized",   "Memoized component with multiple useCallback optimizations"),
    ("high_import",         "Large component with many imports and dependencies"),
    ("map_heavy",           "List rendering component using multiple map calls"),
    ("conditional_complex", "Component with complex conditional rendering logic"),
    ("minimal_wrapper",     "Minimal wrapper component that passes children through"),
    ("event_rich",          "Event-rich component with many click and input handlers"),
    ("large_component",     "Large complex component with extensive logic"),
    ("custom_hooks_heavy",  "Component that uses many custom hooks"),
    ("filter_reduce",       "Data processing component using filter and reduce"),
    ("prop_heavy",          "Component accepting many props including boolean flags"),
    ("shallow_simple",      "Simple shallow component with minimal nesting"),
]

# ═══════════════════════════════════════════════════════════════
# 3. REPO-LEVEL SPLIT
# ═══════════════════════════════════════════════════════════════
repos = df['repo'].unique()
np.random.shuffle(repos)
split = int(len(repos) * 0.8)
train_repos = set(repos[:split])
test_repos  = set(repos[split:])
df_train = df[df['repo'].isin(train_repos)].reset_index(drop=True)
df_test  = df[df['repo'].isin(test_repos)].reset_index(drop=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[FEATS])
X_test  = scaler.transform(df_test[FEATS])

print(f"[SPLIT] Train: {len(df_train):,} ({len(train_repos)} repos) | Test: {len(df_test):,} ({len(test_repos)} repos)")

# Ground truth
gt_train = np.zeros((len(ALL_QUERIES), len(df_train)))
gt_test  = np.zeros((len(ALL_QUERIES), len(df_test)))
for qi, (qid, _) in enumerate(ALL_QUERIES):
    for ci, (_, row) in enumerate(df_train.iterrows()):
        gt_train[qi, ci] = gt_fn(qid, row)
    for ci, (_, row) in enumerate(df_test.iterrows()):
        gt_test[qi, ci] = gt_fn(qid, row)

# ═══════════════════════════════════════════════════════════════
# 4. EXTRACT ADABOOST FEATURE IMPORTANCES (supervision signal)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  EXTRACTING FEATURE IMPORTANCES (AdaBoost per query)")
print(f"{'='*70}")

importance_matrix = np.zeros((len(ALL_QUERIES), len(FEATS)))
for qi, (qid, _) in enumerate(ALL_QUERIES):
    y = (gt_train[qi] >= 2).astype(int)
    if y.sum() >= 5 and (y == 0).sum() >= 5:
        ada = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada.fit(X_train, y)
        importance_matrix[qi] = ada.feature_importances_
    print(f"  [{qid:25s}] top3: {', '.join(np.array(FEATS)[np.argsort(-importance_matrix[qi])[:3]])}")

# ═══════════════════════════════════════════════════════════════
# 5. LEVEL 1: TEMPLATE LOOKUP
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  LEVEL 1: Template Lookup")
print(f"{'='*70}")

# Use leave-one-out: for each test query, find nearest TRAINING query
# by TF-IDF cosine similarity, use its importance weights
tfidf = TfidfVectorizer(max_features=200)
query_texts = [q[1] for q in ALL_QUERIES]
tfidf_matrix = tfidf.fit_transform(query_texts)
query_sims = cosine_similarity(tfidf_matrix)  # (24, 24)

# ═══════════════════════════════════════════════════════════════
# 6. LEVEL 2: TF-IDF → Ridge → Feature Weights
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  LEVEL 2: TF-IDF → Ridge Regression → Feature Weights")
print(f"{'='*70}")

# Train: query_tfidf → importance_vector
ridge = Ridge(alpha=1.0)
ridge.fit(tfidf_matrix.toarray(), importance_matrix)
predicted_weights_ridge = ridge.predict(tfidf_matrix.toarray())
print(f"  Ridge R² on training queries: {ridge.score(tfidf_matrix.toarray(), importance_matrix):.3f}")

# ═══════════════════════════════════════════════════════════════
# 7. LEVEL 3: TF-IDF → MLP → Feature Weights
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  LEVEL 3: TF-IDF → MLP → Feature Weights")
print(f"{'='*70}")

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                   early_stopping=True, random_state=42)
mlp.fit(tfidf_matrix.toarray(), importance_matrix)
predicted_weights_mlp = mlp.predict(tfidf_matrix.toarray())
print(f"  MLP R² on training queries: {mlp.score(tfidf_matrix.toarray(), importance_matrix):.3f}")

# ═══════════════════════════════════════════════════════════════
# 8. RANKING FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def ndcg_at_k(ranked, gt_rel, k=10):
    dcg = sum((2**gt_rel[ranked[i]]-1)/np.log2(i+2) for i in range(min(k, len(ranked))))
    idcg = sum((2**r-1)/np.log2(i+2) for i, r in enumerate(sorted(gt_rel, reverse=True)[:k]))
    return (dcg/idcg) if idcg > 0 else 0.0

def rank_by_weights(weights, X):
    """Score = dot(weights, component_features). Higher = more relevant."""
    # Weights are importances — weight features, compute weighted sum
    # Components with high values on important features get high scores
    scores = X @ weights  # (n_components, n_features) @ (n_features,) = (n_components,)
    return np.argsort(-scores)  # descending

def rank_keyword(qi):
    terms = ALL_QUERIES[qi][1].lower().split()
    scores = np.array([sum(1 for t in terms if t in str(df_test.iloc[i]['component']).lower()) for i in range(len(df_test))])
    return np.argsort(-scores)

def rank_adaboost(qi):
    y = (gt_train[qi] >= 2).astype(int)
    if y.sum() < 5 or (y==0).sum() < 5:
        return np.arange(len(df_test))
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y)
    p = ada.predict_proba(X_test)
    return np.argsort(-p[:,1]) if p.shape[1]==2 else np.arange(len(df_test))

def rank_template_lookup(qi):
    """Find nearest training query, use its AdaBoost importance weights."""
    sims = query_sims[qi].copy()
    sims[qi] = -1  # exclude self
    nearest = np.argmax(sims)
    weights = importance_matrix[nearest]
    return rank_by_weights(weights, X_test)

def rank_ridge_ltr(qi):
    weights = predicted_weights_ridge[qi]
    return rank_by_weights(weights, X_test)

def rank_mlp_ltr(qi):
    weights = predicted_weights_mlp[qi]
    return rank_by_weights(weights, X_test)

def rank_oracle_weights(qi):
    """Oracle: use the ACTUAL AdaBoost importances for this query (upper bound)."""
    weights = importance_matrix[qi]
    return rank_by_weights(weights, X_test)

def rank_hybrid_ridge_ada(qi, recall_k=50):
    """
    TIER 3 HYBRID: Ridge for fast recall → AdaBoost for precise re-ranking.
    1. Ridge LTR scores all components → take top-K candidates
    2. Train AdaBoost on full train set
    3. Re-rank top-K by AdaBoost P(relevant)
    """
    # Step 1: Ridge recall
    ridge_weights = predicted_weights_ridge[qi]
    ridge_scores = X_test @ ridge_weights
    recall_idx = np.argsort(-ridge_scores)[:recall_k]

    # Step 2: AdaBoost re-rank
    y = (gt_train[qi] >= 2).astype(int)
    if y.sum() < 5 or (y == 0).sum() < 5:
        return np.concatenate([recall_idx, np.setdiff1d(np.arange(len(df_test)), recall_idx)])

    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y)
    proba = ada.predict_proba(X_test[recall_idx])
    if proba.shape[1] == 2:
        reranked = recall_idx[np.argsort(-proba[:, 1])]
    else:
        reranked = recall_idx

    # Append remaining indices
    rest = np.setdiff1d(np.arange(len(df_test)), recall_idx)
    return np.concatenate([reranked, rest])

def rank_hybrid_ridge_ada_100(qi):
    return rank_hybrid_ridge_ada(qi, recall_k=100)

def rank_hybrid_ridge_ada_200(qi):
    return rank_hybrid_ridge_ada(qi, recall_k=200)

# ═══════════════════════════════════════════════════════════════
# 9. EVALUATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  EVALUATION — 12 queries on held-out test repos")
print(f"{'='*70}\n")

K = 10
EVAL_QUERIES = list(range(12))  # Evaluate on original 12 queries

MODELS = {
    "Keyword":              rank_keyword,
    "L1: Template Lookup":  rank_template_lookup,
    "L2: Ridge LTR":        rank_ridge_ltr,
    "L3: MLP LTR":          rank_mlp_ltr,
    "Hybrid (top-50)":      lambda qi: rank_hybrid_ridge_ada(qi, 50),
    "Hybrid (top-100)":     rank_hybrid_ridge_ada_100,
    "Hybrid (top-200)":     rank_hybrid_ridge_ada_200,
    "AdaBoost (per-query)": rank_adaboost,
}

results = {}
per_query = {}

for name, fn in MODELS.items():
    t0 = time.time()
    scores = []
    pq = {}
    for qi in EVAL_QUERIES:
        ranked = fn(qi)
        n = ndcg_at_k(ranked, gt_test[qi], k=K)
        scores.append(n)
        pq[ALL_QUERIES[qi][0]] = n
    elapsed = time.time() - t0
    arr = np.array(scores)
    results[name] = arr
    per_query[name] = pq
    print(f"  {name:25s}  NDCG@{K}={arr.mean():.4f}  std={arr.std():.4f}  ({elapsed:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 10. RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  FINAL RESULTS (sorted by NDCG@10)")
print(f"{'='*70}")

# Bootstrap CIs
sorted_models = sorted(results, key=lambda m: results[m].mean(), reverse=True)
print(f"  {'#':>2}  {'Model':25s}  {'NDCG@10':>8}  {'95% CI':>16}")
print(f"  {'-'*55}")
for rank, name in enumerate(sorted_models, 1):
    arr = results[name]
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {rank:>2}  {name:25s}  {arr.mean():>8.4f}  [{lo:.4f}, {hi:.4f}]")

# Statistical significance
print(f"\n  Paired t-tests (AdaBoost vs LTR models):")
ada_arr = results["AdaBoost (per-query)"]
for name in ["L1: Template Lookup", "L2: Ridge LTR", "L3: MLP LTR"]:
    t, p = stats.ttest_rel(ada_arr, results[name])
    print(f"  AdaBoost vs {name:22s}  t={t:+.3f}  p={p:.4f}  {'✅' if p>0.05 else '❌'}")

# Per-query
print(f"\n{'='*70}")
print(f"  PER-QUERY NDCG@10")
print(f"{'='*70}")
header = f"  {'Query':25s}"
for m in sorted_models[:5]:
    header += f"  {m[:12]:>12}"
print(header)
print(f"  {'-'*88}")
for qi in EVAL_QUERIES:
    qid = ALL_QUERIES[qi][0]
    row = f"  {qid:25s}"
    for m in sorted_models[:5]:
        row += f"  {per_query[m].get(qid, 0):>12.4f}"
    print(row)

# Standard vs Hard
std_i = [i for i in range(len(EVAL_QUERIES)) if not ALL_QUERIES[EVAL_QUERIES[i]][0].startswith('deep') and not ALL_QUERIES[EVAL_QUERIES[i]][0].startswith('stateless') and not ALL_QUERIES[EVAL_QUERIES[i]][0].startswith('hooks_shallow') and not ALL_QUERIES[EVAL_QUERIES[i]][0].startswith('tiny')]
hard_i = [i for i in range(len(EVAL_QUERIES)) if i not in std_i]

print(f"\n  Standard ({len(std_i)} queries):", end="")
for m in sorted_models[:4]:
    print(f"  {m[:10]}={np.mean(results[m][std_i]):.4f}", end="")
print(f"\n  Hard ({len(hard_i)} queries):    ", end="")
for m in sorted_models[:4]:
    print(f"  {m[:10]}={np.mean(results[m][hard_i]):.4f}", end="")

# ═══════════════════════════════════════════════════════════════
# 11. GENERALIZATION TEST — Novel query
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("  GENERALIZATION: Novel unseen query")
print(f"{'='*70}")

novel_query = "A component that processes and filters data with multiple transformations"
novel_qid = "filter_reduce"  # closest GT

novel_tfidf = tfidf.transform([novel_query])
novel_ridge_weights = ridge.predict(novel_tfidf.toarray())[0]
novel_mlp_weights   = mlp.predict(novel_tfidf.toarray())[0]

print(f"\n  Query: '{novel_query}'")
print(f"\n  Ridge predicted top features:")
for feat, w in sorted(zip(FEATS, novel_ridge_weights), key=lambda x: x[1], reverse=True)[:8]:
    print(f"    {feat:22s}  weight={w:.4f}")

print(f"\n  MLP predicted top features:")
for feat, w in sorted(zip(FEATS, novel_mlp_weights), key=lambda x: x[1], reverse=True)[:8]:
    print(f"    {feat:22s}  weight={w:.4f}")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}\n")
