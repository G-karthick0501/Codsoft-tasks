"""
classical_approach.py — Proper Feature Engineering + Classical ML
================================================================
No embeddings. No FAISS. No sentence transformers.
Just structured features → EDA → engineered features → classical models.

The problem reframed:
  Given a query describing structural intent, rank components by
  how well their MEASURABLE features match that intent.

Run: python analytics/classical_approach.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & EDA
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("  CLASSICAL ML APPROACH — Feature Engineering + EDA")
print("=" * 65)

df = pd.read_csv('data/master2.csv')
df = df[df['component'].str.strip().ne('')].drop_duplicates(subset=['repo','file']).reset_index(drop=True)

RAW_FEATURES = ['hooks_total', 'useState', 'useEffect', 'useCallback',
                'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom',
                'props', 'jsx_depth', 'jsx_elems', 'conditionals',
                'map_calls', 'filter_calls', 'reduce_calls',
                'has_fetch', 'num_imports', 'event_handlers', 'bool_props',
                'has_children', 'loc']

for col in RAW_FEATURES:
    if col not in df.columns:
        df[col] = 0
df[RAW_FEATURES] = df[RAW_FEATURES].fillna(0)

print(f"\n[DATA] {len(df):,} components | {df['repo'].nunique()} repos | {len(RAW_FEATURES)} raw features\n")

# ── Basic EDA ─────────────────────────────────────────────────
print("── FEATURE DISTRIBUTIONS ──")
for col in RAW_FEATURES:
    vals = df[col]
    print(f"  {col:18s}  mean={vals.mean():6.2f}  std={vals.std():6.2f}  "
          f"min={vals.min():3.0f}  max={vals.max():4.0f}  "
          f"zero%={100*(vals==0).mean():5.1f}%")

# ── Correlations ──────────────────────────────────────────────
print("\n── TOP CORRELATIONS (|r| > 0.3) ──")
corr = df[RAW_FEATURES].corr()
pairs = []
for i, c1 in enumerate(RAW_FEATURES):
    for j, c2 in enumerate(RAW_FEATURES):
        if i < j and abs(corr.loc[c1, c2]) > 0.3:
            pairs.append((c1, c2, corr.loc[c1, c2]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for c1, c2, r in pairs[:15]:
    print(f"  {c1:18s} × {c2:18s}  r={r:+.3f}")

# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  2. FEATURE ENGINEERING")
print(f"{'='*65}")

# Derived features — domain knowledge about React components
df['state_hooks']     = df['useState'] + df['useReducer']
df['effect_hooks']    = df['useEffect'] + df['useCallback'] + df['useMemo']
df['context_hooks']   = df['useContext'] + df['useReducer']
df['ref_hooks']       = df['useRef']
df['hook_diversity']  = (df[['useState','useEffect','useCallback','useMemo',
                             'useContext','useReducer','useRef','useCustom']] > 0).sum(axis=1)
df['complexity_score'] = (df['hooks_total'] * 2 + df['conditionals'] + 
                          df['map_calls'] + df['filter_calls'] + 
                          df['event_handlers'] + df['has_fetch'] * 3)
df['interactivity']   = df['event_handlers'] + df['bool_props'] + df['has_children']
df['data_pattern']    = df['has_fetch'] + (df['useState'] > 0).astype(int) + (df['useEffect'] > 0).astype(int)
df['jsx_density']     = df['jsx_elems'] / (df['loc'].clip(lower=1))
df['hooks_per_loc']   = df['hooks_total'] / (df['loc'].clip(lower=1))
df['props_per_loc']   = df['props'] / (df['loc'].clip(lower=1))
df['is_stateless']    = (df['hooks_total'] == 0).astype(int)
df['is_complex']      = (df['hooks_total'] >= 5).astype(int)
df['is_fetcher']      = df['has_fetch']

ENGINEERED = ['state_hooks', 'effect_hooks', 'context_hooks', 'ref_hooks',
              'hook_diversity', 'complexity_score', 'interactivity',
              'data_pattern', 'jsx_density', 'hooks_per_loc', 'props_per_loc',
              'is_stateless', 'is_complex', 'is_fetcher']

ALL_FEATURES = RAW_FEATURES + ENGINEERED
print(f"  Raw features:        {len(RAW_FEATURES)}")
print(f"  Engineered features: {len(ENGINEERED)}")
print(f"  Total features:      {len(ALL_FEATURES)}")

# ═══════════════════════════════════════════════════════════════
# 3. COMPONENT ARCHETYPES (Unsupervised — KMeans)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  3. COMPONENT ARCHETYPES (KMeans Clustering)")
print(f"{'='*65}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[ALL_FEATURES])

# Find good K
from sklearn.metrics import silhouette_score
best_k, best_sil = 2, -1
for k in [3, 4, 5, 6, 7, 8]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
    if sil > best_sil:
        best_k, best_sil = k, sil
    print(f"  K={k}: silhouette={sil:.3f}")

print(f"\n  Best K={best_k} (silhouette={best_sil:.3f})")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df['cluster'] = km.fit_predict(X_scaled)

print(f"\n  Cluster profiles:")
for c in range(best_k):
    mask = df['cluster'] == c
    n = mask.sum()
    ht = df.loc[mask, 'hooks_total'].mean()
    jd = df.loc[mask, 'jsx_depth'].mean()
    pr = df.loc[mask, 'props'].mean()
    hf = df.loc[mask, 'has_fetch'].mean()
    cs = df.loc[mask, 'complexity_score'].mean()
    lc = df.loc[mask, 'loc'].mean()
    print(f"  Cluster {c}: n={n:4d} | hooks={ht:.1f} depth={jd:.1f} props={pr:.1f} "
          f"fetch={hf:.2f} complex={cs:.1f} loc={lc:.0f}")

# ═══════════════════════════════════════════════════════════════
# 4. QUERY-TO-FEATURE MAPPING (No embeddings needed!)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  4. STRUCTURAL QUERY ENGINE (Feature-Based)")
print(f"{'='*65}")

# A query maps to a TARGET FEATURE PROFILE — what do we want?
# Then we rank by distance to that profile.

QUERIES = [
    {
        "id": "complex_provider",
        "query": "Complex stateful provider with context and many hooks",
        "target": {"hooks_total": 8, "useContext": 1, "useReducer": 1, "props": 5,
                   "jsx_depth": 6, "complexity_score": 15, "is_complex": 1},
    },
    {
        "id": "data_fetching",
        "query": "Component that fetches remote data with loading state",
        "target": {"has_fetch": 1, "useState": 1, "useEffect": 1, "data_pattern": 3,
                   "conditionals": 2},
    },
    {
        "id": "form_component",
        "query": "Interactive form with event handlers and validation",
        "target": {"event_handlers": 3, "props": 6, "useState": 1, "bool_props": 2,
                   "interactivity": 5},
    },
    {
        "id": "presentational",
        "query": "Simple stateless presentational component",
        "target": {"hooks_total": 0, "is_stateless": 1, "has_fetch": 0,
                   "jsx_depth": 3, "loc": 30, "complexity_score": 0},
    },
    {
        "id": "animated",
        "query": "Animated interactive component with refs and effects",
        "target": {"useEffect": 1, "useRef": 1, "useCallback": 1,
                   "conditionals": 3, "effect_hooks": 3},
    },
    {
        "id": "virtualized",
        "query": "High performance list with memoization and mapping",
        "target": {"map_calls": 3, "useMemo": 1, "useCallback": 1, "loc": 200,
                   "jsx_depth": 7},
    },
    {
        "id": "global_state",
        "query": "Global state manager with context and reducer",
        "target": {"useContext": 1, "useReducer": 1, "context_hooks": 2,
                   "hooks_total": 4},
    },
    {
        "id": "dashboard",
        "query": "Complex data dashboard with fetching and many elements",
        "target": {"has_fetch": 1, "jsx_elems": 15, "hooks_total": 5, "loc": 200,
                   "complexity_score": 12},
    },
    # HARD QUERIES
    {
        "id": "HARD:deep_no_hooks",
        "query": "Deeply nested JSX but no state hooks at all",
        "target": {"jsx_depth": 10, "hooks_total": 0, "is_stateless": 1},
    },
    {
        "id": "HARD:stateless_layout",
        "query": "Stateless layout rendering many child elements",
        "target": {"hooks_total": 0, "jsx_elems": 20, "props": 4, "is_stateless": 1},
    },
    {
        "id": "HARD:hooks_shallow",
        "query": "Hook-heavy logic with minimal shallow DOM",
        "target": {"hooks_total": 7, "jsx_depth": 3, "is_complex": 1, "loc": 80},
    },
    {
        "id": "HARD:tiny_pure",
        "query": "Tiny minimal pure component",
        "target": {"loc": 20, "hooks_total": 0, "props": 1, "complexity_score": 0,
                   "is_stateless": 1},
    },
]

# ── Ground truth functions (same as rigorous_evaluation.py) ──
def gt_fn(qid, row):
    if qid == "complex_provider":
        s = int(row['hooks_total'] >= 5) + int(row['hooks_total'] >= 8) + int(row.get('useContext',0) > 0 or row.get('useReducer',0) > 0)
    elif qid == "data_fetching":
        s = int(row.get('has_fetch',0)==1)*2 + int(row.get('useState',0)>0) + int(row.get('useEffect',0)>0)
    elif qid == "form_component":
        s = int(row.get('event_handlers',0)>=2) + int(row.get('props',0)>=4) + int(row.get('useState',0)>0) + int(row.get('bool_props',0)>0)
    elif qid == "presentational":
        s = int(row['hooks_total']==0)*2 + int(row.get('has_fetch',0)==0) + int(row.get('jsx_depth',0)<=4)
    elif qid == "animated":
        s = int(row.get('useEffect',0)>0) + int(row.get('useRef',0)>0) + int(row.get('conditionals',0)>=3) + int(row.get('useCallback',0)>0)
    elif qid == "virtualized":
        s = int(row.get('map_calls',0)>=2) + int(row.get('useMemo',0)>0 or row.get('useCallback',0)>0) + int(row.get('loc',0)>=100) + int(row.get('jsx_depth',0)>=5)
    elif qid == "global_state":
        s = int(row.get('useContext',0)>0)*2 + int(row.get('useReducer',0)>0)*2
    elif qid == "dashboard":
        s = int(row.get('has_fetch',0)==1) + int(row.get('jsx_elems',0)>=10) + int(row['hooks_total']>=4) + int(row.get('loc',0)>=150)
    elif qid == "HARD:deep_no_hooks":
        s = int(row.get('jsx_depth',0)>=8)*2 + int(row['hooks_total']==0)*2
    elif qid == "HARD:stateless_layout":
        s = int(row['hooks_total']==0) + int(row.get('jsx_elems',0)>=15)*2 + int(row.get('props',0)>=3)
    elif qid == "HARD:hooks_shallow":
        s = int(row['hooks_total']>=5)*2 + int(row.get('jsx_depth',0)<=4)*2
    elif qid == "HARD:tiny_pure":
        s = int(row.get('loc',0)<=40) + int(row['hooks_total']==0) + int(row.get('props',0)<=2) + int(row.get('has_fetch',0)==0)
    else:
        s = 0
    return min(s, 3)

# ═══════════════════════════════════════════════════════════════
# 5. RETRIEVAL METHODS
# ═══════════════════════════════════════════════════════════════

def ndcg_at_k(ranked_indices, gt_rel, k=10):
    dcg = sum((2**gt_rel[ranked_indices[i]] - 1) / np.log2(i + 2) for i in range(min(k, len(ranked_indices))))
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted(gt_rel, reverse=True)[:k]))
    return (dcg / idcg) if idcg > 0 else 0.0

def method_keyword(df_eval, query_text):
    terms = query_text.lower().split()
    scores = np.array([sum(1 for t in terms if t in str(df_eval.iloc[i]['component']).lower()) for i in range(len(df_eval))])
    return np.argsort(-scores)

def method_feature_knn(df_eval, target_profile, feature_cols, scaler_fit):
    """KNN on feature space — map query to target feature vector, find nearest."""
    target_vec = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        if col in target_profile:
            target_vec[i] = target_profile[col]
    target_scaled = scaler_fit.transform(target_vec.reshape(1, -1))
    X_eval = scaler_fit.transform(df_eval[feature_cols].values)
    dists = np.linalg.norm(X_eval - target_scaled, axis=1)
    return np.argsort(dists)

def method_weighted_feature(df_eval, target_profile, feature_cols):
    """Weighted feature matching — only compare features specified in the query profile."""
    active_cols = [c for c in target_profile.keys() if c in feature_cols]
    if not active_cols:
        return np.arange(len(df_eval))
    
    scores = np.zeros(len(df_eval))
    for col in active_cols:
        target_val = target_profile[col]
        actual = df_eval[col].values.astype(float)
        
        if target_val == 0:
            # Want zero: penalize non-zero
            scores += np.abs(actual) * 2
        elif target_val == 1 and col.startswith(('is_', 'has_')):
            # Binary feature: exact match reward
            scores -= (actual == target_val).astype(float) * 3
        else:
            # Continuous: penalize distance, reward closeness
            max_val = actual.max() or 1
            scores += np.abs(actual - target_val) / max_val
    
    return np.argsort(scores)

def method_rf_classifier(df_train, df_eval, gt_train, gt_eval_dummy, feature_cols):
    """Random Forest: train relevance classifier, predict on eval."""
    # Binary: relevant (gt >= 2) vs not
    y_train = (gt_train >= 2).astype(int)
    if y_train.sum() < 5 or (y_train == 0).sum() < 5:
        return np.arange(len(df_eval))  # too few examples
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
    rf.fit(df_train[feature_cols].values, y_train)
    proba = rf.predict_proba(df_eval[feature_cols].values)
    if proba.shape[1] == 2:
        return np.argsort(-proba[:, 1])  # rank by relevance probability
    return np.arange(len(df_eval))

# ═══════════════════════════════════════════════════════════════
# 6. REPO-LEVEL HOLDOUT EVALUATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  5. EVALUATION (Repo-level 80/20 holdout)")
print(f"{'='*65}")

all_repos = df['repo'].unique()
np.random.shuffle(all_repos)
split = int(len(all_repos) * 0.8)
train_repos = set(all_repos[:split])
test_repos  = set(all_repos[split:])
df_train = df[df['repo'].isin(train_repos)].reset_index(drop=True)
df_test  = df[df['repo'].isin(test_repos)].reset_index(drop=True)

print(f"  Train: {len(df_train):,} ({len(train_repos)} repos)")
print(f"  Test:  {len(df_test):,} ({len(test_repos)} repos)")

# Fit scaler on train
scaler_train = StandardScaler()
scaler_train.fit(df_train[ALL_FEATURES])

# Pre-compute ground truth for test
K = 10
gt_matrix = np.zeros((len(QUERIES), len(df_test)))
for qi, q in enumerate(QUERIES):
    for ci, (_, row) in enumerate(df_test.iterrows()):
        gt_matrix[qi, ci] = gt_fn(q['id'], row)

# Also compute GT on train (for RF)
gt_train_matrix = np.zeros((len(QUERIES), len(df_train)))
for qi, q in enumerate(QUERIES):
    for ci, (_, row) in enumerate(df_train.iterrows()):
        gt_train_matrix[qi, ci] = gt_fn(q['id'], row)

# ── Run all methods ──
methods = {
    "Keyword (name)":       lambda qi: method_keyword(df_test, QUERIES[qi]['query']),
    "Feature KNN":          lambda qi: method_feature_knn(df_test, QUERIES[qi]['target'], ALL_FEATURES, scaler_train),
    "Weighted Feature":     lambda qi: method_weighted_feature(df_test, QUERIES[qi]['target'], ALL_FEATURES),
    "Random Forest":        lambda qi: method_rf_classifier(df_train, df_test, gt_train_matrix[qi], None, ALL_FEATURES),
}

all_ndcg = {m: [] for m in methods}
per_query = {m: {} for m in methods}

for model_name, fn in methods.items():
    for qi, q in enumerate(QUERIES):
        ranked = fn(qi)
        n = ndcg_at_k(ranked, gt_matrix[qi], k=K)
        all_ndcg[model_name].append(n)
        per_query[model_name][q['id']] = n

print(f"\n  {'Model':22s}  {'NDCG@10':>8}  {'Std':>6}")
print(f"  {'-'*42}")
for m in methods:
    arr = np.array(all_ndcg[m])
    print(f"  {m:22s}  {arr.mean():>8.4f}  {arr.std():>6.4f}")

# ── Statistical significance ──
print(f"\n  Paired t-tests (Weighted Feature vs others):")
wf = np.array(all_ndcg["Weighted Feature"])
for m in ["Keyword (name)", "Feature KNN", "Random Forest"]:
    t, p = stats.ttest_rel(wf, np.array(all_ndcg[m]))
    sig = "✅" if p < 0.05 else "❌"
    print(f"  vs {m:22s}  t={t:+.3f}  p={p:.4f}  {sig}")

# ── Bootstrap CIs ──
print(f"\n  Bootstrap 95% CIs:")
for m in methods:
    arr = np.array(all_ndcg[m])
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {m:22s}  {arr.mean():.4f}  [{lo:.4f}, {hi:.4f}]")

# ── Per-query breakdown ──
print(f"\n{'='*65}")
print("  6. PER-QUERY NDCG@10")
print(f"{'='*65}")
print(f"  {'Query':25s}  {'Keyword':>8}  {'KNN':>8}  {'Weighted':>8}  {'RF':>8}")
print(f"  {'-'*61}")
for q in QUERIES:
    row = f"  {q['id']:25s}"
    for m in methods:
        row += f"  {per_query[m][q['id']]:>8.4f}"
    print(row)

# Standard vs Hard
std_idx = [i for i in range(len(QUERIES)) if not QUERIES[i]['id'].startswith('HARD')]
hard_idx = [i for i in range(len(QUERIES)) if QUERIES[i]['id'].startswith('HARD')]
print(f"\n  Standard avg: ", end="")
for m in methods:
    print(f"  {m[:8]:>8}={np.mean(np.array(all_ndcg[m])[std_idx]):.4f}", end="")
print(f"\n  Hard avg:     ", end="")
for m in methods:
    print(f"  {m[:8]:>8}={np.mean(np.array(all_ndcg[m])[hard_idx]):.4f}", end="")

# ── Feature importance from RF ──
print(f"\n\n{'='*65}")
print("  7. FEATURE IMPORTANCE (Random Forest — averaged across queries)")
print(f"{'='*65}")

importances = np.zeros(len(ALL_FEATURES))
n_valid = 0
for qi in range(len(QUERIES)):
    y = (gt_train_matrix[qi] >= 2).astype(int)
    if y.sum() >= 5 and (y == 0).sum() >= 5:
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
        rf.fit(df_train[ALL_FEATURES].values, y)
        importances += rf.feature_importances_
        n_valid += 1

if n_valid > 0:
    importances /= n_valid
    ranked_feats = sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in ranked_feats[:15]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:22s}  {imp:.4f}  {bar}")

print(f"\n{'='*65}")
print("  DONE")
print(f"{'='*65}\n")
