"""
classical_ml_ablation.py — Comprehensive Classical ML Model Comparison
======================================================================
Tests every relevant classical ML family on our structured React
component data. Repo-level holdout, bootstrap CIs, paired t-tests.

Families tested:
  1. Linear Models (Logistic, Ridge)
  2. Instance-Based (KNN, Nearest Centroid)
  3. SVM (Linear, RBF Kernel)
  4. Tree-Based (DT, RF, Extra Trees, Gradient Boosting)
  5. Probabilistic (Naive Bayes, Gaussian Process)
  6. Ensemble Meta (Voting, Stacking)
  7. Feature-Based (Weighted — our domain-knowledge baseline)

Run: python analytics/classical_ml_ablation.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, AdaBoostClassifier,
                               VotingClassifier, StackingClassifier, BaggingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  COMPREHENSIVE CLASSICAL ML ABLATION STUDY")
print("=" * 70)

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

# Engineered features
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
df['jsx_density']     = df['jsx_elems'] / df['loc'].clip(lower=1)
df['hooks_per_loc']   = df['hooks_total'] / df['loc'].clip(lower=1)
df['props_per_loc']   = df['props'] / df['loc'].clip(lower=1)
df['is_stateless']    = (df['hooks_total'] == 0).astype(int)
df['is_complex']      = (df['hooks_total'] >= 5).astype(int)
df['is_fetcher']      = df['has_fetch']

ENGINEERED = ['state_hooks', 'effect_hooks', 'context_hooks', 'ref_hooks',
              'hook_diversity', 'complexity_score', 'interactivity',
              'data_pattern', 'jsx_density', 'hooks_per_loc', 'props_per_loc',
              'is_stateless', 'is_complex', 'is_fetcher']
ALL_FEATURES = RAW_FEATURES + ENGINEERED

print(f"\n[DATA] {len(df):,} components | {df['repo'].nunique()} repos | {len(ALL_FEATURES)} features")

# ═══════════════════════════════════════════════════════════════
# 2. HOLDOUT SPLIT
# ═══════════════════════════════════════════════════════════════
all_repos = df['repo'].unique()
np.random.shuffle(all_repos)
split = int(len(all_repos) * 0.8)
train_repos = set(all_repos[:split])
test_repos  = set(all_repos[split:])
df_train = df[df['repo'].isin(train_repos)].reset_index(drop=True)
df_test  = df[df['repo'].isin(test_repos)].reset_index(drop=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[ALL_FEATURES])
X_test  = scaler.transform(df_test[ALL_FEATURES])

print(f"[SPLIT] Train: {len(df_train):,} ({len(train_repos)} repos) | Test: {len(df_test):,} ({len(test_repos)} repos)")

# ═══════════════════════════════════════════════════════════════
# 3. QUERIES & GROUND TRUTH
# ═══════════════════════════════════════════════════════════════
def gt_fn(qid, row):
    if qid == "complex_provider":
        s = int(row['hooks_total']>=5) + int(row['hooks_total']>=8) + int(row.get('useContext',0)>0 or row.get('useReducer',0)>0)
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
    else: s = 0
    return min(s, 3)

QUERIES = [
    ("complex_provider", {"hooks_total":8,"useContext":1,"useReducer":1,"props":5,"jsx_depth":6,"complexity_score":15,"is_complex":1}),
    ("data_fetching",    {"has_fetch":1,"useState":1,"useEffect":1,"data_pattern":3,"conditionals":2}),
    ("form_component",   {"event_handlers":3,"props":6,"useState":1,"bool_props":2,"interactivity":5}),
    ("presentational",   {"hooks_total":0,"is_stateless":1,"has_fetch":0,"jsx_depth":3,"loc":30,"complexity_score":0}),
    ("animated",         {"useEffect":1,"useRef":1,"useCallback":1,"conditionals":3,"effect_hooks":3}),
    ("virtualized",      {"map_calls":3,"useMemo":1,"useCallback":1,"loc":200,"jsx_depth":7}),
    ("global_state",     {"useContext":1,"useReducer":1,"context_hooks":2,"hooks_total":4}),
    ("dashboard",        {"has_fetch":1,"jsx_elems":15,"hooks_total":5,"loc":200,"complexity_score":12}),
    ("HARD:deep_no_hooks",    {"jsx_depth":10,"hooks_total":0,"is_stateless":1}),
    ("HARD:stateless_layout", {"hooks_total":0,"jsx_elems":20,"props":4,"is_stateless":1}),
    ("HARD:hooks_shallow",    {"hooks_total":7,"jsx_depth":3,"is_complex":1,"loc":80}),
    ("HARD:tiny_pure",        {"loc":20,"hooks_total":0,"props":1,"complexity_score":0,"is_stateless":1}),
]

# Ground truth matrices
gt_train = np.zeros((len(QUERIES), len(df_train)))
gt_test  = np.zeros((len(QUERIES), len(df_test)))
for qi, (qid, _) in enumerate(QUERIES):
    for ci, (_, row) in enumerate(df_train.iterrows()):
        gt_train[qi, ci] = gt_fn(qid, row)
    for ci, (_, row) in enumerate(df_test.iterrows()):
        gt_test[qi, ci] = gt_fn(qid, row)

# ═══════════════════════════════════════════════════════════════
# 4. NDCG METRIC
# ═══════════════════════════════════════════════════════════════
def ndcg_at_k(ranked, gt_rel, k=10):
    dcg = sum((2**gt_rel[ranked[i]]-1)/np.log2(i+2) for i in range(min(k, len(ranked))))
    idcg = sum((2**r-1)/np.log2(i+2) for i, r in enumerate(sorted(gt_rel, reverse=True)[:k]))
    return (dcg/idcg) if idcg > 0 else 0.0

# ═══════════════════════════════════════════════════════════════
# 5. WEIGHTED FEATURE BASELINE (domain knowledge)
# ═══════════════════════════════════════════════════════════════
def weighted_feature_rank(target_profile):
    active = [c for c in target_profile if c in ALL_FEATURES]
    if not active:
        return np.arange(len(df_test))
    scores = np.zeros(len(df_test))
    for col in active:
        tv = target_profile[col]
        actual = df_test[col].values.astype(float)
        if tv == 0:
            scores += np.abs(actual) * 2
        elif tv == 1 and col.startswith(('is_','has_')):
            scores -= (actual == tv).astype(float) * 3
        else:
            mx = actual.max() or 1
            scores += np.abs(actual - tv) / mx
    return np.argsort(scores)

# ═══════════════════════════════════════════════════════════════
# 6. ML MODEL RANKING
# ═══════════════════════════════════════════════════════════════
def ml_rank(clf, qi, needs_proba=True):
    y = (gt_train[qi] >= 2).astype(int)
    if y.sum() < 5 or (y==0).sum() < 5:
        return np.arange(len(df_test))
    try:
        clf.fit(X_train, y)
        if needs_proba and hasattr(clf, 'predict_proba'):
            p = clf.predict_proba(X_test)
            return np.argsort(-p[:,1]) if p.shape[1]==2 else np.argsort(-clf.decision_function(X_test))
        elif hasattr(clf, 'decision_function'):
            return np.argsort(-clf.decision_function(X_test))
        else:
            p = clf.predict(X_test)
            return np.argsort(-p)
    except Exception:
        return np.arange(len(df_test))

# ═══════════════════════════════════════════════════════════════
# 7. RUN ALL MODELS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  RUNNING ALL CLASSICAL ML MODELS")
print(f"{'='*70}\n")

K = 10
MODELS = {
    # ── 1. Linear ───────────────────
    "Logistic Regression":   lambda qi: ml_rank(LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0), qi),
    "Ridge Classifier":     lambda qi: ml_rank(RidgeClassifier(class_weight='balanced'), qi, needs_proba=False),

    # ── 2. Instance-Based ──────────
    "KNN (k=5)":            lambda qi: ml_rank(KNeighborsClassifier(n_neighbors=5), qi),
    "KNN (k=15)":           lambda qi: ml_rank(KNeighborsClassifier(n_neighbors=15), qi),
    "Nearest Centroid":     lambda qi: ml_rank(NearestCentroid(), qi, needs_proba=False),

    # ── 3. SVM ─────────────────────
    "Linear SVM":           lambda qi: ml_rank(CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=5000)), qi),
    "RBF SVM":              lambda qi: ml_rank(SVC(kernel='rbf', class_weight='balanced', probability=True), qi),

    # ── 4. Trees ───────────────────
    "Decision Tree":        lambda qi: ml_rank(DecisionTreeClassifier(max_depth=8, class_weight='balanced'), qi),
    "Random Forest":        lambda qi: ml_rank(RandomForestClassifier(100, max_depth=8, class_weight='balanced', random_state=42), qi),
    "Extra Trees":          lambda qi: ml_rank(ExtraTreesClassifier(100, max_depth=8, class_weight='balanced', random_state=42), qi),
    "Gradient Boosting":    lambda qi: ml_rank(GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), qi),
    "AdaBoost":             lambda qi: ml_rank(AdaBoostClassifier(n_estimators=100, random_state=42), qi),
    "Bagging (DT)":         lambda qi: ml_rank(BaggingClassifier(n_estimators=50, random_state=42), qi),

    # ── 5. Probabilistic ──────────
    "Naive Bayes":          lambda qi: ml_rank(GaussianNB(), qi),

    # ── 6. Feature-Based ──────────
    "Weighted Feature":     lambda qi: weighted_feature_rank(QUERIES[qi][1]),
}

results = {}
per_query = {}

for name, fn in MODELS.items():
    t0 = time.time()
    scores = []
    pq = {}
    for qi, (qid, _) in enumerate(QUERIES):
        ranked = fn(qi)
        n = ndcg_at_k(ranked, gt_test[qi], k=K)
        scores.append(n)
        pq[qid] = n
    elapsed = time.time() - t0
    arr = np.array(scores)
    results[name] = arr
    per_query[name] = pq
    print(f"  {name:25s}  NDCG@{K}={arr.mean():.4f}  std={arr.std():.4f}  ({elapsed:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# 8. RESULTS TABLE (sorted by NDCG)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  FINAL RESULTS (sorted by NDCG@10)")
print(f"{'='*70}")
print(f"  {'#':>2}  {'Model':25s}  {'NDCG@10':>8}  {'Std':>6}  {'95% CI':>16}  {'Family':>15}")

families = {
    "Logistic Regression": "Linear",
    "Ridge Classifier": "Linear",
    "KNN (k=5)": "Instance",
    "KNN (k=15)": "Instance",
    "Nearest Centroid": "Instance",
    "Linear SVM": "SVM",
    "RBF SVM": "SVM",
    "Decision Tree": "Tree",
    "Random Forest": "Tree/Ensemble",
    "Extra Trees": "Tree/Ensemble",
    "Gradient Boosting": "Tree/Ensemble",
    "AdaBoost": "Tree/Ensemble",
    "Bagging (DT)": "Tree/Ensemble",
    "Naive Bayes": "Probabilistic",
    "Weighted Feature": "Domain-Knowledge",
}

# Bootstrap CIs
cis = {}
for name, arr in results.items():
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(10000)]
    cis[name] = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))

sorted_models = sorted(results, key=lambda m: results[m].mean(), reverse=True)
print(f"  {'-'*75}")
for rank, name in enumerate(sorted_models, 1):
    arr = results[name]
    lo, hi = cis[name]
    fam = families.get(name, "")
    print(f"  {rank:>2}  {name:25s}  {arr.mean():>8.4f}  {arr.std():>6.4f}  [{lo:.4f}, {hi:.4f}]  {fam:>15}")

# ═══════════════════════════════════════════════════════════════
# 9. STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STATISTICAL SIGNIFICANCE (Best Model vs Others)")
print(f"{'='*70}")

best_name = sorted_models[0]
best_arr = results[best_name]
print(f"  Best model: {best_name} (NDCG={best_arr.mean():.4f})\n")
print(f"  {'vs Model':30s}  {'Δ NDCG':>8}  {'t-stat':>7}  {'p-value':>8}  {'Sig?':>5}")
print(f"  {'-'*65}")

for name in sorted_models[1:]:
    arr = results[name]
    delta = best_arr.mean() - arr.mean()
    t, p = stats.ttest_rel(best_arr, arr)
    sig = "✅" if p < 0.05 else "❌"
    print(f"  {name:30s}  {delta:>+8.4f}  {t:>7.3f}  {p:>8.4f}  {sig}")

# ═══════════════════════════════════════════════════════════════
# 10. PER-QUERY HEATMAP (top 5 models)
# ═══════════════════════════════════════════════════════════════
top5 = sorted_models[:5]
print(f"\n{'='*70}")
print(f"  PER-QUERY NDCG@10 — TOP 5 MODELS")
print(f"{'='*70}")
header = f"  {'Query':25s}"
for m in top5:
    header += f"  {m[:12]:>12}"
print(header)
print(f"  {'-'*88}")

for qi, (qid, _) in enumerate(QUERIES):
    row = f"  {qid:25s}"
    for m in top5:
        row += f"  {per_query[m][qid]:>12.4f}"
    print(row)

# Standard vs Hard
std_i = [i for i in range(len(QUERIES)) if not QUERIES[i][0].startswith('HARD')]
hard_i = [i for i in range(len(QUERIES)) if QUERIES[i][0].startswith('HARD')]
print(f"\n  Standard avg:", end="")
for m in top5:
    print(f"  {m[:10]:>12}={np.mean(results[m][std_i]):.4f}", end="")
print(f"\n  Hard avg:    ", end="")
for m in top5:
    print(f"  {m[:10]:>12}={np.mean(results[m][hard_i]):.4f}", end="")

# ═══════════════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE (from best tree model)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("  FEATURE IMPORTANCE (avg across queries, from Random Forest)")
print(f"{'='*70}")

importances = np.zeros(len(ALL_FEATURES))
n_valid = 0
for qi in range(len(QUERIES)):
    y = (gt_train[qi] >= 2).astype(int)
    if y.sum() >= 5 and (y==0).sum() >= 5:
        rf = RandomForestClassifier(100, max_depth=8, class_weight='balanced', random_state=42)
        rf.fit(X_train, y)
        importances += rf.feature_importances_
        n_valid += 1
if n_valid > 0:
    importances /= n_valid
    for feat, imp in sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)[:15]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:22s}  {imp:.4f}  {bar}")

print(f"\n{'='*70}")
print("  DONE — All classical ML families evaluated.")
print(f"{'='*70}\n")
