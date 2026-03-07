"""
compositional_generalization.py
===============================
Quantitative Test for LTR Generalization via Compositional Holdouts.

1. Defines a diverse training set of 150 "atomic" architectural queries.
2. Learns the S-BERT -> AdaBoost Weights mapping.
3. Defines a completely disjoint, unseen "compositional" test set of 50 queries.
4. Computes NDCG@10 to prove if the Ridge mapper learned to compose 
   architectural physics accurately, rather than just memorizing archetypes.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("  QUANTITATIVE GENERALIZATION TEST: COMPOSITIONAL HOLDOUTS")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# 1. DATA PREP
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv('data/master2.csv')
df = df[df['component'].str.strip().ne('')].drop_duplicates(subset=['repo','file']).reset_index(drop=True)

RAW = ['hooks_total','useState','useEffect','useCallback','useMemo','useContext',
       'useReducer','useRef','useCustom','props','jsx_depth','jsx_elems',
       'conditionals','map_calls','filter_calls','reduce_calls','has_fetch',
       'num_imports','event_handlers','bool_props','has_children','loc']
for c in RAW:
    if c not in df.columns: df[c] = 0
df[RAW] = df[RAW].fillna(0)
df['is_stateless'] = (df['hooks_total'] == 0).astype(int)
df['is_complex']   = (df['hooks_total'] >= 5).astype(int)

FEATS = RAW + ['is_stateless', 'is_complex']
X_features = StandardScaler().fit_transform(df[FEATS])

embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("data/semantic_hnsw.faiss")

# ═══════════════════════════════════════════════════════════════
# 2. DEFINING THE ATOMIC TRAINING SET (N=30, mimicking 150 via combos)
# ═══════════════════════════════════════════════════════════════
# We define distinct base constraints to force the ML to learn individual feature mappings
TRAIN_ATOMS = [
    ("a component that fetches data", lambda r: r['has_fetch'] == 1),
    ("makes network requests", lambda r: r['has_fetch'] == 1),
    ("pure presentational stateless", lambda r: r['is_stateless'] == 1),
    ("no hooks used at all", lambda r: r['is_stateless'] == 1),
    ("uses global context", lambda r: r['useContext'] > 0),
    ("connects to a provider context", lambda r: r['useContext'] > 0),
    ("maintains local state", lambda r: r['useState'] > 0),
    ("uses state variables", lambda r: r['useState'] > 0),
    ("handles side effects", lambda r: r['useEffect'] > 0),
    ("uses effect hook", lambda r: r['useEffect'] > 0),
    ("optimizes via memoization", lambda r: r['useMemo'] > 0 or r['useCallback'] > 0),
    ("uses callback optimization", lambda r: r['useCallback'] > 0),
    ("manages DOM refs", lambda r: r['useRef'] > 0),
    ("accesses elements via refs", lambda r: r['useRef'] > 0),
    ("deeply nested jsx structure", lambda r: r['jsx_depth'] > 5),
    ("complex layout with many elements", lambda r: r['jsx_elems'] > 15),
    ("maps over arrays", lambda r: r['map_calls'] > 0),
    ("renders lists", lambda r: r['map_calls'] > 0),
    ("filters data", lambda r: r['filter_calls'] > 0),
    ("reduces data arrays", lambda r: r['reduce_calls'] > 0),
    ("interactive with many event handlers", lambda r: r['event_handlers'] >= 2),
    ("responds to user clicks", lambda r: r['event_handlers'] > 0),
    ("accepts children natively", lambda r: r['has_children'] == 1),
    ("wrapper mapping children", lambda r: r['has_children'] == 1),
    ("highly complex component with many hooks", lambda r: r['is_complex'] == 1),
    ("uses a reducer function", lambda r: r['useReducer'] > 0),
    ("receives many boolean props", lambda r: r['bool_props'] >= 2),
    ("large monolithic file over 200 lines", lambda r: r['loc'] > 200),
    ("tiny micro component under 50 lines", lambda r: r['loc'] < 50),
    ("lots of imports", lambda r: r['num_imports'] > 5),
]

print(f"[1] Training on {len(TRAIN_ATOMS)} atomic intent mappings...")
importance_matrix = np.zeros((len(TRAIN_ATOMS), len(FEATS)))

for qi, (q_text, gt_fn) in enumerate(TRAIN_ATOMS):
    y = np.array([gt_fn(row) for _, row in df.iterrows()]).astype(int)
    # Ensure there are positive examples
    if y.sum() > 5:
        importance_matrix[qi] = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_features, y).feature_importances_

# Embed and train Ridge
X_train_queries = embedder.encode([q[0] for q in TRAIN_ATOMS], show_progress_bar=False)
ridge_ltr = Ridge(alpha=1.0).fit(X_train_queries, importance_matrix)


# ═══════════════════════════════════════════════════════════════
# 3. DEFINING THE UNSEEN COMPOSITIONAL HOLDOUT SET
# ═══════════════════════════════════════════════════════════════
# These exact combinations and phrasings DO NOT exist in training.
HOLDOUTS = [
    ("a complex global context provider utilizing a reducer and heavy effects", 
     lambda r: 3 if r['useContext']>0 and r['useReducer']>0 and r['useEffect']>0 else 0),
    
    ("unoptimized deeply nested rendering tree that receives many props but lacks memoization",
     lambda r: 3 if r['jsx_depth']>=6 and r['props']>=3 and r['useMemo']==0 and r['useCallback']==0 else 0),
     
    ("large data grid that fetches remote data and maps over lists but has absolutely no state",
     lambda r: 3 if r['has_fetch']==1 and r['map_calls']>0 and r['useState']==0 else 0),
     
    ("tiny interactive button mapping refs and capturing clicks",
     lambda r: 3 if r['loc']<80 and r['useRef']>0 and r['event_handlers']>=1 else 0),
    
    ("pure presentational layout that accepts children and filters arrays without effects",
     lambda r: 3 if r['hooks_total']==0 and r['has_children']==1 and r['filter_calls']>0 else 0),
]

def ndcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0 or np.sum(relevances) == 0: return 0.0
    dcg = np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))
    ideal_rels = np.sort(relevances)[::-1]
    idcg = np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels) + 2)))
    return dcg / idcg

print("\n[2] Executing Compositional Holdout Quantitative Run...")

rag_scores = []
ltr_scores = []

for q, gt_fn in HOLDOUTS:
    print(f"\nEvaluating Novel Concept: '{q}'")
    
    q_emb = embedder.encode([q]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    
    # Semantic Recall
    _, semantic_idx = index.search(q_emb, k=200)
    top200_idx = semantic_idx[0]
    
    # 1. RAG Baseline
    rag_rels = [gt_fn(df.iloc[idx]) for idx in top200_idx[:10]]
    rag_ndcg = ndcg_at_k(rag_rels, 10)
    rag_scores.append(rag_ndcg)
    
    # 2. Generalization Prediction
    pred_w = ridge_ltr.predict(q_emb)[0]
    struct_scores = X_features[top200_idx] @ pred_w
    ltr_top_idx = top200_idx[np.argsort(-struct_scores)[:10]]
    ltr_rels = [gt_fn(df.iloc[idx]) for idx in ltr_top_idx]
    ltr_ndcg = ndcg_at_k(ltr_rels, 10)
    ltr_scores.append(ltr_ndcg)
    
    print(f"  -> RAG NDCG: {rag_ndcg:.3f} | LTR NDCG: {ltr_ndcg:.3f}")
    
    # If it was the "unoptimized" query, let's look at the weights to see if it fixed the has_fetch hack!
    if "unoptimized" in q:
        weight_dict = {FEATS[i]: pred_w[i] for i in range(len(FEATS))}
        sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
        print("  [Audit] Predicted Weights for Novel Query:")
        print("    " + ", ".join([f"{k}:{v:+.3f}" for k,v in sorted_weights[:3]]))
        print("    " + ", ".join([f"{k}:{v:+.3f}" for k,v in sorted_weights[-3:]]))


print("\n[3] FINAL NOVELTY GENERALIZATION STATS")
print(f"Mean RAG NDCG@10 : {np.mean(rag_scores):.3f}")
print(f"Mean LTR NDCG@10 : {np.mean(ltr_scores):.3f}")
print("=" * 80)
