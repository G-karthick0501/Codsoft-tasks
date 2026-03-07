"""
ablation_generalization.py
==========================
Tests whether the Learning-to-Rank (LTR) model overfits to its synthetic training set
or if it mathematically generalizes to novel, unseen architectural combinations.

Uses Semantic Embeddings (SentenceTransformer) to encode queries,
mapping to Feature Importances via Ridge Regression.
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
print("  GENERALIZATION TEST: UNSEEN ARCHITECTURAL INTENT")
print("=" * 80)

# 1. LOAD DATA
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

# 2. TRAIN THE SUPERVISED MODEL ON BASE ARCHETYPES
print("[1] Training LTR model on 5 basic archetypes...")
# We use descriptive text so the embedder learns correlations, mapping to synthetic AdaBoost ground truth
TRAIN_QUERIES = [
    ("complex global state provider with multiple contexts", lambda r: min(3, int(r['hooks_total']>=4)+int(r['useContext']>0))),
    ("data fetching component with state and effects", lambda r: min(3, int(r['has_fetch']==1)*2+int(r['useState']>0))),
    ("stateless pure presentational component with no hooks", lambda r: min(3, int(r['hooks_total']==0)*2)),
    ("highly interactive form with event handlers and state", lambda r: min(3, int(r['event_handlers']>=2)+int(r['useState']>0))),
    ("layout wrapper with many children and high jsx depth", lambda r: min(3, int(r['jsx_depth']>=3) + int(r['has_children']>0))),
]

importance_matrix = np.zeros((len(TRAIN_QUERIES), len(FEATS)))
for qi, (q_text, gt_fn) in enumerate(TRAIN_QUERIES):
    y = np.array([gt_fn(row) >= 2 for _, row in df.iterrows()]).astype(int)
    if y.sum() > 0:
        importance_matrix[qi] = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_features, y).feature_importances_

# CRITICAL FIX for Generalization: Encode queries using S-BERT instead of TF-IDF
# This allows the ML model to generalize semantic intent mapped to physical weights.
q_texts = [q[0] for q in TRAIN_QUERIES]
X_train_queries = embedder.encode(q_texts)

# Train Ridge mapping from Semantic Space -> AST Feature Weight Space
ridge_ltr = Ridge(alpha=1.0).fit(X_train_queries, importance_matrix)

# 3. TEST ON UNSEEN NOVEL ARCHITECTURE
print("\n[2] Testing Generalization on Novel Query...")
NOVEL_QUERY = "unoptimized deeply nested rendering tree that receives many props but lacks memoization"

print(f"\n  Query: '{NOVEL_QUERY}'")
print("  (Model has NEVER been explicitly trained on 'unoptimized', 'lacks memoization', or 'many props')\n")

q_novel_emb = embedder.encode([NOVEL_QUERY])
predicted_weights = ridge_ltr.predict(q_novel_emb)[0]

# Display Top 5 Positive and Top 5 Negative Predicted Weights
weight_dict = {FEATS[i]: predicted_weights[i] for i in range(len(FEATS))}
sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)

print("  🧠 PREDICTED AST FEATURE WEIGHTS (ZERO-SHOT INFERENCE):")
print("  Top Drivers:")
for feat, w in sorted_weights[:5]:
    print(f"    + {feat:<15}: {w:+.4f}")

print("\n  Bottom Penalties:")
for feat, w in sorted_weights[-5:]:
    print(f"    - {feat:<15}: {w:+.4f}")

# 4. EXECUTE RERANKING
faiss.normalize_L2(q_novel_emb)
_, semantic_idx = index.search(q_novel_emb, k=200)
recall_idx = semantic_idx[0]

# Rerank mathematically
structural_scores = X_features[recall_idx] @ predicted_weights
reranked_idx = recall_idx[np.argsort(-structural_scores)[:3]]

print("\n🏆 Top 3 Mechanically Verified Results:")
for i, idx in enumerate(reranked_idx):
    row = df.iloc[idx]
    print(f"  {i+1}. {str(row['component'])[:25]:<25}")
    print(f"     AST: jsx_depth={row['jsx_depth']}, props={row['props']}, memo_hooks={row['useMemo']+row['useCallback']}, loc={row['loc']}")

print("\n" + "=" * 80)
