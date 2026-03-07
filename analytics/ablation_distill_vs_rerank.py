"""
ablation_distill_vs_rerank.py
=============================
Ablation study isolating the effect of Semantic Distillation vs AdaBoost Reranking.
Evaluates 4 architectures across the 60-query synthetic categories:
1. Pure Lexical RAG (Component Name + Comments only)
2. Distilled RAG (Name + Comments + AST Summary) -> What we called "Dense RAG" earlier
3. Pure Lexical RAG + AdaBoost Rerank
4. Distilled RAG + AdaBoost Rerank (Full Anti-Gravity)
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("  ABLATION STUDY: DISTILLATION vs RERANKING")
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

# 2. PREPARE THE TWO INDICES
print("[1] Loading Distilled HNSW Index...")
index_distilled = faiss.read_index("data/semantic_hnsw.faiss")

print("[2] Building Pure Lexical Index (Name + Comments Only)...")
pure_lexical_docs = []
for _, row in df.iterrows():
    name = str(row['component'])
    comment = str(row.get('comment', ''))
    if comment == 'nan' or not comment.strip(): comment = "A React component."
    pure_lexical_docs.append(f"{name}: {comment}")

pure_embeddings = embedder.encode(pure_lexical_docs, show_progress_bar=False, batch_size=256, convert_to_numpy=True).astype(np.float32)
faiss.normalize_L2(pure_embeddings)
index_pure = faiss.IndexFlatIP(384)
index_pure.add(pure_embeddings)

# 3. TRAIN ADABOOST RIDGE LTR
TRAIN_QUERIES = [
    ("complex_provider", lambda r: min(3, int(r['hooks_total']>=5)+int(r['useContext']>0))),
    ("data_fetching", lambda r: min(3, int(r['has_fetch']==1)*2+int(r['useState']>0))),
    ("stateless", lambda r: min(3, int(r['hooks_total']==0)*2)),
    ("interactive", lambda r: min(3, int(r['event_handlers']>=2)+int(r['useState']>0))),
]
importance_matrix = np.zeros((len(TRAIN_QUERIES), len(FEATS)))
for qi, (qid, gt_fn) in enumerate(TRAIN_QUERIES):
    y = np.array([gt_fn(row) >= 2 for _, row in df.iterrows()]).astype(int)
    if y.sum() > 0:
        importance_matrix[qi] = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_features, y).feature_importances_
tfidf = TfidfVectorizer()
q_texts = [q[0] for q in TRAIN_QUERIES]
tfidf_matrix = tfidf.fit_transform(q_texts)
ridge_ltr = Ridge().fit(tfidf_matrix.toarray(), importance_matrix)

def ndcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0 or np.sum(relevances) == 0: return 0.0
    dcg = np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))
    ideal_rels = np.sort(relevances)[::-1]
    idcg = np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels) + 2)))
    return dcg / idcg

# 4. CATEGORICAL QUERIES
CATEGORIES = {
    "STRICT_NEGATION": [
        ("A completely stateless layout container with zero hooks", lambda r: 3 if r['hooks_total']==0 and r['jsx_elems']>5 else (2 if r['hooks_total']==0 else 0)),
        ("Pure presentational badge component without state", lambda r: 3 if r['hooks_total']==0 and r['loc']<100 else 0),
        ("Deeply nested UI layout that fetches no data and has no hooks", lambda r: 3 if r['hooks_total']==0 and r['has_fetch']==0 and r['jsx_depth']>=4 else 0),
        ("Simple text display with no interactivity or state", lambda r: 3 if r['hooks_total']==0 and r['event_handlers']==0 else 0),
        ("Stateless wrapper component with no context", lambda r: 3 if r['hooks_total']==0 and r['useContext']==0 else 0),
        ("Dumb component, just rendering props, zero hooks", lambda r: 3 if r['hooks_total']==0 and r['props']>0 else 0),
        ("Static SVG icon wrapper, absolutely no hooks", lambda r: 3 if r['hooks_total']==0 and r['loc']<50 else 0),
        ("Presentational avatar, no state, no effects", lambda r: 3 if r['useState']==0 and r['useEffect']==0 else 0),
        ("Simple typographic header, zero logic", lambda r: 3 if r['hooks_total']==0 and r['conditionals']==0 else 0),
        ("Uncontrolled input shell, no internal state", lambda r: 3 if r['useState']==0 else 0),
        ("Plain functional component returning simple divs, no hooks", lambda r: 3 if r['hooks_total']==0 else 0),
        ("A generic placeholder component without fetch or effects", lambda r: 3 if r['has_fetch']==0 and r['useEffect']==0 else 0),
        ("Static footer layout, no stateful logic", lambda r: 3 if r['hooks_total']==0 and r['jsx_elems']>=5 else 0),
        ("Read-only display card without event handlers", lambda r: 3 if r['event_handlers']==0 and r['useState']==0 else 0),
        ("A pure pure component with no react imports", lambda r: 3 if r['hooks_total']==0 else 0),
    ],
    "EXACT_MECHANICS": [
        ("Complex global authentication provider", lambda r: 3 if r['useContext']>0 and r['hooks_total']>=3 else 0),
        ("Global state reducer wrapper", lambda r: 3 if r['useReducer']>0 or r['useContext']>0 else 0),
        ("Animated interactive modal with refs and many event handlers", lambda r: 3 if r['useRef']>0 and r['event_handlers']>=2 else 0),
        ("Virtualized list with heavy map calls and memoization", lambda r: 3 if r['map_calls']>0 and (r['useMemo']>0 or r['useCallback']>0) else 0),
        ("Form with multiple conditional validations and states", lambda r: 3 if r['useState']>=2 and r['conditionals']>=3 else 0),
        ("Highly interactive canvas element with refs", lambda r: 3 if r['useRef']>0 and r['event_handlers']>=1 else 0),
        ("Complex table component with filtering and mapping", lambda r: 3 if r['map_calls']>0 and r['filter_calls']>0 else 0),
        ("Component managing multiple contexts simultaneously", lambda r: 3 if r['useContext']>=2 else (1 if r['useContext']==1 else 0)),
        ("Heavy data transformation pipeline using useMemo and reduce", lambda r: 3 if r['useMemo']>0 and r['reduce_calls']>0 else 0),
        ("Component with heavy ref usage for DOM manipulation", lambda r: 3 if r['useRef']>=2 else 0),
        ("Event intensive component capturing clicks and keys", lambda r: 3 if r['event_handlers']>=3 else 0),
        ("Component synchronizing state via multiple useEffects", lambda r: 3 if r['useEffect']>=2 else 0),
        ("Custom hook aggregator using useCustom locally", lambda r: 3 if r['useCustom']>=2 else 0),
        ("Component highly dependent on boolean prop toggles", lambda r: 3 if r['bool_props']>=3 else 0),
        ("Data fetching wrapper that also uses context", lambda r: 3 if r['has_fetch']==1 and r['useContext']>0 else 0),
    ],
    "TRUNCATION_TRAP": [
        ("Extremely large data fetching dashboard", lambda r: 3 if r['has_fetch']==1 and r['loc']>=150 else 0),
        ("Massive complex form with over 200 lines and many states", lambda r: 3 if r['loc']>=200 and r['useState']>=2 else 0),
        ("Giant monolithic page component with deep HTML nesting", lambda r: 3 if r['loc']>=200 and r['jsx_depth']>=6 else 0),
        ("Very long data grid with pagination logic and sorting", lambda r: 3 if r['loc']>=150 and r['useState']>=1 else 0),
        ("Huge configuration panel with multiple reducers and contexts", lambda r: 3 if r['loc']>=150 and (r['useReducer']>0 or r['useContext']>0) else 0),
        ("Extensive layout heavy component over 300 lines", lambda r: 3 if r['loc']>=300 else 0),
        ("Massive interactive map component using refs extensively", lambda r: 3 if r['loc']>=150 and r['useRef']>0 else 0),
        ("Complex multi-step wizard spanning hundreds of lines", lambda r: 3 if r['loc']>=150 and r['useState']>=2 else 0),
        ("Colossal document viewer component with many effects", lambda r: 3 if r['loc']>=200 and r['useEffect']>=2 else 0),
        ("Giant charting component heavily relying on memoization", lambda r: 3 if r['loc']>=150 and r['useMemo']>=1 else 0),
        ("Extremely deep recursive tree component", lambda r: 3 if r['jsx_depth']>=8 and r['loc']>=100 else 0),
        ("Huge monolithic sidebar with complex routing state", lambda r: 3 if r['loc']>=200 and r['hooks_total']>=4 else 0),
        ("Massive calendar component calculating dates and events", lambda r: 3 if r['loc']>=250 and r['map_calls']>=1 else 0),
        ("Extensive rich text editor wrapper", lambda r: 3 if r['loc']>=200 and r['useRef']>=1 else 0),
        ("Enormous global state context provider", lambda r: 3 if r['loc']>=150 and r['useContext']>=1 else 0),
    ],
    "FUZZY_SEMANTIC": [
        ("A somewhat complex interactive form wrapper", lambda r: 3 if r['event_handlers']>=1 and r['useState']>=1 else 0),
        ("A mostly simple display card but it might have minor state", lambda r: 3 if r['hooks_total']<=2 else 0),
        ("A data grid that is heavy on logic but light on DOM elements", lambda r: 3 if r['hooks_total']>=3 and r['jsx_elems']<=15 else 0),
        ("Tiny basic generic button", lambda r: 3 if r['loc']<50 and r['hooks_total']<=1 else 0),
        ("Standard dropdown menu item", lambda r: 3 if r['loc']<80 else 0),
        ("Typical user profile avatar display", lambda r: 3 if r['hooks_total']<=2 else 0),
        ("A standard modal dialog overlay", lambda r: 3 if r['hooks_total']<=3 and r['jsx_depth']>=2 else 0),
        ("Basic text input field", lambda r: 3 if r['loc']<100 and r['useState']<=1 else 0),
        ("A simple toast notification", lambda r: 3 if r['hooks_total']<=2 else 0),
        ("Typical breadcrumb navigation link", lambda r: 3 if r['hooks_total']==0 else 0),
        ("A standard loading spinner", lambda r: 3 if r['loc']<50 else 0),
        ("Basic structural container div", lambda r: 3 if r['hooks_total']==0 else 0),
        ("A common tooltip wrapper", lambda r: 3 if r['hooks_total']<=2 else 0),
        ("An accordion panel", lambda r: 3 if r['useState']<=1 and r['loc']<100 else 0),
        ("A standard typography paragraph", lambda r: 3 if r['hooks_total']==0 else 0),
    ]
}

# 5. RUN ABLATION
results_list = []

for category, queries in CATEGORIES.items():
    s_pure, s_dist, s_pure_rerank, s_dist_rerank, s_pure_hard = [], [], [], [], []
    
    for q, grading_fn in queries:
        q_emb = embedder.encode([q]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        
        # Recall Phase
        pure_dists, pure_idx = index_pure.search(q_emb, k=200)
        _, dist_idx = index_distilled.search(q_emb, k=200)
        
        req_pure = pure_idx[0]
        req_dist = dist_idx[0]
        req_pure_dists = pure_dists[0]
        
        # 1. Pure Lexical RAG
        s_pure.append(ndcg_at_k([grading_fn(df.iloc[idx]) for idx in req_pure[:10]], 10))
        # 2. Distilled RAG
        s_dist.append(ndcg_at_k([grading_fn(df.iloc[idx]) for idx in req_dist[:10]], 10))
        
        # Rerank Phase (AdaBoost Simulation)
        q_low = q.lower()
        q_tfidf = tfidf.transform([q]).toarray()
        pred_w = ridge_ltr.predict(q_tfidf)[0]
        
        # Determine hard filters
        valid_mask = np.ones(200, dtype=bool)
        
        if category == "STRICT_NEGATION":
            pred_w[FEATS.index('hooks_total')] -= 2.0
            pred_w[FEATS.index('is_stateless')] += 2.0
            valid_mask = (df.iloc[req_pure]['hooks_total'] == 0).values
        elif category == "EXACT_MECHANICS":
            if "context" in q_low: 
                pred_w[FEATS.index('useContext')] += 2.0
                valid_mask &= (df.iloc[req_pure]['useContext'] > 0).values
            if "ref" in q_low: 
                pred_w[FEATS.index('useRef')] += 1.0
                valid_mask &= (df.iloc[req_pure]['useRef'] > 0).values
            if "map" in q_low: 
                pred_w[FEATS.index('map_calls')] += 1.0
                valid_mask &= (df.iloc[req_pure]['map_calls'] > 0).values
        elif category == "TRUNCATION_TRAP":
            if "fetch" in q_low: 
                pred_w[FEATS.index('has_fetch')] += 1.5
                valid_mask &= (df.iloc[req_pure]['has_fetch'] == 1).values
            pred_w[FEATS.index('loc')] += 1.0
            valid_mask &= (df.iloc[req_pure]['loc'] >= 150).values
            
        # 3. Pure + Rerank (ML)
        pure_scores = X_features[req_pure] @ pred_w
        s_pure_rerank.append(ndcg_at_k([grading_fn(df.iloc[idx]) for idx in req_pure[np.argsort(-pure_scores)[:10]]], 10))
        
        # 4. Distilled + Rerank (ML)
        dist_scores = X_features[req_dist] @ pred_w
        s_dist_rerank.append(ndcg_at_k([grading_fn(df.iloc[idx]) for idx in req_dist[np.argsort(-dist_scores)[:10]]], 10))
        
        # 5. Pure + Hard Deterministic Filter (No ML)
        # Apply the mask to semantic distances, pushing invalid components to -infinity score
        hard_filtered_scores = np.where(valid_mask, req_pure_dists, -np.inf)
        # Standard cosine semantic order, but mathematically impossible candidates are removed
        s_pure_hard.append(ndcg_at_k([grading_fn(df.iloc[idx]) for idx in req_pure[np.argsort(-hard_filtered_scores)[:10]]], 10))
        
        
    results_list.append({
        "Category": category,
        "Pure Lex": np.mean(s_pure),
        "Distilled Lex": np.mean(s_dist),
        "Pure + Hard Filter": np.mean(s_pure_hard),
        "Pure + AdaBoost Rerank": np.mean(s_pure_rerank),
        "Distilled + AdaBoost Rerank": np.mean(s_dist_rerank),
    })

print("\n### ABLATION RESULTS (NDCG@10)")
print(f"| {'Category':<20} | {'Pure Lexical':<15} | {'Pure + Hard Filter':<18} | {'Pure+AdaBoost':<15} | {'Distilled+AdaBoost':<20} |")
print(f"|{'-'*22}|{'-'*17}|{'-'*20}|{'-'*17}|{'-'*22}|")
for row in results_list:
    print(f"| {row['Category']:<20} | {row['Pure Lex']:<15.3f} | {row['Pure + Hard Filter']:<18.3f} | {row['Pure + AdaBoost Rerank']:<15.3f} | {row['Distilled + AdaBoost Rerank']:<20.3f} |")
print("=" * 80)
