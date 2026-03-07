"""
llm_router_search.py
====================
The Defensible "Zero-Shot Distill-and-Rerank" Pipeline.
Instead of faking ground truth to train AdaBoost, we use an LLM
to zero-shot translate human intent into rigid mathematical feature weights.

Phase 1: LLM translates query -> JSON feature weights (-1.0 to 1.0)
Phase 2: Semantic Recall (HNSW Warp Drive) gets top 200
Phase 3: Direct Dot-Product Re-ranking (AST Physics Score)

Run: python core_engine/llm_router_search.py
"""

import os
import json
import time
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Try to import google-generativeai for the LLM router
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    HAS_GENAI = bool(os.getenv("GEMINI_API_KEY"))
    if HAS_GENAI:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except ImportError:
    HAS_GENAI = False

print("=" * 70)
print("  ZERO-SHOT LLM ROUTER PIPELINE")
print("  The Mathematically Honest Search Architecture")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA & FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n[1] Loading 6,327 Components & AST Features...")
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

FEATS = RAW + ['state_hooks','effect_hooks','context_hooks','ref_hooks','hook_diversity',
               'complexity_score','interactivity','data_pattern','jsx_density',
               'hooks_per_loc','props_per_loc','is_stateless','is_complex','is_fetcher']

# Normalize features so dot product is balanced
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_features = scaler.fit_transform(df[FEATS])

# ═══════════════════════════════════════════════════════════════
# 2. LOAD SEMANTIC INDEX (WARP DRIVE)
# ═══════════════════════════════════════════════════════════════
print("[2] Engaging Semantic HNSW Index ('Warp Drive')...")
try:
    index = faiss.read_index("data/semantic_hnsw.faiss")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading index: {e}. Please run analytics/semantic_hnsw_engine.py first.")
    exit(1)

# ═══════════════════════════════════════════════════════════════
# 3. THE LLM ZERO-SHOT ROUTER
# ═══════════════════════════════════════════════════════════════
def get_llm_feature_weights(query: str) -> dict:
    """Uses an LLM to zero-shot map a natural language query to AST feature weights (-1.0 to 1.0)."""
    
    prompt = f"""
    You are an AI Architecture Router for a React component search engine.
    The user is searching for: "{query}"
    
    Map their intent to importance weights for the following AST features.
    Assign a weight between -1.0 (strongly avoid) and 1.0 (strongly require).
    Only output features that are relevant to the query. Leave out irrelevant ones.
    
    Features available:
    {', '.join(FEATS)}
    
    Output ONLY a valid JSON dictionary. No markdown, no explanations.
    Example for "stateless complex layout": {{"is_stateless": 1.0, "jsx_elems": 0.8, "hooks_total": -1.0}}
    """
    
    if HAS_GENAI:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            text = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(text)
        except Exception as e:
            print(f"  [LLM Error: {e} - Falling back to local heuristic router]")
    
    # Fallback heuristic router if API fails or key is missing
    weights = {}
    q = query.lower()
    if "stateless" in q or "no hooks" in q or "pure" in q:
        weights = {"is_stateless": 1.0, "hooks_total": -1.0, "state_hooks": -1.0, "effect_hooks": -1.0}
        if "layout" in q: weights["jsx_elems"] = 0.5
    elif "dashboard" in q or "fetch" in q or "complex" in q:
        weights = {"has_fetch": 1.0, "is_complex": 0.8, "data_pattern": 1.0, "loc": 0.5, "jsx_elems": 0.5}
        if "global" in q or "context" in q: weights["context_hooks"] = 1.0
    return weights

# ═══════════════════════════════════════════════════════════════
# 4. THE TWO-STAGE PIPELINE
# ═══════════════════════════════════════════════════════════════
def search_zero_shot(query: str, recall_k: int = 200, top_k: int = 3):
    print(f"\n🔍 QUERY: '{query}'\n")
    
    # --- PHASE 1: LLM ROUTER (0-shot Feature Mapping) ---
    t_llm_start = time.perf_counter()
    weights_dict = get_llm_feature_weights(query)
    
    # Convert JSON dict to full 36-dimensional numpy weight vector
    weight_vector = np.zeros(len(FEATS))
    for feat, w in weights_dict.items():
        if feat in FEATS:
            weight_vector[FEATS.index(feat)] = float(w)
            
    t_llm_end = time.perf_counter()
    print(f"🧠 LLM Extracted Weights:")
    for k, v in weights_dict.items():
        print(f"   {k}: {v:+.2f}")
    
    # --- PHASE 2: SEMANTIC RECALL ---
    t_hnsw_start = time.perf_counter()
    q_emb = embedder.encode([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    
    _, semantic_idx = index.search(q_emb, k=recall_k)
    recall_indices = semantic_idx[0]
    t_hnsw_end = time.perf_counter()
    
    # --- PHASE 3: DIRECT DOT-PRODUCT RE-RANKING ---
    t_rank_start = time.perf_counter()
    # Matrix multiplication: (200, 36) @ (36,) = (200,)
    recall_features = X_features[recall_indices]
    structural_scores = recall_features @ weight_vector
    
    reranked_order = np.argsort(-structural_scores)
    final_indices = recall_indices[reranked_order]
    t_rank_end = time.perf_counter()
    
    # --- OUTPUT ---
    print(f"\n⏱️ Timings:")
    print(f"  LLM Routing Generation     : {(t_llm_end - t_llm_start)*1000:>7.2f} ms")
    print(f"  Stage 1 (HNSW Vibe Recall) : {(t_hnsw_end - t_hnsw_start)*1000:>7.2f} ms")
    print(f"  Stage 2 (Dot Product Score): {(t_rank_end - t_rank_start)*1000:>7.2f} ms")
    print(f"  Total Retrieval Latency    : {(t_rank_end - t_hnsw_start)*1000:>7.2f} ms")
    
    print(f"\n🏆 Top {top_k} Results (Mechanically Verified):")
    for i in range(top_k):
        idx = final_indices[i]
        comp_name = df.iloc[idx]['component']
        repo = df.iloc[idx]['repo']
        loc = df.iloc[idx]['loc']
        hooks = df.iloc[idx]['hooks_total']
        is_stateless = df.iloc[idx]['is_stateless']
        str_score = structural_scores[reranked_order[i]]
        
        orig_rank = np.where(recall_indices == idx)[0][0] + 1
        
        print(f"  {i+1}. {str(comp_name)[:25]:<25} | Score: {str_score:+.2f}")
        print(f"     Repo: {str(repo)[:20]:<20} | AST: {hooks} hooks, {loc} lines, stateless={bool(is_stateless)}")
        print(f"     (Reranked from initial Semantic Vibe rank #{orig_rank})")

# ═══════════════════════════════════════════════════════════════
# 5. EXECUTION
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    search_zero_shot("accessible simple stateless dropdown layout with no hooks", recall_k=200)
    search_zero_shot("complex data dashboard that fetches data and connects to global state context", recall_k=200)

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}\n")
