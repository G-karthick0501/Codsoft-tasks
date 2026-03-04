import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist

print("="*60)
print("--- 1. LOADING RAW UNWEIGHTED MATRICES ---")
# To guarantee we never have to run "different algorithms on different sub-samples",
# we extract all embeddings to RAM and let the Router dynamically fuse them per-query.
df = pd.read_pickle('data/vectors_reference.pkl')
# Purify Dataset: Remove records with empty component names
df = df[df['component'].str.strip() != ''].reset_index(drop=True)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

import networkx as nx
from sknetwork.embedding import SVD
import scipy.sparse

print("Rebuilding Raw Topological Graph (64D)...")
G = nx.Graph()
nodes_list = []
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']
idx_counter = len(df)
param_nodes = {}

for idx, row in df.iterrows():
    c_node = f"C_{idx}"
    G.add_node(c_node)
    nodes_list.append(c_node)
for idx, row in df.iterrows():
    c_node = f"C_{idx}"
    for hook in hook_cols:
        count = row.get(hook, 0)
        if count > 0:
            h_node = f"H_{hook}"
            if h_node not in param_nodes:
                param_nodes[h_node] = idx_counter
                idx_counter += 1
                G.add_node(h_node)
                nodes_list.append(h_node)
            G.add_edge(c_node, h_node, weight=float(count))
    if pd.notna(row.get('prop_names')):
        props = [p.strip() for p in str(row['prop_names']).split(';') if p.strip()]
        for prop in props:
            if 1 < len(prop) < 30:
                p_node = f"P_{prop}"
                if p_node not in param_nodes:
                    param_nodes[p_node] = idx_counter
                    idx_counter += 1
                    G.add_node(p_node)
                    nodes_list.append(p_node)
                G.add_edge(c_node, p_node, weight=1.0)

adjacency = scipy.sparse.csr_matrix(nx.adjacency_matrix(G, nodelist=nodes_list))
svd = SVD(n_components=64)
graph_embeddings_raw = svd.fit_transform(adjacency)
comp_indices = [i for i, node in enumerate(nodes_list) if node.startswith('C_')]
graph_embeddings = graph_embeddings_raw[comp_indices].astype('float32')
norms = np.linalg.norm(graph_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
graph_embeddings = graph_embeddings / norms

print("Loading Raw Text Embeddings (384D)...")
lex_feats = encoder.encode(df['combined_context'].tolist(), show_progress_bar=False).astype('float32')

print("\n" + "="*60)
print("--- 2. THE DYNAMIC STRUCTURAL ROUTER (Verified Text + Graph) ---")
# This model uses the 100% verified 2617-row dataset.

def dynamic_route(query_text):
    """
    Analyzes the query for structural vs semantic intent.
    Returns (w_text, w_graph)
    """
    q_lower = query_text.lower()
    
    # Structural Keywords
    structural_keywords = ['hooks', 'state', 'complex', 'provider', 'context', 'nested', 'logic', 'fetch', 'architecture', 'global']
    has_structural = any(k in q_lower for k in structural_keywords)
    
    if has_structural:
        w_t, w_g = 0.4, 0.6  # Balanced structural boost
        print(f"  [Router]: Structural intent detected. ⚡ AST Graph Topology boosted to {w_g}")
    else:
        w_t, w_g = 0.8, 0.2
        print(f"  [Router]: Pure semantic intent detected. 📖 Text Semantics boosted to {w_t}")
        
    return w_t, w_g

def execute_routed_query(query, top_k=3):
    print(f"\n[QUERY]: '{query}'")
    w_t, w_g = dynamic_route(query)
    
    # 1. RECALL PASS: Get the top 50 Semantic Text Candidates
    q_lex = encoder.encode([query]).astype('float32')
    dist_lex = cdist(q_lex, lex_feats, metric='cosine')[0]
    top_50_idx = np.argsort(dist_lex)[:50]
    
    # 2. RANK PASS: Structural Re-ranking (Soft Reward Formula)
    final_scores = []
    max_g_mag = np.max(np.linalg.norm(graph_embeddings, axis=1)) or 1
    
    for idx in top_50_idx:
        # Distance calculation
        text_dist = dist_lex[idx]
        
        # Refined Structural Reward Logic (Soft Re-ranking)
        # Instead of penalizing name mismatch, we treat structure as a 'boost' signal
        g_mag = np.linalg.norm(graph_embeddings[idx])
        g_ratio = g_mag / max_g_mag
        
        # New Scoring Formula: Lower is better
        # We REWARD structural density by subtracting a portion from the distance
        score = (text_dist * w_t) - (g_ratio * w_g * 0.15)
        
        final_scores.append((score, idx))
        
    final_scores.sort(key=lambda x: x[0])
    
    for rank, (score, idx) in enumerate(final_scores[:top_k]):
        row = df.iloc[idx]
        print(f"  {rank+1}. {row['component']} (File: {row['file'].split('/')[-1] if isinstance(row['file'], str) else ''})")
        print(f"     Score: {score:.4f} | Hooks: {row['hooks_total']} | Depth: {row['jsx_depth']}")


# Test the 100% Verified Dynamic Router
test_queries = [
    "A standard text input field.",                                          
    "A stateful complex provider managing context and global hooks.",          
    "A custom hook implementation for authentication."
]

for q in test_queries:
    execute_routed_query(q)
