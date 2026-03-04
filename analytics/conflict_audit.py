import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from sknetwork.embedding import SVD
import scipy.sparse
import scipy.spatial.distance
import faiss
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("--- 1. LOADING DATA AND RE-COMPUTING EMBEDDINGS ---")
df = pd.read_pickle('data/vectors_reference.pkl')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Rebuild Raw Topological Graph (64D)
print("Building the Topological SVD Graph...")
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

# Normalize graph
norms = np.linalg.norm(graph_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
graph_embeddings = graph_embeddings / norms

print("Encoding text semantics...")
lex_feats = encoder.encode(df['combined_context'].tolist(), show_progress_bar=False).astype('float32')

# Normalize text for cosine distance comparison
text_norms = np.linalg.norm(lex_feats, axis=1, keepdims=True)
text_norms[text_norms == 0] = 1
lex_feats = lex_feats / text_norms

print("\n" + "="*60)
print("--- 2. RUNNING THE CONFLICT AUDIT ---")
# To find conflicts, we match every component to its mathematically closest "Semantic Twin" 
# (the component that sounds identical in text), and then measure how far apart they are 
# in Topological Graph space.

index_text = faiss.IndexFlatIP(384) # Inner product for normalized vectors = cosine similarity
index_text.add(lex_feats)
# Get top 2 (1st is itself, 2nd is the nearest semantic twin)
D_text, I_text = index_text.search(lex_feats, 2)

conflicts = []
for i in range(len(df)):
    neighbor_idx = I_text[i][1]
    
    # 1. Semantic Distance (How similarly they are named/commented)
    t_dist = scipy.spatial.distance.cosine(lex_feats[i], lex_feats[neighbor_idx])
    
    # 2. Graph Distance (How different their actual code architecture is)
    g_dist = scipy.spatial.distance.cosine(graph_embeddings[i], graph_embeddings[neighbor_idx])
    
    # We are looking for "Liars": High semantic match (low text distance) 
    # but completely opposite architecture (high graph distance).
    conflict_score = g_dist - t_dist
    
    conflicts.append({
        'score': conflict_score,
        'compA_idx': i,
        'compB_idx': neighbor_idx,
        't_dist': t_dist,
        'g_dist': g_dist
    })

# Sort by the largest conflict (Largest gap between what it says it is, and what its structure actually is)
conflicts.sort(key=lambda x: x['score'], reverse=True)

print("--- THE CONFLICT AUDIT: TOP 10 'TROUBLEMAKERS' ---")
print("These are components that perfectly expose the flaw in standard text-only RAG.\n")

for rank, item in enumerate(conflicts[:10]):
    compA = df.iloc[item['compA_idx']]
    compB = df.iloc[item['compB_idx']]
    
    print(f"{rank+1}. CONFLICT SCORE: +{item['score']:.4f} (Text Dist: {item['t_dist']:.4f}, Graph Dist: {item['g_dist']:.4f})")
    print(f"   [Semantic Twin A]: {compA['component']} (File: {compA['file'].split('/')[-1] if isinstance(compA['file'], str) else 'N/A'})")
    print(f"       -> Architecture: {compA['hooks_total']} Hooks | {compA['jsx_depth']} JSX Depth | {compA['props']} Props")
    print(f"   [Semantic Twin B]: {compB['component']} (File: {compB['file'].split('/')[-1] if isinstance(compB['file'], str) else 'N/A'})")
    print(f"       -> Architecture: {compB['hooks_total']} Hooks | {compB['jsx_depth']} JSX Depth | {compB['props']} Props")
    print("-" * 60)
