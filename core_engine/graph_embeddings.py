import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse
from sentence_transformers import SentenceTransformer
import faiss

# Load df
df = pd.read_pickle('data/vectors_reference.pkl')
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']

print("Building numerical graph matrix...")
G = nx.Graph()

# Add all components as nodes
node_to_idx = {}
idx_to_node = {}

for idx, row in df.iterrows():
    node = f"C_{idx}"
    node_to_idx[node] = idx
    idx_to_node[idx] = node
    G.add_node(node)

# Add hooks and props as structural bridge nodes
hook_idx_start = len(df)
param_nodes = {}

idx_counter = hook_idx_start
for idx, row in df.iterrows():
    c_node = f"C_{idx}"
    
    # Hooks
    for hook in hook_cols:
        count = row.get(hook, 0)
        if count > 0:
            h_node = f"H_{hook}"
            if h_node not in param_nodes:
                param_nodes[h_node] = idx_counter
                idx_counter += 1
                G.add_node(h_node)
            G.add_edge(c_node, h_node, weight=float(count))
            
    # Props
    if pd.notna(row.get('prop_names')):
        props = [p.strip() for p in str(row['prop_names']).split(';') if p.strip()]
        for prop in props:
            if 1 < len(prop) < 30:
                p_node = f"P_{prop}"
                if p_node not in param_nodes:
                    param_nodes[p_node] = idx_counter
                    idx_counter += 1
                    G.add_node(p_node)
                G.add_edge(c_node, p_node, weight=1.0)

from sknetwork.embedding import SVD

# Convert to adjacency matrix
# In NetworkX, nodes are ordered but let's be explicit
nodes = list(G.nodes())
adjacency = nx.adjacency_matrix(G, nodelist=nodes)
import scipy.sparse
adjacency = scipy.sparse.csr_matrix(adjacency)

# Compute Graph Embedding using SVD
print("Computing Graph Embedding (Structural Topology)...")
svd = SVD(n_components=64)
embeddings = svd.fit_transform(adjacency)

# We only need embeddings for the first N nodes (the actual components, not the hooks/props)
# Find the indices of the component nodes
comp_indices = [i for i, node in enumerate(nodes) if node.startswith('C_')]
graph_embeddings = embeddings[comp_indices].astype('float32')

# Normalize the graph embeddings
norms = np.linalg.norm(graph_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
graph_embeddings = graph_embeddings / norms

print("Encoding text...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
lex_feats = encoder.encode(df['combined_context'].tolist(), show_progress_bar=True).astype('float32')

# Weighting: 20% Graph Topology, 80% Semantic Intent
weighted_graph = graph_embeddings * 0.2
weighted_lex = lex_feats * 0.8

fusion_embeddings = np.hstack((weighted_graph, weighted_lex))

print("Building new GraphRAG Faiss index...")
dim = fusion_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(fusion_embeddings)

faiss.write_index(index, 'data/graphrag_index.faiss')
np.save('data/graph_embeddings.npy', graph_embeddings)
print(f"Index built with {index.ntotal} components. Dimensions: {dim} (64 Graph + 384 Text)")
