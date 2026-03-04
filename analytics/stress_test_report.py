import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Adversarial Queries designed to expose model breaking points
adversarial_queries = [
    # 1. Architectural Mirror: Lexical="simple", but we want structural mismatch
    "A component with 10 hooks that renders nothing but a div.",
    # 2. Common Name, Unique Shape: Should we get the button, or the provider?
    "A simple button.",
    # 3. High Logic, Internal State (Graph heavy expectation)
    "A data grid relying heavily on internal state and refs, without any children.",
    # 4. Pure UI Leaf Node (Text heavy expectation)
    "A purely visual SVG icon wrapper with no interactive logic.",
    # 5. Hybrid chimera
    "A modal dialogue that contains a complex multi-step form and data fetching.",
    # 6. Topological Drift trap (Testing 'useTheme' / 'useId' blindness)
    "A layout wrapper that just injects theme context and an ID.",
    # 7. Semantic Ambiguity (Visual vs Layout)
    "A card element used to display a user profile.",
    # 8. Provider Blindness (Testing if it returns the Provider or the Leaf)
    "A global notification provider that wraps the app.",
    # 9. Dead-code Aliasing (looking for test/story files)
    "A test fixture for a dropdown menu.",
    # 10. The GraphRAG specific trap: Graph expects forms, text says 'map'
    "A map component displaying geographical markers."
]

print("Loading dataset and models...")
df = pd.read_pickle('data/vectors_reference.pkl')
index = faiss.read_index('data/graphrag_index.faiss')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Recompute graph embeddings from the file to measure "conflict"
import networkx as nx
from sknetwork.embedding import SVD
import scipy.sparse

print("Rebuilding graph for conflict analysis...")
G = nx.Graph()
nodes_list = []
comp_to_idx = {}
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']
idx_counter = len(df)
param_nodes = {}

for idx, row in df.iterrows():
    c_node = f"C_{idx}"
    comp_to_idx[idx] = c_node
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

print("\n" + "="*80)
print("--- STRESS TEST REPORT (Graph vs Text Conflict Analysis) ---")

for q in adversarial_queries:
    print(f"\n[QUERY]: '{q}'")
    lex_feat = encoder.encode([q]).astype('float32') * 0.8
    # Baseline query (Neutral Graph)
    zero_graph = np.zeros((1, 64), dtype='float32')
    query_vec = np.hstack((zero_graph, lex_feat))
    
    D, I = index.search(query_vec, 3)
    
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        
        # Calculate semantic similarity to the text query
        query_lex_base = lex_feat.flatten() / 0.8 # Unweight
        doc_lex_base = encoder.encode([row['combined_context']]).astype('float32').flatten()
        semantic_sim = 1 - cosine(query_lex_base, doc_lex_base)
        
        # Graph magnitude vector (How "strong" of a graph identity does this component have?)
        doc_graph_mag = np.linalg.norm(graph_embeddings[idx])
        
        # Print Diagnostics
        print(f"  {i+1}. {row['component']} (File: {row['file'].split('/')[-1]})")
        print(f"     Dist: {D[0][i]:.4f} | Semantic Sim: {semantic_sim:.3f} | Graph Magnitude: {doc_graph_mag:.3f}")
        print(f"     Struct: Hooks={row['hooks_total']}, Props={row['props']}, Depth={row['jsx_depth']}, JsxElems={row['jsx_elems']}")
        
        # Conflict Detector
        if semantic_sim > 0.6 and row['hooks_total'] > 5 and 'simple' in q:
            print("     ⚠️ CONFLICT WARNING: High text match, but structural graph is too heavy for 'simple'.")
        elif semantic_sim < 0.4 and D[0][i] < 0.7:
            print("     ⚠️ GRAPH DOMINANCE: Text match is weak, but Graph similarity forced this to the top.")
        elif row['hooks_total'] == 0 and 'logic' in q.lower():
            print("     ⚠️ STRUCTURAL MISS: Query asked for logic, but Graph returned a leaf/presentation node.")
