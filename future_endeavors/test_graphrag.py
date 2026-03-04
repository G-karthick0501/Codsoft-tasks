import pandas as pd
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from scipy import sparse
from sknetwork.embedding import SVD

queries = [
    "A component that fetches user data from an API and handles loading states.",
    "A high-performance virtualized list or table.",
    "A complex multi-step signup form.",
    "A toast notification manager.",
    "A sidebar navigation layout with multiple nested sections."
]

print("Loading index and dataset...")
df = pd.read_pickle('vectors_reference.pkl')
index = faiss.read_index('graphrag_index.faiss')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Since our Graph embeddings represent the topological position of components, 
# a pure Semantic Query doesn't natively map to the graph edge structure.
# To perform a "GraphRAG" query, we assume a neutral/average topological position 
# (represented by a zero vector for the 64 graph dimensions) and let Semantic Intent drive,
# while the topological distance acts as the secondary ranker.
zero_graph_feat = np.zeros((1, 64), dtype='float32')

print("\n--- Comparing GraphRAG Retrieval ---")
for q in queries:
    print(f"\nQUERY: '{q}'")
    lex_feat = encoder.encode([q]).astype('float32') * 0.8
    query_vec = np.hstack((zero_graph_feat, lex_feat))
    
    D, I = index.search(query_vec, 3)
    
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        print(f"  {i+1}. {row['component']} | Dist: {D[0][i]:.4f}")
        print(f"     File: {row['file']}")
        print(f"     Comment: {row['comment']}")
        print(f"     Struct: Hooks={row['hooks_total']}, Props={row['props']}, Depth={row['jsx_depth']}, JsxElems={row['jsx_elems']}")
