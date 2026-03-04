import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

queries = [
    # Visual Dissonance Targets
    "A card element used to display a user profile.",
    "A modal dialogue that contains a complex multi-step form and data fetching.",
    "A toast notification manager.",
    "A map component displaying geographical markers.",
    "A purely visual SVG icon wrapper with no interactive logic."
]

print("Loading Omnimodal Index (Graph + Text + Vision)...")
df = pd.read_pickle('vectors_reference.pkl')
index = faiss.read_index('omnimodal_index.faiss')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Since we are querying via TEXT intent only, we pad the non-semantic vectors:
# 64 Graph (Zeros) + 384 Semantic (Actual Embedding) + 512 Vision (Zeros)
zero_graph_feat = np.zeros((1, 64), dtype='float32')
zero_vision_feat = np.zeros((1, 512), dtype='float32')

print("\n" + "="*80)
print("--- OMNIMODAL RAG TEST ---")
for q in queries:
    print(f"\n[QUERY]: '{q}'")
    # In the fusion script: Lexical was weighted 0.6
    lex_feat = encoder.encode([q]).astype('float32') * 0.6
    query_vec = np.hstack((zero_graph_feat, lex_feat, zero_vision_feat))
    
    D, I = index.search(query_vec, 3)
    
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        print(f"  {i+1}. {row['component']} (File: {row['file'].split('/')[-1] if isinstance(row['file'], str) else ''})")
        print(f"     Dist: {D[0][i]:.4f} | Struct: Hooks={row['hooks_total']}, Props={row['props']}, \
Depth={row['jsx_depth']}, JsxElems={row['jsx_elems']}")
