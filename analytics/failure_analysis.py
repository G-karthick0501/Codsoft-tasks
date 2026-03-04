import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

queries = [
    "A component that fetches user data from an API and handles loading states.",
    "A simple button with an icon and label.",
    "A sidebar navigation layout with multiple nested sections.",
    "A date picker calendar with range selection.",
    "A high-performance virtualized list or table.",
    "A dark mode toggle switch.",
    "A rich text editor with toolbars.",
    "A complex multi-step signup form.",
    "A toast notification manager.",
    "A responsive image gallery with lightbox."
]

print("Loading index and models...")
df = pd.read_pickle('data/vectors_reference.pkl')
index = faiss.read_index('data/graphrag_index.faiss')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Struct feat is dummy because what matters is that the texts get matched.
# The index already contains [scaled_struct*0.2, lex_feat*0.8]
struct_feat = np.zeros((1, 11), dtype='float32')

for q in queries:
    print("\n" + "="*80)
    print(f"QUERY: '{q}'")
    lex_feat = encoder.encode([q]).astype('float32') * 0.8
    query_vec = np.hstack((struct_feat, lex_feat))
    
    D, I = index.search(query_vec, 3)
    
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        print(f"  {i+1}. {row['component']} | Dist: {D[0][i]:.4f}")
        print(f"     File: {row['file']}")
        print(f"     Comment: {row['comment']}")
        print(f"     Struct: Hooks={row['hooks_total']}, Props={row['props']}, Depth={row['jsx_depth']}, JsxElems={row['jsx_elems']}")
