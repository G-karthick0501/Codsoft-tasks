import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys

def search(query, k=5):
    print("Loading data...")
    df = pd.read_pickle('vectors_reference.pkl')
    index = faiss.read_index('data_index.faiss')
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"\nQuerying: '{query}'")
    lex_feat = encoder.encode([query]).astype('float32') * 0.8
    # Dummy scaled structural features (mean values after StandardScaler are 0)
    struct_feat = np.zeros((1, 11), dtype='float32')
    
    query_vec = np.hstack((struct_feat, lex_feat))
    
    D, I = index.search(query_vec, k)
    
    print(f"\nTop {k} results:")
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        print(f"\n{i+1}. Component: {row['component']} (File: {row['file']})")
        print(f"   Dist: {D[0][i]:.4f}")
        print(f"   Comment: {row['comment']}")
        print(f"   Hooks: {row['hooks_total']}, Props: {row['props']}, Jsx Depth: {row['jsx_depth']}")

if __name__ == '__main__':
    search_query = sys.argv[1] if len(sys.argv) > 1 else "A button with click handler"
    search(search_query)
