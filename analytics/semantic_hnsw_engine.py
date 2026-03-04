"""
semantic_hnsw_engine.py
=======================
Implements the "Warp Drive" Semantic Engine using HNSW and all-MiniLM-L6-v2.
This proves out the semantic side of the architecture, encoding component intent
and searching it via a sub-millisecond graph index.

Run: python analytics/semantic_hnsw_engine.py
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  SEMANTIC HNSW ENGINE (The Warp Drive)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA & SYNTHESIZE SEMANTIC DOCUMENTS
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv('data/master2.csv')
df = df[df['component'].str.strip().ne('')].drop_duplicates(subset=['repo','file']).reset_index(drop=True)

print(f"\n[1] Synthesizing Semantic Documents for {len(df):,} components...")
t0 = time.time()

# We construct a "Semantic Document" that acts as the target for the embedding.
# It includes the component name, its natural language comments, and its AST properties
# translated into plain English so the NLP model can "read" its structure.
semantic_docs = []
for _, row in df.iterrows():
    name = str(row['component'])
    comment = str(row.get('comment', ''))
    if comment == 'nan' or not comment.strip():
        comment = "A React component."
        
    # Translate AST to English
    ast_desc = []
    if row.get('has_fetch', 0) == 1:
        ast_desc.append("fetches remote data")
    if row.get('useContext', 0) > 0 or row.get('useReducer', 0) > 0:
        ast_desc.append("manages global state")
    if row.get('hooks_total', 0) == 0:
        ast_desc.append("is a pure stateless layout component")
    elif row.get('hooks_total', 0) >= 5:
        ast_desc.append("contains complex state logic")
    
    ast_str = " It " + " and ".join(ast_desc) + "." if ast_desc else ""
    doc = f"{name}: {comment}{ast_str}"
    semantic_docs.append(doc)

df['semantic_doc'] = semantic_docs
print(f"    Done. (Example: '{semantic_docs[0]}')")

# ═══════════════════════════════════════════════════════════════
# 2. GENERATE 384D SEMANTIC EMBEDDINGS
# ═══════════════════════════════════════════════════════════════
print(f"\n[2] Generating 384D Embeddings (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
t0 = time.time()
embeddings = model.encode(semantic_docs, show_progress_bar=True, batch_size=128, convert_to_numpy=True)
embeddings = embeddings.astype(np.float32)
faiss.normalize_L2(embeddings)  # L2 normalize for cosine similarity
print(f"    Encoded in {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# 3. BUILD HNSW INDEX (The "Warp Drive")
# ═══════════════════════════════════════════════════════════════
print("\n[3] Building HNSW Index for Million-Scale Sub-MS Search...")
d = embeddings.shape[1]
M = 32              # Number of connections per node
ef_construction = 200 # Build time search depth

t0 = time.time()
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.add(embeddings)
index.hnsw.efSearch = 64  # Search time depth

faiss.write_index(index, "data/semantic_hnsw.faiss")
print(f"    HNSW Index Built & Saved in {time.time() - t0:.2f}s")
print(f"    Total Nodes: {index.ntotal} | Edges per node: {M}")

# ═══════════════════════════════════════════════════════════════
# 4. BENCHMARK: SEMANTIC INTENT QUERIES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TESTING SEMANTIC QUERIES (Meaning vs Mechanism)")
print(f"{'='*70}")

SEMANTIC_QUERIES = [
    "dark mode theme toggle button",
    "accessible animated dropdown menu",
    "skeleton loading screen fallback",
    "tabular data grid with pagination",
    "authentication login form with validation"
]

for q in SEMANTIC_QUERIES:
    print(f"\n🔍 Query: '{q}'")
    q_emb = model.encode([q]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    
    t0 = time.time()
    distances, indices = index.search(q_emb, k=3)
    elapsed_ms = (time.time() - t0) * 1000
    
    print(f"  [Search Latency: {elapsed_ms:.2f} ms]")
    for i, idx in enumerate(indices[0]):
        dist = distances[0][i]  # Inner product since normalized
        comp_name = df.iloc[idx]['component']
        repo = df.iloc[idx]['repo']
        print(f"  {i+1}. {str(comp_name)[:25]:<25} (sim: {dist:.3f}) | Repo: {str(repo)[:20]}")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}\n")
