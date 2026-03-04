"""
rebuild_index.py — Full pipeline rebuild from master2.csv
==========================================================
Reads data/master2.csv → encodes text → builds graph SVD →
fuses embeddings → writes data/vectors_reference.pkl + data/graphrag_index.faiss

Run from project root:
    python rebuild_index.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sknetwork.embedding import SVD

# ─── 1. LOAD & PURIFY ─────────────────────────────────────────────────────────
print("=" * 55)
print("[1/5] Loading data/master2.csv...")
df = pd.read_csv('data/master2.csv')

# Purify
df = df[df['component'].str.strip().ne('')].copy()
df = df.drop_duplicates(subset=['repo', 'file']).reset_index(drop=True)

# Fill missing columns
numeric_cols = ['hooks_total', 'props', 'jsx_depth', 'jsx_elems',
                'event_handlers', 'conditionals', 'map_calls',
                'filter_calls', 'reduce_calls', 'has_fetch', 'num_imports']
for col in numeric_cols:
    if col not in df.columns:
        df[col] = 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# Build ENRICHED combined_context — inject structural features into text
# so MiniLM can encode structural intent, not just names/comments
df['component'] = df['component'].fillna('')
df['comment']   = df['comment'].fillna('')

def build_enriched_context(row):
    parts = [f"Component: {row['component']}"]

    # Hook signature
    hooks_used = []
    for h in ['useState', 'useEffect', 'useCallback', 'useMemo',
              'useContext', 'useReducer', 'useRef']:
        if row.get(h, 0) > 0:
            hooks_used.append(h)
    if hooks_used:
        parts.append(f"Uses hooks: {', '.join(hooks_used)}")

    # Complexity tier
    ht = row.get('hooks_total', 0)
    if ht >= 8:
        parts.append("Complexity: very high stateful component")
    elif ht >= 5:
        parts.append("Complexity: high stateful component")
    elif ht >= 2:
        parts.append("Complexity: moderate stateful component")
    elif ht == 0:
        parts.append("Complexity: stateless presentational component")

    # JSX depth
    depth = row.get('jsx_depth', 0)
    if depth >= 10:
        parts.append("Deep nested JSX structure")
    elif depth >= 5:
        parts.append("Moderate JSX nesting")

    # Behavioral signals
    if row.get('has_fetch', 0) == 1:
        parts.append("Fetches remote data")
    if row.get('event_handlers', 0) >= 2:
        parts.append("Has multiple event handlers")
    if row.get('map_calls', 0) >= 2:
        parts.append("Renders lists with map")
    if row.get('conditionals', 0) >= 3:
        parts.append("Complex conditional rendering")
    if row.get('props', 0) >= 5:
        parts.append(f"Accepts {int(row['props'])} props")

    # Comment
    if str(row.get('comment', '')).strip():
        parts.append(f"Description: {row['comment']}")

    return ' | '.join(parts)

df['combined_context'] = df.apply(build_enriched_context, axis=1)
print(f"  Example context: {df['combined_context'].iloc[0][:120]}...")

print(f"  Clean components: {len(df):,}  |  Repos: {df['repo'].nunique()}")

# ─── 2. TEXT ENCODING ─────────────────────────────────────────────────────────
print("\n[2/5] Encoding text with MiniLM...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
lex_feats = encoder.encode(
    df['combined_context'].tolist(),
    show_progress_bar=True,
    batch_size=64
).astype('float32')
print(f"  Text embeddings: {lex_feats.shape}")

# ─── 3. GRAPH TOPOLOGY (SVD on Hook+Prop co-occurrence) ───────────────────────
print("\n[3/5] Building structural graph (Hook + Prop topology)...")
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo',
             'useContext', 'useReducer', 'useRef', 'useCustom']

G = nx.Graph()
nodes_list = []
param_nodes = {}
idx_counter = len(df)

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
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("  Running SVD (64 components)...")
svd = SVD(n_components=64)
all_embeddings = svd.fit_transform(adjacency)

comp_indices = [i for i, node in enumerate(nodes_list) if node.startswith('C_')]
graph_embeddings_raw = all_embeddings[comp_indices].astype('float32')

# Save RAW magnitudes (structural density signal) BEFORE normalization
graph_mags = np.linalg.norm(graph_embeddings_raw, axis=1)
print(f"  Graph mags: min={graph_mags.min():.3f} max={graph_mags.max():.3f} mean={graph_mags.mean():.3f}")

# Normalize for directional similarity in FAISS
norms = graph_mags.copy().reshape(-1, 1)
norms[norms == 0] = 1
graph_embeddings = graph_embeddings_raw / norms
print(f"  Graph embeddings: {graph_embeddings.shape}")

# ─── 4. FUSE & BUILD FAISS INDEX ──────────────────────────────────────────────
print("\n[4/5] Fusing embeddings (20% Graph + 80% Text = 448D)...")
weighted_graph = graph_embeddings * 0.2
weighted_lex   = lex_feats   * 0.8
fusion = np.hstack((weighted_graph, weighted_lex))

index = faiss.IndexFlatL2(fusion.shape[1])
index.add(fusion)
print(f"  FAISS index: {index.ntotal} vectors × {index.d} dims")

# ─── 5. SAVE ──────────────────────────────────────────────────────────────────
print("\n[5/5] Saving artifacts to data/...")
Path('data').mkdir(exist_ok=True)
faiss.write_index(index, 'data/graphrag_index.faiss')
np.save('data/graph_embeddings.npy', graph_embeddings)
np.save('data/graph_mags.npy', graph_mags)
df.to_pickle('data/vectors_reference.pkl')

print("\n" + "=" * 55)
print("  DONE. Artifacts written:")
print(f"  data/graphrag_index.faiss  ({Path('data/graphrag_index.faiss').stat().st_size/1e6:.1f} MB)")
print(f"  data/graph_embeddings.npy  ({Path('data/graph_embeddings.npy').stat().st_size/1e6:.1f} MB)")
print(f"  data/vectors_reference.pkl ({Path('data/vectors_reference.pkl').stat().st_size/1e6:.1f} MB)")
print(f"\n  Next: python analytics/benchmark_evaluator.py")
