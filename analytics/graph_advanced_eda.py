import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from sknetwork.embedding import SVD
import scipy.sparse
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("--- 1. REBUILDING GRAPH EMBEDDINGS ---")
df = pd.read_pickle('data/vectors_reference.pkl')

hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']
structural_cols = ['hooks_total', 'props', 'jsx_depth', 'jsx_elems', 'event_handlers', 'conditionals', 'map_calls', 'filter_calls', 'reduce_calls', 'has_fetch', 'num_imports']

G = nx.Graph()
nodes_list = []
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

print("\n" + "="*60)
print("--- 2. CORRELATIONAL MATRIX & HEATMAP (Graph Dims vs Structural Features) ---")
# To understand what the graph actually learned, we correlate the Top 10 Graph Dimensions
# directly against the 11 original AST structural integers.
graph_df = pd.DataFrame(graph_embeddings[:, :10], columns=[f"G_Dim_{i}" for i in range(10)])
struct_df = df[structural_cols].fillna(0).reset_index(drop=True)

combined_df = pd.concat([graph_df, struct_df], axis=1)
corr_matrix = combined_df.corr().loc[[f"G_Dim_{i}" for i in range(10)], structural_cols]

print("Top absolute correlations between Graph Dimensions and AST Logic:")
top_corrs = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
for (struct_feat, g_dim), val in top_corrs.head(10).items():
    actual_val = corr_matrix.loc[g_dim, struct_feat]
    print(f"  {g_dim} <-> {struct_feat:<15}: {actual_val:.4f}")

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Correlation Heatmap: Graph Dimensions vs AST Structural Features")
plt.tight_layout()
plt.savefig('data/visualizations/graph_ast_heatmap.png')
plt.close()
print("Saved 'data/visualizations/graph_ast_heatmap.png'")

print("\n" + "="*60)
print("--- 3. t-SNE PERPLEXITY ANALYSIS ---")
# Perplexity acts as a "zoom" knob for manifold learning.
# Low perplexity (5) focuses on local neighbor structures.
# High perplexity (100) focuses on global topological structures.
perplexities = [5, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, p in enumerate(perplexities):
    print(f"Computing t-SNE with perplexity={p}...")
    tsne = TSNE(n_components=2, perplexity=p, random_state=42)
    X_tsne = tsne.fit_transform(graph_embeddings)
    scatter = axes[i].scatter(X_tsne[:,0], X_tsne[:,1], c=struct_df['hooks_total'], cmap='plasma', s=10, alpha=0.7)
    axes[i].set_title(f"t-SNE (Perplexity: {p})")
    if i == 2:
        fig.colorbar(scatter, ax=axes[i], label='hooks_total')

plt.tight_layout()
plt.savefig('data/visualizations/tsne_perplexity_grid.png')
plt.close()
print("Saved 'data/visualizations/tsne_perplexity_grid.png'")

print("\n" + "="*60)
print("--- 4. GMM CLUSTERING & AIC/BIC ON GRAPH EMBEDDINGS ---")
# Finding the "true" number of archetypes in the 64D Graph using Information Criterion
n_components_range = [2, 4, 8, 12]
aic_scores = []
bic_scores = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='diag', random_state=42)
    gmm.fit(graph_embeddings)
    aic_scores.append(gmm.aic(graph_embeddings))
    bic_scores.append(gmm.bic(graph_embeddings))

print("GMM Model Selection (Lower is better):")
for n, aic, bic in zip(n_components_range, aic_scores, bic_scores):
    print(f"  Clusters: {n:<2} | AIC: {aic:.0f} | BIC: {bic:.0f}")

optimal_k = n_components_range[np.argmin(bic_scores)]
print(f"-> Optimal number of Topological Gaussian Clusters based on BIC: {optimal_k}")

print("\n" + "="*60)
print("--- 5. KNN DENSITY ANALYSIS (K-Nearest Neighbors) ---")
# How dense is the graph space? Are components clumped together or spread out?
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(graph_embeddings)
distances, indices = knn.kneighbors(graph_embeddings)

# Mean distance to the 5th nearest neighbor
avg_dist = np.mean(distances[:, 4])
print(f"Average Cosine Distance to 5th Nearest Neighbor: {avg_dist:.4f}")

# Find the most "Isolated" component (Highest distance to its 5th neighbor)
isolated_idx = np.argmax(distances[:, 4])
isolated_comp = df.iloc[isolated_idx]
print(f"Most Topologically Isolated Component:")
print(f"  {isolated_comp['component']} (File: {isolated_comp['file'].split('/')[-1] if isinstance(isolated_comp['file'], str) else ''})")
print(f"  Dist to 5th neighbor: {distances[isolated_idx, 4]:.4f}")
print(f"  Hooks: {isolated_comp['hooks_total']}, Props: {isolated_comp['props']}, JsxElems: {isolated_comp['jsx_elems']}")

print("\n" + "="*60)
print("--- 6. AGGLOMERATIVE (HIERARCHICAL) CLUSTERING ---")
# Agglomerative clustering builds a strict hierarchy of component evolution
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(graph_embeddings)
df['agg_cluster'] = agg_labels
print("Agglomerative Hierarchy Sizes (Top-Down groupings):")
print(df['agg_cluster'].value_counts().sort_index())
