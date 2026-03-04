import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import networkx as nx
from sknetwork.embedding import SVD
import scipy.sparse
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("--- 1. LOADING DATA AND RE-COMPUTING GRAPH EMBEDDINGS ---")
df = pd.read_pickle('vectors_reference.pkl')

G = nx.Graph()
nodes_list = []
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']
idx_counter = len(df)
param_nodes = {}

# Build exact same graph as before
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

# SVD for 64 dims
adjacency = scipy.sparse.csr_matrix(nx.adjacency_matrix(G, nodelist=nodes_list))
svd = SVD(n_components=64)
graph_embeddings_raw = svd.fit_transform(adjacency)

comp_indices = [i for i, node in enumerate(nodes_list) if node.startswith('C_')]
graph_embeddings = graph_embeddings_raw[comp_indices].astype('float32')
norms = np.linalg.norm(graph_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
graph_embeddings = graph_embeddings / norms

# Convert embeddings to a DataFrame for classical ML
# X features = the 64 graph dimensions
# y targets = the actual structural integers (hooks_total, jsx_depth)
X = pd.DataFrame(graph_embeddings, columns=[f"G_Dim_{i}" for i in range(64)])
y_hooks = df['hooks_total'].values
y_depth = df['jsx_depth'].values

print("\n" + "="*60)
print("--- 2. EDA: PCA ON TOPOLOGICAL EMBEDDINGS ---")
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

print("Explained Variance Ratio of Top 5 Graph Dimensions:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f}")
print(f"  Total variance captured by just 5 topological vectors: {sum(pca.explained_variance_ratio_):.4f}")

print("\n" + "="*60)
print("--- 3. t-SNE: VISUALIZING THE GRAPH CODE UNIVERSE ---")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
# Color by hooks_total to see if topology groups high-state components together
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_hooks, cmap='viridis', s=15, alpha=0.7)
plt.colorbar(scatter, label='Total Hooks')
plt.title("t-SNE of 64D Component Graph Embeddings")
plt.savefig('tsne_graph_embeddings.png')
plt.close()
print("Saved 'tsne_graph_embeddings.png'.")

print("\n" + "="*60)
print("--- 4. RANDOM FOREST GINI: WHICH GRAPH DIMS HOLD THE SECRET TO HOOKS? ---")
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X, y_hooks)

print(f"Random Forest R^2 predicting 'hooks_total': {rf.score(X, y_hooks):.4f}")
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 5 most predictive Graph Dimensions:")
for f, imp in importances.head(5).items():
    print(f"  {f:<10}: {imp:.4f}")

print("\n" + "="*60)
print("--- 5. REGRESSION MODELS: PREDICTING JSX DEPTH FROM TOPOLOGY ---")
X_train, X_test, y_train, y_test = train_test_split(X, y_depth, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression R^2:    {lr.score(X_test, y_test):.4f}")

# Lasso (L1 Regularization to find true sparse topological features)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train, y_train)
print(f"Lasso Regression R^2:     {lasso.score(X_test, y_test):.4f}")
non_zero = sum(lasso.coef_ != 0)
print(f"  -> Lasso isolated {non_zero} out of 64 Graph Dims as meaningful.")

# Polynomial Regression (Degree 2)
# Does multiplying dimensions together improve prediction?
poly = make_pipeline(PolynomialFeatures(2, interaction_only=True), LinearRegression())
poly.fit(X_train, y_train)
print(f"Polynomial (interaction) R^2: {poly.score(X_test, y_test):.4f}")

# Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree R^2:        {dt.score(X_test, y_test):.4f}")
