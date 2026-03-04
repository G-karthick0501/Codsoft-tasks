import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('master2.csv')
numeric_cols = ['hooks_total', 'props', 'jsx_depth', 'jsx_elems', 'event_handlers', 'conditionals', 'map_calls', 'filter_calls', 'reduce_calls', 'has_fetch', 'num_imports']

df_filtered = df[~((df['hooks_total'] == 0) & (df['props'] <= 1) & (df['jsx_depth'] <= 3))].copy()
for col in numeric_cols:
    if col not in df_filtered.columns:
        df_filtered[col] = 0

X = df_filtered[numeric_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("="*60)
print("--- 1. GMM (Gaussian Mixture Model) Clustering ---")
# Choosing 4 components to compare with previous KMeans
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

clusters = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

df_filtered['gmm_cluster'] = clusters
df_filtered['gmm_confidence'] = probs.max(axis=1)

print("GMM Cluster sizes:")
print(df_filtered['gmm_cluster'].value_counts().sort_index())

print("\nAverage confidence by cluster:")
print(df_filtered.groupby('gmm_cluster')['gmm_confidence'].mean())

print("\nTop 5 uncertain components (lowest max probability, i.e., hybrids):")
uncertain = df_filtered.nsmallest(5, 'gmm_confidence')
for _, row in uncertain.iterrows():
    print(f"  {row['component']} ({row['file']}) - Confidence: {row['gmm_confidence']:.3f}")
    
print("\nTop 5 highly confident components:")
confident = df_filtered.nlargest(5, 'gmm_confidence')
for _, row in confident.iterrows():
    print(f"  {row['component']} ({row['file']}) - Confidence: {row['gmm_confidence']:.3f}")

print("\nComputing t-SNE (2 components) for GMM...")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=clusters, cmap='viridis', s=10, alpha=0.6)
plt.legend(*scatter.legend_elements(), title="GMM Clusters")
plt.title("t-SNE components visualization (colored by GMM)")
plt.savefig('tsne_gmm_clusters.png')
plt.close()

print("="*60)
