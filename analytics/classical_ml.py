import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('master2.csv')
numeric_cols = ['hooks_total', 'props', 'jsx_depth', 'jsx_elems', 'event_handlers', 'conditionals', 'map_calls', 'filter_calls', 'reduce_calls', 'has_fetch', 'num_imports']

# Filter trivial
df_filtered = df[~((df['hooks_total'] == 0) & (df['props'] <= 1) & (df['jsx_depth'] <= 3))].copy()
for col in numeric_cols:
    if col not in df_filtered.columns:
        df_filtered[col] = 0

X = df_filtered[numeric_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

print("="*60)
print("--- 1. CORRELATION ANALYSIS ---")
corr = X.corr()
print("Top correlated feature pairs (r > 0.5):")
corr_pairs = corr.unstack().sort_values(ascending=False).drop_duplicates()
for (c1, c2), r in corr_pairs.items():
    if c1 != c2 and r > 0.5:
        print(f"  {c1} <-> {c2} : {r:.3f}")

# Save heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Structural Features")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("\n" + "="*60)
print("--- 2. PCA (Principal Component Analysis) ---")
pca = PCA()
pca.fit(X_scaled)
print("Explained Variance Ratio by Component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f}")
print(f"  Cumulative Variance (Top 3): {sum(pca.explained_variance_ratio_[:3]):.4f}")

components_df = pd.DataFrame(pca.components_, columns=numeric_cols)
print("\nPC1 feature weights (Primary Structural Axis):")
print(components_df.iloc[0].sort_values(ascending=False).head(5))
print("\nPC2 feature weights (Secondary Axis):")
print(components_df.iloc[1].sort_values(ascending=False).head(5))

print("\n" + "="*60)
print("--- 3. KMeans & t-SNE ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_filtered['cluster'] = clusters

print("Cluster sizes:")
print(df_filtered['cluster'].value_counts().sort_index())

print("\nComputing t-SNE (2 components)...")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=clusters, cmap='viridis', s=10)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("t-SNE components visualization (colored by KMeans)")
plt.savefig('tsne_clusters.png')
plt.close()

print("\n" + "="*60)
print("--- 4. Random Forest Feature Importance (Predicting jsx_elems) ---")
target_col = 'jsx_elems'
features = [c for c in numeric_cols if c != target_col]
X_rf = X_scaled_df[features]
y_rf = X[target_col]

rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_rf, y_rf)
print(f"Target: {target_col}")
print("Gini Importance:")
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
for f, imp in importances.items():
    print(f"  {f:<15}: {imp:.4f}")

print("\n" + "="*60)
print("--- 5. Decision Tree Regressor (Predicting jsx_depth) ---")
target_col_dt = 'jsx_depth'
features_dt = [c for c in numeric_cols if c != target_col_dt]
X_dt = X_scaled_df[features_dt]
y_dt = X[target_col_dt]

dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(X_dt, y_dt)
print(f"Decision Tree R2 Score (Target={target_col_dt}): {dt.score(X_dt, y_dt):.4f}")
importances_dt = pd.Series(dt.feature_importances_, index=features_dt).sort_values(ascending=False)
print("Top 3 split features:")
print(importances_dt.head(3))

print("\n" + "="*60)
print("--- 6. Lasso Regression (Predicting hooks_total) ---")
target_col_lasso = 'hooks_total'
features_lasso = [c for c in numeric_cols if c != target_col_lasso]
X_lasso = X_scaled_df[features_lasso]
y_lasso = X[target_col_lasso]

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_lasso, y_lasso)
print(f"Target: {target_col_lasso}")
print("Lasso Non-Zero Coefficients (alpha=0.1):")
coef = pd.Series(lasso.coef_, index=features_lasso).sort_values(ascending=False)
for f, c in coef.items():
    if abs(c) > 0:
        print(f"  {f:<15}: {c:.4f}")
