import pandas as pd 
from sentence_transformers import SentenceTransformer 
import numpy as np 
import os
import logging
from sklearn.preprocessing import StandardScaler
import faiss

logging.basicConfig(level=logging.INFO)
logging.info('Loading DF')
df = pd.read_csv('master2.csv')
df_filtered = df[~((df['hooks_total'] == 0) & (df['props'] <= 1) & (df['jsx_depth'] <= 3))].copy()

numeric_cols = ['hooks_total', 'props', 'jsx_depth', 'jsx_elems', 'event_handlers', 'conditionals', 'map_calls', 'filter_calls', 'reduce_calls', 'has_fetch', 'num_imports']
for col in numeric_cols:
  if col not in df_filtered.columns:
    df_filtered[col] = 0

structured_matrix = df_filtered[numeric_cols].fillna(0)
scaler = StandardScaler()
scaled_feats = scaler.fit_transform(structured_matrix).astype('float32') * 0.2

logging.info('Encoding text')
encoder = SentenceTransformer('all-MiniLM-L6-v2')
df_filtered['component'] = df_filtered['component'].fillna('')
df_filtered['comment'] = df_filtered['comment'].fillna('')
df_filtered['combined_context'] = 'Component: ' + df_filtered['component'].astype(str) + ' Comments: ' + df_filtered['comment'].astype(str)

lexical_feats = encoder.encode(df_filtered['combined_context'].tolist(), show_progress_bar=False).astype('float32') * 0.8

logging.info('Combining & Indexing')
combined = np.hstack((scaled_feats, lexical_feats))
dimension = combined.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(combined)

logging.info(f'Write {index.ntotal} dims {dimension}')
faiss.write_index(index, 'data_index.faiss')
df_filtered.to_pickle('vectors_reference.pkl')
logging.info('Saved successfully')
