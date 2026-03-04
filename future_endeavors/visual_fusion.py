import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss

print("="*60)
print("--- 1. LOADING VISUAL VETO BACKBONE (CLIP ViT) ---")
# Using OpenAI's CLIP (Contrastive Language-Image Pretraining) with a ViT-B/32 backbone.
# This gives us a 512-dimensional vector that represents the "Visual Semantics" of an image.
model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_id} on {device}...")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

df = pd.read_pickle('vectors_reference.pkl')
SCREENSHOT_DIR = "component_screenshots"

# We will create a dummy matrix for the vision embeddings. 
# This ensures we have a unified space of dimensions for the ENTIRE dataset 
# (Even components that failed to render or don't have screenshots yet).
# If a component has no screenshot, it gets a vector of Zeros.
vision_dim = 512
vision_embeddings = np.zeros((len(df), vision_dim), dtype='float32')

print(f"\n--- 2. EXTRACTING VISION VECTORS (Handling Sub-Samples Gracefully) ---")
# We only process screenshots that actually exist in the folder.
# This solves your worry! We don't drop rows. We just add a "0" vector for missing ones.
has_image_count = 0
for idx, row in df.iterrows():
    img_path = os.path.join(SCREENSHOT_DIR, f"{row['component']}.png")
    
    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            # Create the embedding
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # Normalize the visual vector
            feat = image_features.cpu().numpy()[0]
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            
            vision_embeddings[idx] = feat
            has_image_count += 1
            if has_image_count % 10 == 0:
                print(f"  Processed {has_image_count} images...")
        except Exception as e:
            print(f"  Failed to process {img_path}: {e}")

print(f"\nSuccessfully extracted Vision Veto vectors for {has_image_count} out of {len(df)} components.")

print("\n" + "="*60)
print("--- 3. CREATING THE 'TRIPLE-THREAT' FUSION EMBEDDING ---")

# Let's load the Graph + Text index components we already saved in the dataframe
# Re-running the precise weighting from previous success
from sentence_transformers import SentenceTransformer
import networkx as nx
from sknetwork.embedding import SVD
import scipy.sparse

print("Rebuilding Graph Dims...")
# (Standard fast graph rebuild to get the exact 64D array back)
G = nx.Graph()
nodes_list = []
comp_to_idx = {}
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']
idx_counter = len(df)
param_nodes = {}
for idx, row in df.iterrows():
    c_node = f"C_{idx}"
    comp_to_idx[idx] = c_node
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

print("Encoding text...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
lex_feats = encoder.encode(df['combined_context'].tolist(), show_progress_bar=False).astype('float32')

print("Fusing the holy trinity: Graph (Architecture) + Text (Intent) + Vision (Perception)...")
# The New Weighting Formula:
# Graph Topology: 20%
# Semantic Text:  60%
# Visual Intent:  20%
weighted_graph = graph_embeddings * 0.2
weighted_lex = lex_feats * 0.6
weighted_vision = vision_embeddings * 0.2

# Stack them horizontally: 64 (Graph) + 384 (Text) + 512 (Vision) = 960 Dimensions!
fusion_embeddings = np.hstack((weighted_graph, weighted_lex, weighted_vision))

print("\n" + "="*60)
print("--- 4. BUILDING OMNIMODAL FAISS INDEX ---")
dim = fusion_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(fusion_embeddings)

faiss.write_index(index, 'omnimodal_index.faiss')
# Save the vision embeddings alongside it so we don't have to recompute
np.save('vision_embeddings.npy', vision_embeddings)

print(f"Index built successfully with {index.ntotal} components.")
print(f"Total Dimensions: {dim} -> (64 Topological + 384 Semantic + 512 Visual)")
print("Components without screenshots received zero-vectors, perfectly maintaining matrix dimensions without breaking any downstream queries!")
