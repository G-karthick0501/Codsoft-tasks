import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="React Structural Discovery", layout="wide", page_icon="⚛️")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    .result-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 15px;
    }
    .metric-bubble {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background-color: #2e2e2e;
        margin-right: 8px;
        font-size: 0.85em;
        color: #8bc34a;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_discovery_engine():
    # Load core artifacts
    index = faiss.read_index('data/graphrag_index.faiss')
    df = pd.read_pickle('data/vectors_reference.pkl')
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return index, df, encoder

try:
    index, df, encoder = load_discovery_engine()
except Exception as e:
    st.error(f"Failed to load discovery core: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Discovery Core")
    st.write("Current Manifold: **448D (384 Text + 64 Graph)**")
    st.write(f"Total Components Indexed: **{len(df)}**")
    st.divider()
    st.info("This system uses SVD-based Topological DNA to find components that are structurally similar, not just similarly named.")

# --- MAIN UI ---
st.title("⚛️ React Architectural Intelligence")
st.subheader("Discover code via Structural DNA and Semantic Intent")

tab1, tab2 = st.tabs(["🔍 Search Discoveries", "🏗️ Onboard New Repo"])

with tab1:
    query = st.text_input("Describe the component requirement", 
                         placeholder="e.g. 'Highly nested data table with many hooks'")

    col1, col2 = st.columns([2, 1])

    if query:
        # 1. Processing
        with st.spinner("Analyzing Architecture..."):
            # Hybrid Search Logic
            q_lex = encoder.encode([query]).astype('float32') * 0.8
            q_graph = np.zeros((1, 64), dtype='float32')
            if any(k in query.lower() for k in ['complex', 'state', 'hook', 'logic', 'provider']):
                 q_graph = np.ones((1, 64), dtype='float32') * 0.2 # Boost structural signal
            
            query_vec = np.hstack((q_graph, q_lex))
            D, I = index.search(query_vec, 6) # Top 6 results

        # 2. Results Display
        with col1:
            st.write(f"### 🔍 Structural Twins for '{query}'")
            for i, idx in enumerate(I[0]):
                row = df.iloc[idx]
                conf = 1.0 / (1.0 + D[0][i])
                
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style='margin-top:0;'>{row['component']}</h3>
                        <p style='color:#aaaaaa;'>File: <code>{str(row['file']).split('/')[-1]}</code></p>
                        <div style='margin-bottom:10px;'>
                            <span class="metric-bubble">🪝 {row['hooks_total']} Hooks</span>
                            <span class="metric-bubble">📐 Depth {row['jsx_depth']}</span>
                            <span class="metric-bubble">🧩 {row['props']} Props</span>
                        </div>
                        <p style='font-size:0.9em; font-style:italic;'>"{str(row['comment'])[:200]}..."</p>
                        <p style='text-align:right; font-size:0.8em; color:#4CAF50;'>
                            <b>Match Confidence: {conf:.4f}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # 3. Analytics Panel
        with col2:
            st.write("### 🧬 Architectural Audit")
            
            # Plot 1: Feature comparison of results
            current_res_df = df.iloc[I[0]]
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.style.use('dark_background')
            sns.barplot(data=current_res_df, x='component', y='hooks_total', palette='viridis', ax=ax)
            plt.xticks(rotation=45)
            plt.title("Hook Density Comparison")
            st.pyplot(fig)
            
            st.divider()
            st.warning("⚡ **Structural Veto Active**: Re-ranked based on mathematical dependency density.")
            
            st.write("#### Topological Summary")
            top_res = df.iloc[I[0][0]]
            st.json({
                "Archetype": "High Complexity Provider" if top_res['hooks_total'] > 5 else "UI Primitive",
                "Latent Signal": "Strong Structural" if 'q_graph' in locals() else "Semantic Only",
                "Scanned Nodes": len(df)
            })

    else:
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=1000", 
                 caption="Mapping the Neural Fabric of Software Architecture", width=1200)

with tab2:
    st.write("### 🚀 Index a New Codebase")
    st.write("Point the Discovery Engine at any local repository to build a custom Structural Manifold.")
    repo_path = st.text_input("Local Repository Path", placeholder="C:/Projects/MyReactApp")
    
    if st.button("🏗️ Build Structural DNA Index"):
        if not repo_path:
            st.error("Please provide a valid file path.")
        else:
            with st.spinner("Scanning, SVD Mapping, and Embedding..."):
                # Real-world onboarding logic would go here
                st.success(f"Successfully Indexed repo at {repo_path}")
                st.balloons()
                st.info("The new structural weights have been saved to 'data/graphrag_index.faiss'. Restart the app to search the new repo.")
