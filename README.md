# Omnimodal React Structural Discovery

This project implements a multi-modal GraphRAG system for discovering and retrieving React components based on their architectural DNA (topological structure) and semantic intent.

## 📁 Project Structure

### 🧠 [core_engine/](core_engine/)
Contains the primary logic for the discovery engine.
- `graph_embeddings.py`: Generates 64D SVD structural embeddings.
- `dynamic_router.py`: The live inference engine with late-fusion gating.
- `set_reproducibility.py`: Seeds for deterministic research results.
- `embed_components.py`: Base script for embedding generation.

### 📊 [analytics/](analytics/)
Scientific evaluation and forensic audit tools.
- `latent_correlation_audit.py`: Proves the mapping between SVD dimensions and AST features.
- `find_topological_outliers.py`: Identifies "Rogue Planets" (structurally unique components).
- `conflict_audit.py`: Benchmarks GraphRAG against standard Text-only RAG.
- `graph_advanced_eda.py`: Performs BIC/AIC clustering and manifold visualization.

### 🍱 [data/](data/)
Storage for raw data and generated artifacts.
- `master2.csv`: The primary dataset (2,617 components).
- `vectors_reference.pkl`: Preservation of the scraped dataframe.
- `graphrag_index.faiss`: The compiled vector database.
- **[visualizations/](data/visualizations/)**: Manifold plots, heatmaps, and feature importance charts.

### 🚀 [future_endeavors/](future_endeavors/)
Advanced multi-modal extensions (Proposed/Simulation phase).
- **Vision Layer**: CLIP-ViT integration scripts.
- **Motion DNA**: Lucas-Kanade optical flow analysis.

### 📜 [archives/](archives/)
Legacy scripts, raw logs, and intermediate test datasets.

---

## ⚡ Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a Dynamic Query**:
   ```bash
   python core_engine/dynamic_router.py
   ```

3. **Verify the Math (Reproducibility)**:
   ```bash
   python analytics/latent_correlation_audit.py
   ```

## 🎓 Research Summary
This architecture proves that React components possess a stable **Topological DNA**. Dimension 1 of our SVD manifold correlates with physical **State Complexity (Hooks)** at a factor of **0.96**, enabling high-fidelity retrieval without code execution.
