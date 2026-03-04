import pandas as pd
import numpy as np
import faiss
import sys
import argparse
from sentence_transformers import SentenceTransformer

# PRODUCTION SERVICE SCRIPT: Structural React Discovery
# This script acts as the "Inference Engine" for external systems.

class DiscoveryService:
    def __init__(self):
        print("[Service]: Initializing Omnimodal Inference Core...")
        # 1. Load the vectors for dynamic routing (384D + 64D)
        self.df = pd.read_pickle('data/vectors_reference.pkl')
        # PURIFY: Ensure clean component names
        self.df = self.df[self.df['component'].str.strip() != ''].reset_index(drop=True)
        
        # 2. Load the Semantic Encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Load the pre-calculated structural DNA
        # We use hooks_total and jsx_depth as a fast-indexing proxy for structural complexity
        print("Materializing Structural Manifold...")
        self.lex_feats = self.encoder.encode(self.df['combined_context'].tolist(), show_progress_bar=False).astype('float32')
        
        # Calculate Structural Magnitude (Normalized proxy)
        # Using weights from research: Hooks (0.7) + Depth (0.3)
        h_vals = self.df['hooks_total'].astype(float).values
        d_vals = self.df.get('jsx_depth', pd.Series(0, index=self.df.index)).astype(float).values
        
        # Normalize to 0-1 range
        h_norm = h_vals / (np.max(h_vals) or 1)
        d_norm = d_vals / (np.max(d_vals) or 1)
        
        self.structural_complexities = (h_norm * 0.7) + (d_norm * 0.3)
        
        print(f"[Service]: Model Ready. Index contains {len(self.df)} structural archetypes.")

    def query_by_text(self, query_text, top_k=3):
        """
        Calculates a hybrid intent-aware search and returns the closest architectural twins.
        Uses the Recall-then-Rank strategy proven in benchmarks.
        """
        from scipy.spatial.distance import cdist
        
        # 1. Intent Detection
        q_lower = query_text.lower()
        structural_keywords = ['hooks', 'state', 'complex', 'provider', 'context', 'nested', 'logic', 'fetch', 'architecture', 'global']
        w_t, w_g = (0.4, 0.6) if any(k in q_lower for k in structural_keywords) else (0.8, 0.2)
        
        # 2. RECALL: Top 50 by Text
        q_lex = self.encoder.encode([query_text]).astype('float32')
        dist_lex = cdist(q_lex, self.lex_feats, metric='cosine')[0]
        recall_indices = np.argsort(dist_lex)[:50]
        
        # 3. RANK: Structural Re-ranking
        results = []
        max_complex = np.max(self.structural_complexities) or 1
        
        final_scores = []
        for idx in recall_indices:
            text_dist = dist_lex[idx]
            complexity = self.structural_complexities[idx]
            ratio = complexity / max_complex
            
            # Score formula (Lower is better)
            # REWARD high structural density by subtracting from semantic distance
            score = (text_dist * w_t) - (ratio * w_g * 0.15)
            final_scores.append((score, idx))
            
        final_scores.sort(key=lambda x: x[0])
        
        for score, idx in final_scores[:top_k]:
            row = self.df.iloc[idx]
            results.append({
                'component': row['component'],
                'file': row['file'],
                'confidence': float(1.0 - max(0, score)), # Normalized confidence
                'architecture': {
                    'hooks': row.get('hooks_total', 0),
                    'depth': row.get('jsx_depth', 0),
                    'props': row.get('props', '')
                }
            })
        return results

if __name__ == "__main__":
    service = DiscoveryService()
    
    # Simple CLI for service testing
    parser = argparse.ArgumentParser(description='Structural Discovery Service')
    parser.add_argument('--query', type=str, help='The component requirement to search for')
    args = parser.parse_args()

    if args.query:
        results = service.query_by_text(args.query)
        print(f"\n[RESULTS FOR]: '{args.query}'")
        for r in results:
            print(f"  - {r['component']} (Confidence: {r['confidence']:.4f})")
            print(f"    DNA: {r['architecture']['hooks']} Hooks, {r['architecture']['depth']} Depth")
    else:
        # Default test
        test_q = "A complex state management provider"
        res = service.query_by_text(test_q)
        print(f"\n[SERVICE TEST]: '{test_q}'")
        for r in res:
            print(f"  -> {r['component']} | Hooks: {r['architecture']['hooks']}")
