import pandas as pd
import numpy as np
import time
import os
import faiss
import warnings
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

# We'll sample 5 queries from each of the 4 categories to make a digestible 20-query 
# human validation subset first. (20 queries * 5 docs = 100 annotations). 
# We can scale to 50 queries once the tool is tested.
HUMAN_EVAL_QUERIES = [
    # STRICT_NEGATION
    "completely stateless layout container with zero hooks",
    "pure presentational badge with no state at all",
    "deeply nested layout that fetches no data and has no hooks",
    "text display component with no interactivity or state",
    "stateless wrapper with no context connection",
    
    # EXACT_MECHANICS
    "complex global authentication provider using context",
    "animated modal with refs and multiple event handlers",
    "form with multiple conditional validations and state",
    "complex table with filtering and mapping",
    "data fetching component with state and side effects",
    
    # TRUNCATION_TRAP
    "extremely large data fetching dashboard over 150 lines",
    "massive complex form over 200 lines with many states",
    "giant monolithic page component with deep html nesting",
    "extensive layout heavy component over 300 lines",
    "monolithic sidebar with complex state over 200 lines",

    # FUZZY_SEMANTIC
    "somewhat complex interactive form wrapper",
    "mostly simple display card with one state at most",
    "data grid heavy on logic but light on dom elements",
    "tiny basic generic button component under 50 lines",
    "standard modal dialog with minimal state"
]

def load_data():
    print("Loading component database and FAISS index...")
    df = pd.read_csv('data/master2.csv')
    df = df[df['component'].str.strip().ne('')].drop_duplicates(subset=['repo','file']).reset_index(drop=True)
    
    RAW = ['hooks_total','useState','useEffect','useContext','useRef','jsx_depth','jsx_elems',
           'has_fetch','event_handlers','loc']
    for c in RAW:
        if c not in df.columns: df[c] = 0
    df[RAW] = df[RAW].fillna(0)
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("data/semantic_hnsw.faiss")
    return df, embedder, index

def get_blinded_candidates(query, df, embedder, index, k=5):
    """
    For a human evaluation, you want to pool results from BOTH your baseline (Dense RAG)
    and your novel method (AG+Router), shuffle them, and hide the source from the annotator.
    To keep this script lightweight, we will just retrieve the Top-5 from Dense RAG here 
    as a placeholder, but naturally you'd pool and drop duplicates.
    """
    q_emb = embedder.encode([query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q_emb)
    _, top_idx = index.search(q_emb, k=k)
    return top_idx[0]

def main():
    print("="*60)
    print(" 🧑‍🔬 ANTI-GRAVITY: HUMAN ANNOTATION & VALIDATION TOOL")
    print("="*60)
    print("This tool enables blind human grading for Code Search.")
    print("Relevance Scale:")
    print("  3: Perfect Match (Hits all structural & semantic constraints)")
    print("  2: Highly Relevant (Minor structural drift, but solves intent)")
    print("  1: Weakly Relevant (Semantically related but fails constraints)")
    print("  0: Irrelevant (Completely wrong or hallucinates logic)")
    print("\nPress 'q' at any time to save and quit.\n")
    
    annotator_name = input("Enter your Annotator Name/ID: ").strip().replace(" ","_")
    if not annotator_name: annotator_name = "anonymous"
    
    output_file = f"data/human_annotations_{annotator_name}.csv"
    
    # Load previous progress if exists
    if os.path.exists(output_file):
        answers_df = pd.read_csv(output_file)
        print(f"Loaded existing progress: {len(answers_df)} judgments found.")
        graded_pairs = set(zip(answers_df['query'], answers_df['component_name']))
    else:
        answers_df = pd.DataFrame(columns=['annotator', 'query', 'component_name', 'score', 'repo', 'file'])
        graded_pairs = set()

    df, embedder, index = load_data()
    
    new_records = []
    
    for q_idx, query in enumerate(HUMAN_EVAL_QUERIES):
        print(f"\n[{q_idx+1}/{len(HUMAN_EVAL_QUERIES)}] QUERY: '{query}'")
        candidates_idx = get_blinded_candidates(query, df, embedder, index, k=5)
        
        for rank, c_idx in enumerate(candidates_idx):
            comp = df.iloc[c_idx]
            comp_name = comp['component']
            
            if (query, comp_name) in graded_pairs:
                continue # Already graded
            
            print("-"*50)
            print(f"Candidate {rank+1}/5:")
            print(f"  Name: {comp_name}")
            print(f"  Repo: {comp.get('repo', 'N/A')}")
            print(f"  Doc:  {comp.get('comment', 'No documentation available')[:200]}")
            print(f"  Stats: {comp['loc']} lines | {comp['hooks_total']} total hooks | {comp['jsx_elems']} UI elements")
            print(f"  Hooks: useState({comp['useState']}), useEffect({comp['useEffect']}), context({comp['useContext']})")
            print("-"*50)
            
            while True:
                score = input("  Score (0-3) or 'q' to quit: ").strip().lower()
                if score == 'q':
                    break
                if score in ['0','1','2','3']:
                    new_records.append({
                        'annotator': annotator_name,
                        'query': query,
                        'component_name': comp_name,
                        'score': int(score),
                        'repo': comp.get('repo', ''),
                        'file': comp.get('file', '')
                    })
                    graded_pairs.add((query, comp_name))
                    break
                print("  Invalid input. Please enter 0, 1, 2, or 3.")
            
            if score == 'q': break
        if score == 'q': break
        
    # Save newly graded to CSV
    if new_records:
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([answers_df, new_df], ignore_index=True) if not answers_df.empty else new_df
        combined_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(new_records)} new judgments to {output_file}.")
    else:
        print("\nNo new judgments to save.")
        
    print("Thank you for your annotations!")

if __name__ == "__main__":
    main()
