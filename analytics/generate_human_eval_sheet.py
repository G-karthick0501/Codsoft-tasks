import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

HUMAN_EVAL_QUERIES = [
    # STRICT_NEGATION
    ("completely stateless layout container with zero hooks", lambda df: df['hooks_total'] == 0 ),
    ("pure presentational badge with no state at all", lambda df: df['hooks_total'] == 0),
    ("text display component with no interactivity or state", lambda df: (df['hooks_total'] == 0) & (df['event_handlers'] == 0)),
    
    # EXACT_MECHANICS
    ("animated modal with refs and multiple event handlers", lambda df: (df['useRef'] > 0) & (df['event_handlers'] >= 2)),
    ("form with multiple conditional validations and state", lambda df: (df['useState'] >= 2) & (df['conditionals'] >= 3)),
    ("complex table with filtering and mapping", lambda df: (df['map_calls'] > 0) & (df['filter_calls'] > 0)),
    
    # TRUNCATION_TRAP
    ("extremely large data fetching dashboard over 150 lines", lambda df: (df['has_fetch'] == 1) & (df['loc'] >= 150)),
    ("giant monolithic page component with deep html nesting", lambda df: (df['loc'] >= 200) & (df['jsx_depth'] >= 6)),
    ("massive interactive map using refs over 150 lines", lambda df: (df['loc'] >= 150) & (df['useRef'] > 0)),

    # FUZZY_SEMANTIC
    ("somewhat complex interactive form wrapper", lambda df: (df['event_handlers'] >= 1) & (df['useState'] >= 1)),
    ("mostly simple display card with one state at most", lambda df: (df['hooks_total'] <= 1) & (df['loc'] < 80)),
    ("tiny basic generic button component under 50 lines", lambda df: (df['loc'] < 50) & (df['hooks_total'] <= 1))
]

def generate_annotation_sample():
    print("Loading component database...")
    df = pd.read_csv('data/master2.csv')
    df = df[df['component'].str.strip().ne('')].drop_duplicates(subset=['repo','file']).reset_index(drop=True)
    
    # Clean NaNs
    RAW = ['hooks_total','useState','useEffect','useContext','useRef','jsx_depth','jsx_elems',
           'has_fetch','event_handlers','loc', 'map_calls', 'filter_calls', 'conditionals']
    for c in RAW:
        if c not in df.columns: df[c] = 0
    df[RAW] = df[RAW].fillna(0)

    records = []
    pair_id = 1
    
    print("Sampling 5 candidates per query (2 matches, 2 non-matches, 1 borderline)...")
    
    for q_idx, (q_text, label_rule_fn) in enumerate(HUMAN_EVAL_QUERIES):
        print(f"  Sampling for: {q_text[:40]}...")
        
        # Calculate ground truth based on rule
        df['is_match'] = label_rule_fn(df)
        
        matches = df[df['is_match']].sample(n=min(2, sum(df['is_match'])), random_state=42)
        non_matches = df[~df['is_match']].sample(n=min(2, sum(~df['is_match'])), random_state=42)
        
        # We need a borderline case (e.g. failing only one small condition). 
        # To keep it simple, we just grab a random non-match that isn't already selected.
        pool = df[~df['is_match'] & ~df.index.isin(non_matches.index)]
        borderlines = pool.sample(n=1, random_state=42) if not pool.empty else []
        
        candidates = pd.concat([matches, non_matches, borderlines])
        candidates = candidates.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle!
        
        for _, comp in candidates.iterrows():
            # Format the component code to display. We don't have raw code, but we have 
            # documentation and stats, which developers can judge.
            doc_str = str(comp.get('comment', ''))
            if doc_str in ('nan', 'None', ''): doc_str = "No documentation."
            
            code_proxy = (
                f"// Component Name: {comp['component']}\n"
                f"// Original File: {comp['repo']}/{comp['file']}\n"
                f"// Purpose: {doc_str[:300].strip()}\n\n"
                f"// --- AST Structural Properties ---\n"
                f"Line Count: {comp['loc']}\n"
                f"UI Elements: {comp['jsx_elems']} (Depth: {comp['jsx_depth']})\n"
                f"Total Hooks: {comp['hooks_total']}\n"
                f"useState: {comp['useState']}, useEffect: {comp['useEffect']}, useRef: {comp['useRef']}, useContext: {comp['useContext']}\n"
                f"Event Handlers: {comp['event_handlers']}\n"
                f"Array Maps: {comp['map_calls']}, Filters: {comp['filter_calls']}\n"
                f"Data Fetching: {'Yes' if comp['has_fetch']==1 else 'No'}"
            )

            records.append({
                "Pair ID": f"Q{q_idx+1}-C{pair_id}",
                "Query": q_text,
                "Component Snapshot (AST Data)": code_proxy,
                "Is Relevant? (1=Yes, 0=No)": "",
                "Annotator Notes (Optional)": "",
                "GroundTruth_Hidden": int(comp['is_match'])
            })
            pair_id += 1

    out_df = pd.DataFrame(records)
    
    # Save the grading sheet for annotators (Drop the hidden truth)
    annotator_sheet = out_df.drop(columns=['GroundTruth_Hidden'])
    annotator_sheet.to_csv("data/human_eval_form_annotator_1.csv", index=False)
    annotator_sheet.to_csv("data/human_eval_form_annotator_2.csv", index=False)
    
    # Save the master lock key for calculating Kappa later
    out_df.to_csv("data/human_eval_GROUND_TRUTH_KEY.csv", index=False)
    
    print("\n✅ Setup Complete!")
    print(f"   Generated {len(out_df)} grading pairs.")
    print(f"   Saved to: data/human_eval_form_annotator_1.csv and data/human_eval_form_annotator_2.csv")
    print(f"   (Hidden ground truth saved separately for Kappa calculation later)\n")

    print("""
    INSTRUCTIONS FOR GOOGLE SHEETS:
    1. Go to sheets.google.com
    2. Upload 'human_eval_form_annotator_1.csv'
    3. Share the link with Annotator 1 (Ask them to type 1 for Yes, 0 for No).
    4. Do the same for Annotator 2.
    5. Download both as CSV when they finish.
    """)

if __name__ == "__main__":
    generate_annotation_sample()
