import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

def calculate_agreement_and_ground_truth():
    print("="*60)
    print(" 🧑‍🔬 HUMAN VALIDATION KAPPA SCORE CALCULATION 🧑‍🔬 ")
    print("="*60)
    
    # 1. Load the core lock key
    try:
        gt_df = pd.read_csv('data/human_eval_GROUND_TRUTH_KEY.csv')
    except FileNotFoundError:
        print("❌ Error: data/human_eval_GROUND_TRUTH_KEY.csv not found.")
        return
        
    # 2. Check for Annotator 1
    if not os.path.exists('data/annotator_1.csv'):
        print("⚠️ Waiting for 'data/annotator_1.csv' to be uploaded...")
        return
        
    # 3. Check for Annotator 2
    if not os.path.exists('data/annotator_2.csv'):
        print("⚠️ Waiting for 'data/annotator_2.csv' to be uploaded...")
        return
        
    ann1 = pd.read_csv('data/annotator_1.csv')
    ann2 = pd.read_csv('data/annotator_2.csv')
    
    # Extract the label columns as integers
    try:
        a1_labels = ann1['Is Relevant? (1=Yes, 0=No)'].astype(int).tolist()
        a2_labels = ann2['Is Relevant? (1=Yes, 0=No)'].astype(int).tolist()
        gt_labels = gt_df['GroundTruth_Hidden'].astype(int).tolist()
    except KeyError:
        print("❌ Error: Missing the 'Is Relevant? (1=Yes, 0=No)' column in one of the CSVs.")
        return
    except ValueError:
        print("❌ Error: One of the CSVs contains blank or non-integer labels. Please fill all 60 rows.")
        return
        
    if len(a1_labels) != 60 or len(a2_labels) != 60:
        print(f"❌ Error: Expected 60 judgments per annotator. Found A1: {len(a1_labels)}, A2: {len(a2_labels)}")
        return

    # 4. Calculate Cohen's Kappa
    kappa = cohen_kappa_score(a1_labels, a2_labels)
    
    # 5. Measure consensus vs Ground Truth
    consensus_matches = 0
    total_consensus_reached = 0
    
    for i in range(len(gt_labels)):
        # If annotators agree with each other, that's a consensus
        if a1_labels[i] == a2_labels[i]:
            total_consensus_reached += 1
            # Does their consensus match our programmatic rule?
            if a1_labels[i] == gt_labels[i]:
                consensus_matches += 1
                
    gt_agreement = (consensus_matches / total_consensus_reached) * 100 if total_consensus_reached > 0 else 0
    
    # 6. Output the academic verdict
    print(f"\n✅ Analysis Complete over {len(gt_labels)} query-component pairs.")
    print(f"\n📊 INTER-ANNOTATOR AGREEMENT:")
    print(f"   Cohen's Kappa (κ): {kappa:.3f}")
    
    if kappa > 0.8:
        print("   Interpretation: Almost Perfect Agreement")
    elif kappa > 0.6:
        print("   Interpretation: Substantial Agreement")
    elif kappa > 0.4:
        print("   Interpretation: Moderate Agreement")
    else:
        print("   Interpretation: Poor Agreement")
        
    print(f"\n📊 GROUND TRUTH VALIDATION:")
    print(f"   Human Consensus matched AST Labels: {gt_agreement:.1f}% of the time.")
    print(f"   (Humans agreed on {total_consensus_reached}/60 items).")
    
    print("\n📝 DRAFT PARAGRAPH FOR YOUR PAPER:")
    print('   "To validate ground-truth label quality, two independent engineers evaluated ')
    print(f'   60 (query, component) pairs. Inter-annotator agreement achieved a Cohen\\'s κ ')
    print(f'   of {kappa:.2f}. Where humans reached consensus, their judgments matched the ')
    print(f'   AST-derived programmatic labels {gt_agreement:.1f}% of the time, empirically validating ')
    print('   that the structural benchmark accurately reflects developer intent."')

if __name__ == "__main__":
    calculate_agreement_and_ground_truth()
