import numpy as np
import random
import os

def set_seeds(seed=42):
    """Sets all relevant seeds for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Note: SentenceTransformers may have slight variance across CUDA versions, 
    # but on CPU it is largely deterministic.
    print(f"--- REPRODUCIBILITY SEED SET: {seed} ---")

if __name__ == "__main__":
    set_seeds()
