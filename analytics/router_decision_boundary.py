import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# Re-import test queries and router logic from the eval script
# To do this safely without running the whole script, we just copy the dictionary.
TEST_QUERIES = {
    "STRICT_NEGATION": [
        "completely stateless layout container with zero hooks",
        "pure presentational badge with no state at all",
        "deeply nested layout that fetches no data and has no hooks",
        "text display component with no interactivity or state",
        "stateless wrapper with no context connection",
        "dumb component rendering props with zero hooks",
        "static svg icon wrapper with absolutely no hooks",
        "presentational avatar no state no effects",
        "simple typographic header with zero logic",
        "uncontrolled input shell with no internal state",
        "static footer layout with no stateful logic",
        "read-only display card without event handlers",
        "no-op wrapper with no hooks and no children modification",
        "pure prop-through component with zero hook imports",
        "completely without state functional component",
        "stateless nav bar item with no internal logic",
        "skeleton loader with no hooks",
        "no state purely structural page section",
        "immutable display component without context",
        "empty state placeholder with no hooks",
        "helper wrapper without any react hooks",
        "zero-state container purely rendering children",
        "caption text component without state or interactivity",
        "label component receiving no context no hooks",
        "totally stateless tag chip component",
    ],
    "EXACT_MECHANICS": [
        "complex global authentication provider using context",
        "global state reducer wrapper with context",
        "animated modal with refs and multiple event handlers",
        "virtualized list with map calls and memoization",
        "form with multiple conditional validations and state",
        "canvas element with refs and one event handler",
        "complex table with filtering and mapping",
        "managing multiple contexts simultaneously",
        "data transformation pipeline using useMemo and reduce",
        "heavy ref usage for DOM manipulation",
        "event intensive capturing clicks and keyboard input",
        "synchronizing state via multiple useEffects",
        "component with useReducer managing complex state transitions",
        "callback-memoized list with filter logic",
        "context consumer that also reduces data",
        "heavy boolean prop component with many flags",
        "data fetching component with state and side effects",
        "memoized computation with dependencies",
        "custom hook wrapper aggregating multiple hooks",
        "high interactivity form with callbacks on every input",
        "animated scrolling ref-based component",
        "provider wrapping children with two contexts",
        "map-filtered data list with memoization",
        "conditional tree fetching based on props",
        "dual-state toggle component with callbacks",
    ],
    "TRUNCATION_TRAP": [
        "extremely large data fetching dashboard over 150 lines",
        "massive complex form over 200 lines with many states",
        "giant monolithic page component with deep html nesting",
        "very long data grid with pagination state",
        "huge configuration panel with reducers and contexts over 150 lines",
        "extensive layout heavy component over 300 lines",
        "massive interactive map using refs over 150 lines",
        "complex multi-step wizard spanning hundreds of lines",
        "colossal document viewer with many effects over 200 lines",
        "giant charting component relying on memoization over 150 lines",
        "deeply recursive tree component with complex nesting",
        "monolithic sidebar with complex state over 200 lines",
        "very large admin panel over 250 lines with many hooks",
        "huge router component conditional rendering over 150 lines",
        "massive settings page fetching user data over 200 lines",
        "extensive table with sorting filtering mapping over 200 lines",
        "large modal with many form fields and validations over 150 lines",
        "massive calendar widget over 300 lines with date logic",
        "gigantic dashboard combining multiple widgets over 300 lines",
        "sprawling analytics page with charts and data fetch over 200 lines",
        "long data upload form with progress tracking over 150 lines",
        "enormous notification center with subscriptions over 200 lines",
        "huge report generator fetching and mapping data over 250 lines",
        "large multi-tab interface with complex state over 200 lines",
        "massive form wizard with context and reducer over 150 lines",
    ],
    "FUZZY_SEMANTIC": [
        "somewhat complex interactive form wrapper",
        "mostly simple display card with one state at most",
        "data grid heavy on logic but light on dom elements",
        "tiny basic generic button component under 50 lines",
        "standard small dropdown menu item",
        "typical user profile avatar display without hooks",
        "standard modal dialog with minimal state",
        "basic text input with one state",
        "simple toast notification under 60 lines",
        "typical breadcrumb navigation no hooks required",
        "standard loading spinner under 40 lines",
        "a common simple tooltip wrapper",
        "small accordion panel with one toggle state",
        "standard typography text with no hooks",
        "simple progress bar component with one prop",
        "small stepper indicator no state",
        "minimal tab header with one useState",
        "icon button with aria label no hooks",
        "basic close button with one callback",
        "compact tag label component without logic",
        "minimal divider component under 30 lines",
        "plain container div wrapper functional component",
        "small avatar with initials no hooks",
        "simple card footer with one prop and no logic",
        "generic wrapper with className pass-through",
    ]
}

STRUCTURAL_KEYWORDS = {
    'usestate','useeffect','usecontext','usememo','usecallback','useref','usereducer',
    'stateless','no hooks','zero hooks','without state','no state','without hooks',
    'absolutely no','no effects','no context','no fetch','without context','no event',
    'no interactivity','pure presentational','dumb component',
    'fetches','fetch','context','reducer','refs','dom',
    'over 150','over 200','over 300','150 lines','200 lines','300 lines',
    'hundreds of','monolithic','massive','colossal','giant','enormous','very long',
    'deeply nested','jsx depth','deep html','spanning hundreds',
    'map calls','filter call','reduce call','maps over','memoization','memoized',
    'callback','event handler','multiple contexts','global state','large data',
}

def classify_query(q):
    return 'structural' if any(kw in q.lower() for kw in STRUCTURAL_KEYWORDS) else 'fuzzy'

print("Encoding queries...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
queries = []
gt_labels = []

for cat, qs in TEST_QUERIES.items():
    for q in qs:
        queries.append(q)
        gt_labels.append('structural' if cat != 'FUZZY_SEMANTIC' else 'fuzzy')

embs = embedder.encode(queries)
pca = PCA(n_components=2, random_state=42)
embs_2d = pca.fit_transform(embs)

# Extract predictions
pred_labels = [classify_query(q) for q in queries]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

colors = {'structural': '#4c6ef5', 'fuzzy': '#fa5252'}
markers = {'structural': 'o', 'fuzzy': 'X'}

for i in range(len(queries)):
    gt = gt_labels[i]
    pred = pred_labels[i]
    
    # We use Ground Truth for the Color
    # We use Prediction for the Marker shape
    # Wait, it's easier to explicitly show: Correct vs Incorrect and Boundary
    
    # Actually, let's just color by GT, but circle matches and cross mismatches
    c = colors[gt]
    m = 'o' if gt == pred else 'X'
    edge = 'black' if gt == pred else 'darkred'
    size = 120 if gt == pred else 180
    alpha = 0.8
    
    ax.scatter(embs_2d[i, 0], embs_2d[i, 1], c=c, marker=m, edgecolor=edge, s=size, alpha=alpha, linewidth=1.5 if gt != pred else 0.5)

# Create custom legend
import matplotlib.lines as mlines
l1 = mlines.Line2D([], [], color='#4c6ef5', marker='o', linestyle='None', markersize=10, label='Structural (GT)')
l2 = mlines.Line2D([], [], color='#fa5252', marker='o', linestyle='None', markersize=10, label='Fuzzy (GT)')
l3 = mlines.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=12, markeredgecolor='darkred', label='Router Misclassification')

ax.legend(handles=[l1, l2, l3], loc='upper left', frameon=True)
ax.set_title("Anti-Gravity Router: Query Semantic Decision Boundary (PCA)", fontweight='bold')
ax.set_xlabel("Principal Component 1", fontweight='bold')
ax.set_ylabel("Principal Component 2", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs("analytics/plots", exist_ok=True)
plt.savefig("analytics/plots/router_decision_boundary.png", dpi=300)
print("Saved router decision boundary plot.")
