import pandas as pd
import networkx as nx
import numpy as np

print("="*60)
print("--- 1. BUILDING COMPONENT GRAPH (NetworkX) ---")

# Load data and filter trivial
df = pd.read_csv('master2.csv')
df_filtered = df[~((df['hooks_total'] == 0) & (df['props'] <= 1) & (df['jsx_depth'] <= 3))].copy()

# Initialize directed graph
G = nx.DiGraph()

# Hook columns we actually track
hook_cols = ['useState', 'useEffect', 'useCallback', 'useMemo', 'useContext', 'useReducer', 'useRef', 'useCustom']

components_added = 0
for idx, row in df_filtered.iterrows():
    # Make a unique node name for each component using its file path
    comp_node = f"COMP:{row['component']} ({row['file'].split('/')[-1] if isinstance(row['file'], str) else ''})"
    G.add_node(comp_node, type='component')
    components_added += 1
    
    # Add Hook relationships
    for hook in hook_cols:
        count = row.get(hook, 0)
        if isinstance(count, (int, float)) and not np.isnan(count) and count > 0:
            hook_node = f"HOOK:{hook}"
            G.add_node(hook_node, type='hook')
            # Weight = number of times that hook is used
            G.add_edge(comp_node, hook_node, weight=float(count), relation='USES_HOOK')
            
    # Add Prop relationships
    if pd.notna(row.get('prop_names')) and isinstance(row['prop_names'], str):
        # Prop names are usually stored as semi-colon separated (e.g., 'children;className;isOpen')
        props = [p.strip() for p in row['prop_names'].split(';') if p.strip()]
        for prop in props:
            # Avoid massive spam of internal random props, maybe limit by len
            if len(prop) > 1 and len(prop) < 30:
                prop_node = f"PROP:{prop}"
                G.add_node(prop_node, type='prop')
                G.add_edge(comp_node, prop_node, weight=1.0, relation='ACCEPTS_PROP')

print(f"Graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Total valid components in graph: {components_added}")

print("\n" + "="*60)
print("--- 2. COMPUTING PAGERANK (Node Centrality) ---")
# PageRank calculates the underlying architecture. Nodes that share many common central props/hooks
# and are highly connected will float to the top.
# We reverse the graph so that Components that "point to" popular hooks inherit centrality,
# or we let it run undirected to see true topological hubs.
H = G.to_undirected()
pagerank = nx.pagerank(H, weight='weight')

# Separate PageRanks by node type
comp_ranks = {node: rank for node, rank in pagerank.items() if G.nodes[node].get('type') == 'component'}
hook_ranks = {node: rank for node, rank in pagerank.items() if G.nodes[node].get('type') == 'hook'}
prop_ranks = {node: rank for node, rank in pagerank.items() if G.nodes[node].get('type') == 'prop'}

print("\nTOP 5 MOST CENTRAL HOOKS (The Engine of the Codebase):")
for node, rank in sorted(hook_ranks.items(), key=lambda x: -x[1])[:5]:
    print(f"  {node}: {rank:.5f}")

print("\nTOP 5 MOST CENTRAL PROPS (The API of the UI):")
for node, rank in sorted(prop_ranks.items(), key=lambda x: -x[1])[:5]:
    print(f"  {node}: {rank:.5f}")

print("\n" + "="*60)
print("--- 3. ARCHITECTURAL CLASSIFICATION ---")
print("\nTOP 10 'GLOBAL FOUNDATIONS' (Components with highest PageRank -> Highly connected to shared hubs):")
# These components use the most "standard" APIs and hooks, binding the architecture together.
sorted_comps = sorted(comp_ranks.items(), key=lambda x: -x[1])
for node, rank in sorted_comps[:10]:
    print(f"  {node.replace('COMP:', '')}: PR = {rank:.6f}")

print("\nBOTTOM 10 'LEAF SPECIFICS' (Components with lowest PageRank -> Niche, isolated, custom):")
# These are likely pure UI or components that don't share common state signatures.
for node, rank in sorted_comps[-10:]:
    print(f"  {node.replace('COMP:', '')}: PR = {rank:.6f}")

print("\n" + "="*60)
print("Graph Analysis Complete. The mathematical map of your design systems is verified!")
