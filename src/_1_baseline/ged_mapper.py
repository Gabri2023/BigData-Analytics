import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any

print("Loading SBERT model...")
# Loaded globally to avoid re-initializing it for every function call
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_graph_text(G: nx.DiGraph) -> str:
    """
    Extracts a 'sentence' from the graph by joining node labels according 
    to their topological sorting (causal execution order).
    """
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        sorted_nodes = sorted(G.nodes())
        
    return " ".join([G.nodes[n].get('label', '') for n in sorted_nodes])

def node_match(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    """Helper function for Graph Edit Distance computation."""
    return n1.get('label') == n2.get('label')

def get_features(anomalous_graphs: Dict[str, nx.DiGraph], 
                 correct_subgraphs: Dict[str, nx.DiGraph], 
                 target_subs: List[str], 
                 freq_dict: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """
    Computes structural (GED) and semantic (Cosine Similarity via SBERT) 
    distances between anomalous subgraphs and their closest correct counterpart.
    """
    features_dict = {}

    for anom_id in target_subs:
        if anom_id not in anomalous_graphs:
            print(f"[WARNING] Graph for {anom_id} missing from TXT file. Skipping.")
            continue
            
        G_anom = anomalous_graphs[anom_id]
        anom_node_count = len(G_anom.nodes())
        
        min_ged = float('inf')
        best_match_id = None
        best_match_G = None
        
        # 1. Find the most structurally similar correct subgraph (Minimum GED)
        for corr_id, G_corr in correct_subgraphs.items():
            corr_node_count = len(G_corr.nodes())
            
            # --- OPTIMIZATION 1: Absolute Minimum Bound ---
            # The GED can NEVER be smaller than the difference in node counts.
            # If the size difference alone is worse than our current best GED, skip entirely!
            if abs(anom_node_count - corr_node_count) >= min_ged:
                continue
            
            # --- OPTIMIZATION 2: Upper Bound & Timeout ---
            # It uses NetworkX to stop calculating if it exceeds our current min_ged,
            # and to give up if it takes more than 5 seconds for a single comparison.
            dist = nx.graph_edit_distance(
                G_anom, G_corr, 
                node_match=node_match, 
                upper_bound=min_ged,
                timeout=5.0
            )
            
            if dist is not None and dist < min_ged:
                min_ged = dist
                best_match_id = corr_id
                best_match_G = G_corr
                
        # Fallback if timeout prevented finding a match
        if best_match_G is None:
            # Just take the first one if everything timed out (rare fallback)
            best_match_id = list(correct_subgraphs.keys())[0]
            best_match_G = correct_subgraphs[best_match_id]
            min_ged = 99.0 # Placeholder for "too complex to calculate"
                
        # 2. Calculate the semantic similarity with the best structural match
        text_anom = get_graph_text(G_anom)
        text_corr = get_graph_text(best_match_G)
        
        embedding_anom = sbert_model.encode([text_anom])
        embedding_corr = sbert_model.encode([text_corr])
        sim_score = cosine_similarity(embedding_anom, embedding_corr)[0][0]
        
        # 3. Store the extracted features
        features_dict[anom_id] = {
            'ged': min_ged,
            'semantic_sim': sim_score,
            'freq': freq_dict[anom_id],
            'matched_with': best_match_id,
            'text_anom': text_anom, 
            'text_corr': text_corr
        }
        
        print(f"{anom_id} completed -> GED: {min_ged:.1f} | Sim: {sim_score:.3f} | Freq: {freq_dict[anom_id]}")
        
    return features_dict