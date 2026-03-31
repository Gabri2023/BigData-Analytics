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
        # Check if the graph actually exists in the TXT
        if anom_id not in anomalous_graphs:
            print(f"[WARNING] Graph for {anom_id} missing from TXT file. Skipping.")
            continue
            
        G_anom = anomalous_graphs[anom_id]
        
        min_ged = float('inf')
        best_match_id = None
        best_match_G = None
        
        # 1. Find the most structurally similar correct subgraph (Minimum GED)
        for corr_id, G_corr in correct_subgraphs.items():
            dist = nx.graph_edit_distance(G_anom, G_corr, node_match=node_match)
            
            if dist < min_ged:
                min_ged = dist
                best_match_id = corr_id
                best_match_G = G_corr
                
        # 2. Calculate the semantic similarity with the best structural match
        text_anom = get_graph_text(G_anom)
        text_corr = get_graph_text(best_match_G)
        
        embedding_anom = sbert_model.encode([text_anom])
        embedding_corr = sbert_model.encode([text_corr])
        sim_score = cosine_similarity(embedding_anom, embedding_corr)[0][0]
        
        # 3. Store the extracted features along with the raw frequency
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