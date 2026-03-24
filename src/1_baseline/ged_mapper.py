# ==========================================
# BLOCCO 3: Estrazione Feature (GED e SBERT)
# ==========================================
print("Caricamento del modello SBERT...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_graph_text(G):
    """Estrae una 'frase' dal grafo seguendo l'ordinamento topologico delle attività"""
    try:
        # Se è un DAG, usiamo l'ordinamento topologico (causale)
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Se ci sono dei cicli (loop), facciamo un fallback sui nodi ordinati per ID
        sorted_nodes = sorted(G.nodes())
        
    return " ".join([G.nodes[n].get('label', '') for n in sorted_nodes])

# Funzione per matchare i nodi durante il calcolo della GED
def node_match(n1, n2):
    return n1.get('label') == n2.get('label')

def get_features(anomalous_graphs, correct_subgraphs, target_subs, freq_dict):
    features_dict = {}

    for anom_id in target_subs:
        G_anom = anomalous_graphs[anom_id]
        
        min_ged = float('inf')
        best_match_id = None
        best_match_G = None
        
        # 1. Trova il sottografo corretto più simile strutturalmente (GED minima)
        for corr_id, G_corr in correct_subgraphs.items():
            dist = nx.graph_edit_distance(G_anom, G_corr, node_match=node_match)
            
            if dist < min_ged:
                min_ged = dist
                best_match_id = corr_id
                best_match_G = G_corr
                
        # 2. Calcola la similarità semantica con il best match
        text_anom = get_graph_text(G_anom)
        text_corr = get_graph_text(best_match_G)
        
        embedding_anom = sbert_model.encode([text_anom])
        embedding_corr = sbert_model.encode([text_corr])
        sim_score = cosine_similarity(embedding_anom, embedding_corr)[0][0]
        
        # 3. Salviamo le feature (aggiungendo anche la frequenza grezza dal dizionario precedente)
        features_dict[anom_id] = {
            'ged': min_ged,
            'semantic_sim': sim_score,
            'freq': freq_dict[anom_id],
            'matched_with': best_match_id,
            'text_anom': text_anom, 
            'text_corr': text_corr
        }
        
        print(f"{anom_id} completato -> GED: {min_ged:.1f} | Sim: {sim_score:.3f} | Freq: {freq_dict[anom_id]}")
        
    return features_dict
