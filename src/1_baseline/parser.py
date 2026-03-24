# Funzione per il parsing del formato 'S v 1 Label d 1 2' in oggetti NetworkX
def parse_subelements(file_content, custom_ids=None):
    """Parsa il formato 'S v 1 Label d 1 2' in oggetti NetworkX."""
    graphs = {}
    blocks = re.split(r'\nS\n|^S\n', file_content.strip())
    valid_blocks = [b for b in blocks if b.strip()]
    
    for i, block in enumerate(valid_blocks):
        G = nx.DiGraph()
        lines = block.strip().split('\n')
        
        # Assegnazione ID: usa custom_ids se forniti, altrimenti usa CorrSub_X
        if custom_ids and i < len(custom_ids):
            sub_id = custom_ids[i]
        else:
            sub_id = f"CorrSub_{i+1}"
            
        for line in lines:
            parts = line.split()
            if len(parts) < 3: continue
            
            # Controllo sul primo elemento della riga
            if parts[0] == 'v':
                G.add_node(int(parts[1]), label=parts[2])
            elif parts[0] in ['d', 'e']:
                G.add_edge(int(parts[1]), int(parts[2]))
                
        graphs[sub_id] = G
    return graphs
