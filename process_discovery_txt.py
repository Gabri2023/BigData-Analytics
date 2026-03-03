import os
import pandas as pd
import pm4py
from collections import defaultdict, deque

# ======== CONFIG ========
txt_file = "datasets/custom/anomalous_sub.txt"
output_folder = "Modello_Anomalous"
os.makedirs(output_folder, exist_ok=True)

# ======== 1️⃣ Parsing sottografi ========

def parse_subgraphs(txt_file):
    subgraphs = []
    current_nodes = {}
    current_edges = []
    
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            
            if line == "S":
                if current_nodes:
                    subgraphs.append((current_nodes, current_edges))
                current_nodes = {}
                current_edges = []
            
            elif line.startswith("v"):
                parts = line.split()
                node_id = int(parts[1])
                activity = parts[2]
                current_nodes[node_id] = activity
            
            elif line.startswith("d") or line.startswith("e"):
                parts = line.split()
                source = int(parts[1])
                target = int(parts[2])
                current_edges.append((source, target))
        
        if current_nodes:
            subgraphs.append((current_nodes, current_edges))
    
    return subgraphs


# ======== 2️⃣ Topological sort ========

def topological_sort(nodes, edges):
    graph = defaultdict(list)
    indegree = {n: 0 for n in nodes}
    
    for src, tgt in edges:
        graph[src].append(tgt)
        indegree[tgt] += 1
    
    queue = deque([n for n in nodes if indegree[n] == 0])
    ordered = []
    
    while queue:
        n = queue.popleft()
        ordered.append(n)
        
        for neigh in graph[n]:
            indegree[neigh] -= 1
            if indegree[neigh] == 0:
                queue.append(neigh)
    
    if len(ordered) != len(nodes):
        raise ValueError("Errore: il sottografo contiene cicli o parallelismi non lineari.")
    
    return ordered


# ======== 3️⃣ Conversione in DataFrame ========

print("Parsing sottografi...")
subgraphs = parse_subgraphs(txt_file)

rows = []

for case_id, (nodes, edges) in enumerate(subgraphs):
    ordered_nodes = topological_sort(nodes.keys(), edges)
    
    import datetime

    base_time = datetime.datetime(2020, 1, 1)

    for idx, node_id in enumerate(ordered_nodes):
        rows.append({
            "case_id": case_id,
            "activity": nodes[node_id],
            "timestamp": base_time + datetime.timedelta(seconds=idx)
        })

df = pd.DataFrame(rows)

print(f"Sottografi convertiti in {len(subgraphs)} trace.")


# ======== 4️⃣ Creazione Event Log PM4Py ========

print("Creazione Event Log PM4Py...")
log = pm4py.format_dataframe(
    df,
    case_id="case_id",
    activity_key="activity",
    timestamp_key="timestamp"
)

log = pm4py.convert_to_event_log(log)


# ======== 5️⃣ Process Discovery ========

print("Scoperta ProcessTree con Inductive Miner...")
ptree = pm4py.discover_process_tree_inductive(log, noise_threshold=0.2)

print("Conversione in Petri Net...")
net, im, fm = pm4py.convert_to_petri_net(ptree)


# ======== 6️⃣ Esportazione ========

pnml_path = os.path.join(output_folder, "petri_net_anomalous.pnml")
png_path = os.path.join(output_folder, "petri_net_anomalous.png")

pm4py.write_pnml(net, im, fm, pnml_path)
pm4py.save_vis_petri_net(net, im, fm, png_path)

print("Completato.")
print(f"PNML salvato in: {pnml_path}")
print(f"PNG salvato in: {png_path}")