import pm4py
from typing import List, Union, Tuple
from pathlib import Path

def extract_process_metrics(log_path: Union[str, Path], top_k: int = 3) -> Tuple[List[str], List[str], List[str]]:
    """
    Reads an Event Log and automatically calculates:
    1. The major temporal bottlenecks (based on average waiting time).
    2. The early activities (start nodes of the process).
    3. The late activities (end nodes of the process).
    
    Args:
        log_path (Union[str, Path]): The path to the .xes file.
        top_k (int): The number of bottlenecks to extract (e.g., the top 3 worst).
        
    Returns:
        Tuple[List[str], List[str], List[str]]: 
            - bottleneck_labels: List of activities acting as bottlenecks.
            - early_labels: List of start activities.
            - late_labels: List of end activities.
    """
    print(f"Automatically computing metrics for log: {log_path} ...")
    
    # 1. Load the event log
    log = pm4py.read_xes(str(log_path), return_legacy_log_object=False)
    
    # 2. Discover the Directly-Follows Graph (DFG)
    dfg, start_activities, end_activities = pm4py.discover_performance_dfg(log)
    
    # --- BOTTLENECKS CALCULATION ---
    # 3. Aggregate waiting times for each target node
    node_delays = {}
    node_incoming_count = {}
    
    for (source, target), edge_data in dfg.items():
        # ROBUST FIX: Handle both old (float) and new (dict) pm4py behaviors
        if isinstance(edge_data, dict):
            # Extract 'mean' (or fallback to 'median' or 0.0 if not found)
            avg_time_seconds = float(edge_data.get('mean', edge_data.get('median', 0.0)))
        else:
            avg_time_seconds = float(edge_data)
            
        if target not in node_delays:
            node_delays[target] = 0.0
            node_incoming_count[target] = 0
            
        node_delays[target] += avg_time_seconds
        node_incoming_count[target] += 1
            
    # 4. Calculate the average INCOMING waiting time for that activity
    avg_node_delays = {}
    for node in node_delays:
        if node_incoming_count[node] > 0:
            avg_node_delays[node] = node_delays[node] / node_incoming_count[node]
        else:
            avg_node_delays[node] = 0.0
        
    # 5. Sort activities from slowest to fastest and extract the Top K
    sorted_nodes = sorted(avg_node_delays.items(), key=lambda item: item[1], reverse=True)
    bottleneck_labels = [node for node, delay in sorted_nodes[:top_k]]
    
    # --- EARLY & LATE ACTIVITIES EXTRACTION ---
    early_labels = list(start_activities.keys())
    late_labels = list(end_activities.keys())
    
    # --- CONSOLE OUTPUT ---
    print(f"\n--- PROCESS METRICS DETECTED ---")
    print(f"[!] Top {top_k} Bottlenecks : {bottleneck_labels}")
    for node, delay_sec in sorted_nodes[:top_k]:
        print(f"    -> {node}: {delay_sec / 86400:.1f} days avg delay")
        
    print(f"[>] Early Activities  : {early_labels}")
    print(f"[<] Late Activities   : {late_labels}")
    print("--------------------------------\n")
        
    return bottleneck_labels, early_labels, late_labels

if __name__ == "__main__":
    bottlenecks, early, late = extract_process_metrics("data/fineExp/fineExp.xes", top_k=2)