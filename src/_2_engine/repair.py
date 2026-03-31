import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from typing import List, Dict, Any
from copy import deepcopy
import networkx as nx
from datetime import timedelta

def _get_label_sequence(graph: nx.DiGraph) -> List[str]:
    """Extracts the ordered sequence of labels from a Directed Graph."""
    try:
        sorted_nodes = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        sorted_nodes = sorted(graph.nodes())
    return [graph.nodes[n].get('label', '') for n in sorted_nodes]

def _find_subsequence(sequence: List[str], subseq: List[str]) -> int:
    """Finds the starting index of a contiguous subsequence in a sequence. Returns -1 if not found."""
    n, m = len(sequence), len(subseq)
    for i in range(n - m + 1):
        if sequence[i:i+m] == subseq:
            return i
    return -1

def run_repair(log: EventLog, 
               anomalous_graphs: Dict[str, nx.DiGraph], 
               correct_subgraphs: Dict[str, nx.DiGraph], 
               features_dict: Dict[str, Dict[str, Any]], 
               target_anomalies: List[str]) -> EventLog:
    """
    Scans the Event Log and replaces occurrences of target anomalies with their correct counterparts.
    
    Args:
        log (EventLog): The original pm4py Event Log.
        anomalous_graphs (Dict): Dictionary of anomalous NetworkX subgraphs.
        correct_subgraphs (Dict): Dictionary of correct NetworkX subgraphs.
        features_dict (Dict): Dictionary containing the mapping (produced in Phase 1).
        target_anomalies (List[str]): List of anomalous IDs to repair in this specific scenario.
        
    Returns:
        EventLog: A new, altered pm4py Event Log.
    """
    print(f"Starting REPAIR engine for {len(target_anomalies)} target anomalies...")
    
    # Create a deepcopy to avoid modifying the original log in memory
    repaired_log = deepcopy(log)
    
    # Pre-compute the label sequences for fast searching
    repair_mapping = {}
    for anom_id in target_anomalies:
        if anom_id not in features_dict:
            continue
            
        corr_id = features_dict[anom_id]['matched_with']
        seq_anom = _get_label_sequence(anomalous_graphs[anom_id])
        seq_corr = _get_label_sequence(correct_subgraphs[corr_id])
        
        repair_mapping[anom_id] = {
            'anom_seq': seq_anom,
            'corr_seq': seq_corr
        }

    traces_modified = 0

    for trace in repaired_log:
        trace_labels = [event["concept:name"] for event in trace]
        trace_was_modified = False
        
        # Check for each target anomaly in the trace
        for anom_id, mapping in repair_mapping.items():
            anom_seq = mapping['anom_seq']
            corr_seq = mapping['corr_seq']
            
            # Find where the anomaly starts in the trace
            start_idx = _find_subsequence(trace_labels, anom_seq)
            
            while start_idx != -1:
                end_idx = start_idx + len(anom_seq) - 1
                
                # Extract timestamps to preserve the timeframe
                start_time = trace[start_idx]["time:timestamp"]
                end_time = trace[end_idx]["time:timestamp"]
                
                # Calculate time step for new events
                num_new_events = len(corr_seq)
                time_diff = end_time - start_time
                step = time_diff / max(1, (num_new_events - 1)) if num_new_events > 1 else timedelta(0)
                
                # Create new correct events
                new_events = []
                for i, label in enumerate(corr_seq):
                    new_event = Event()
                    new_event["concept:name"] = label
                    new_event["time:timestamp"] = start_time + (step * i)
                    # Add any mandatory custom attributes your log might have (e.g., Resource)
                    new_event["lifecycle:transition"] = "complete" 
                    new_events.append(new_event)
                
                # Replace the slice in the trace
                trace[start_idx : end_idx + 1] = new_events
                trace_was_modified = True
                
                # Update trace_labels for the next iteration (in case of multiple anomalies in one trace)
                trace_labels = [event["concept:name"] for event in trace]
                
                # Search again in case the same anomaly happens twice in the same trace
                start_idx = _find_subsequence(trace_labels, anom_seq)
                
        if trace_was_modified:
            traces_modified += 1

    print(f"Repair complete. Modified {traces_modified} out of {len(repaired_log)} traces.")
    return repaired_log