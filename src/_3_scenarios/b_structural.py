from typing import Dict, List, Any
import networkx as nx

def filter_by_ged(features_dict: Dict[str, Dict[str, Any]], 
                  exact_ged: float = None, 
                  min_ged: float = None, 
                  max_ged: float = None) -> List[str]:
    """
    Scenario B.1 - Anomaly Complexity (GED Thresholds).
    Filters anomalies based on their Graph Edit Distance (GED) from the correct subgraph.
    
    Args:
        features_dict (Dict): The dictionary containing all features.
        exact_ged (float, optional): Selects anomalies with this exact GED (e.g., 2).
        min_ged (float, optional): Minimum GED threshold.
        max_ged (float, optional): Maximum GED threshold.
        
    Returns:
        List[str]: A list of anomalous subgraph IDs matching the GED criteria.
    """
    selected = []
    
    for anom_id, features in features_dict.items():
        ged = features['ged']
        
        if exact_ged is not None:
            if ged == exact_ged:
                selected.append(anom_id)
        else:
            # Check within min and max bounds if provided
            if (min_ged is None or ged >= min_ged) and (max_ged is None or ged <= max_ged):
                selected.append(anom_id)
                
    print(f"Scenario B.1 (GED Filter): Selected {len(selected)} anomalies.")
    return selected


def filter_by_bottleneck(features_dict: Dict[str, Dict[str, Any]], 
                         anomalous_graphs: Dict[str, nx.DiGraph], 
                         bottleneck_nodes: List[str]) -> List[str]:
    """
    Scenario B.2 - Critical Nodes (Bottlenecks).
    Selects anomalies that contain at least one of the specified bottleneck activities.
    
    Args:
        features_dict (Dict): The dictionary containing all features.
        anomalous_graphs (Dict): Dictionary of anomalous NetworkX subgraphs.
        bottleneck_nodes (List[str]): List of activity labels considered bottlenecks.
        
    Returns:
        List[str]: IDs of anomalies involving bottleneck activities.
    """
    selected = []
    
    for anom_id in features_dict.keys():
        graph = anomalous_graphs[anom_id]
        
        # Extract all activity labels present in the anomalous subgraph
        labels = [data.get('label', '') for _, data in graph.nodes(data=True)]
        
        # Check if there is any intersection between the subgraph labels and the bottlenecks
        if any(b_node in labels for b_node in bottleneck_nodes):
            selected.append(anom_id)
            
    print(f"Scenario B.2 (Bottlenecks): Selected {len(selected)} anomalies involving {bottleneck_nodes}.")
    return selected


def filter_by_position(features_dict: Dict[str, Dict[str, Any]], 
                       anomalous_graphs: Dict[str, nx.DiGraph], 
                       target_activities: List[str],
                       position_name: str = "Target") -> List[str]:
    """
    Scenario B.3 - Topological Position (Early / Late).
    Selects anomalies that involve specific early or late activities in the process.
    
    Args:
        features_dict (Dict): The dictionary containing all features.
        anomalous_graphs (Dict): Dictionary of anomalous NetworkX subgraphs.
        target_activities (List[str]): List of early (e.g., 'CreateFine') or late activities.
        position_name (str): Label for printing purposes (e.g., "Early" or "Late").
        
    Returns:
        List[str]: IDs of anomalies occurring in the specified topological position.
    """
    selected = []
    
    for anom_id in features_dict.keys():
        graph = anomalous_graphs[anom_id]
        labels = [data.get('label', '') for _, data in graph.nodes(data=True)]
        
        if any(t_node in labels for t_node in target_activities):
            selected.append(anom_id)
            
    print(f"Scenario B.3 ({position_name} Position): Selected {len(selected)} anomalies.")
    return selected