from typing import Dict, List, Any

def filter_all_anomalies(features_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Scenario A.1 - Global Modification (All-in).
    Selects all available anomalies to evaluate the absolute maximum impact on the metrics.
    
    Args:
        features_dict (Dict): The dictionary containing all features for each anomaly.
        
    Returns:
        List[str]: A list of all anomalous subgraph IDs.
    """
    print(f"Scenario A.1: Selected ALL {len(features_dict)} anomalies.")
    return list(features_dict.keys())

def filter_top_k_frequent(features_dict: Dict[str, Dict[str, Any]], k: int = 5) -> List[str]:
    """
    Scenario A.2 (Top-K) - Act by Frequency.
    Selects the top 'K' most frequent anomalous subgraphs.
    Useful to identify high-volume variations that might be standard practices.
    
    Args:
        features_dict (Dict): The dictionary containing all features.
        k (int): The number of top anomalies to select. Defaults to 5.
        
    Returns:
        List[str]: A list of the top K most frequent anomalous subgraph IDs.
    """
    # Sort anomalies by frequency in descending order
    sorted_anomalies = sorted(
        features_dict.items(), 
        key=lambda item: item[1]['freq'], 
        reverse=True
    )
    
    # Extract just the IDs
    top_k_ids = [anom_id for anom_id, _ in sorted_anomalies[:k]]
    
    print(f"Scenario A.2 (Top-{k}): Selected anomalies {top_k_ids}.")
    return top_k_ids

def filter_bottom_k_frequent(features_dict: Dict[str, Dict[str, Any]], k: int = 5) -> List[str]:
    """
    Scenario A.2 (Bottom-K) - Act by Frequency.
    Selects the bottom 'K' least frequent anomalous subgraphs.
    Useful to isolate rare noise or isolated user errors.
    
    Args:
        features_dict (Dict): The dictionary containing all features.
        k (int): The number of bottom anomalies to select. Defaults to 5.
        
    Returns:
        List[str]: A list of the bottom K least frequent anomalous subgraph IDs.
    """
    # Sort anomalies by frequency in ascending order
    sorted_anomalies = sorted(
        features_dict.items(), 
        key=lambda item: item[1]['freq'], 
        reverse=False
    )
    
    # Extract just the IDs
    bottom_k_ids = [anom_id for anom_id, _ in sorted_anomalies[:k]]
    
    print(f"Scenario A.2 (Bottom-{k}): Selected anomalies {bottom_k_ids}.")
    return bottom_k_ids