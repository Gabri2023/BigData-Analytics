import pandas as pd
from typing import Dict, Union
from pathlib import Path

def extract_frequencies(csv_path: Union[str, Path]) -> Dict[str, int]:
    """
    Reads a CSV file containing subgraph occurrences and extracts their frequencies.
    
    Args:
        csv_path (Union[str, Path]): Path to the CSV file.
        
    Returns:
        Dict[str, int]: A dictionary mapping subgraph IDs (columns) to their total 
                        occurrences (sum > 0).
    """
    # Load the dataset using semicolon separator
    df = pd.read_csv(csv_path, sep=';')

    # Drop the 'grafo' column to sum only the numerical subgraph occurrences
    # Using errors='ignore' prevents crashes if the column doesn't exist in some datasets
    subgraphs_only = df.drop(columns=['grafo'], errors='ignore')

    # Sum occurrences column-wise and filter out those with 0 occurrences
    occurrences = subgraphs_only.sum()
    occurrences = occurrences[occurrences > 0]

    # Convert the resulting Pandas Series to a standard Python dictionary
    freq_dict = occurrences.to_dict()

    print(f"Number of anomalous subgraphs found in CSV: {len(freq_dict)}")
    print(f"List of extracted IDs:\n{list(freq_dict.keys())}")
    print("-" * 30)
    
    return freq_dict