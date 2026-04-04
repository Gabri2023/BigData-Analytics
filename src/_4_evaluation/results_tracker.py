import pandas as pd
from pathlib import Path
from typing import Dict, Any

def update_results_matrix(matrix_path: Path, 
                          dataset: str, 
                          strategy: str, 
                          scenario: str, 
                          modified_traces: int, 
                          metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Updates the central CSV matrix with the results of the latest experiment.
    If the file or its parent directories don't exist, it creates them.
    """
    print(f"Updating results matrix at {matrix_path}...")
    
    # Create the new row as a dictionary
    new_row = {
        'Dataset': dataset,
        'Strategy': strategy.upper(),
        'Scenario': scenario,
        'Modified_Traces': modified_traces,
        'Fitness': round(metrics['fitness'], 4),
        'Precision': round(metrics['precision'], 4),
        'Generalization': round(metrics['generalization'], 4),
        'Simplicity': round(metrics['simplicity'], 4)
    }
    
    # If the file exists, load it. Otherwise, create an empty dataframe.
    if matrix_path.exists():
        df = pd.read_csv(matrix_path)
    else:
        df = pd.DataFrame(columns=new_row.keys())
        
    # Check if this exact experiment already exists (to overwrite it instead of duplicating)
    condition = (df['Dataset'] == dataset) & (df['Strategy'] == strategy.upper()) & (df['Scenario'] == scenario)
    
    if condition.any():
        print("[INFO] Overwriting existing experiment results in the matrix.")
        idx = df.index[condition].tolist()[0]
        for key, value in new_row.items():
            df.at[idx, key] = value
    else:
        # Append the new row using modern pandas concatenation
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
    # Ensure the target directory (e.g., 'results/') exists before saving
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save back to CSV
    df.to_csv(matrix_path, index=False)
    print("Matrix updated successfully!")
    
    return df

def is_baseline_calculated(matrix_path: Path, dataset: str) -> bool:
    """Checks if the baseline metrics for a dataset are already in the matrix."""
    if not matrix_path.exists():
        return False
    df = pd.read_csv(matrix_path)
    condition = (df['Dataset'] == dataset) & (df['Strategy'] == 'BASELINE')
    return condition.any()