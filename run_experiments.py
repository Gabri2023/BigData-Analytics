import argparse
import subprocess
import yaml
import itertools
from pathlib import Path
import sys

# Parameter mapping for scenarios
SCENARIO_PARAM_MAP = {
    "A2_top": ["top_k"],
    "A2_bottom": ["bottom_k"],
    "B1_exact": ["exact_ged"],
    "B1_extreme_min": ["min_extreme_ged"],
    "B1_extreme_max": ["max_extreme_ged"],
    "B2_bottleneck": ["top_k_bottlenecks"]
}

def sanitize_dict(d):
    """
    Freezes unused lists by extracting the first element.
    This prevents downstream errors when unmapped parameters remain as lists.

    Args:
        d (dict): The configuration dictionary to sanitize.
    """
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            clean[k] = sanitize_dict(v)
        elif isinstance(v, list) and len(v) > 0:
            clean[k] = v[0]
        else:
            clean[k] = v
    return clean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset (e.g., 'fineExp').")
    parser.add_argument("--strategy", default="repair", choices=["infect", "repair"],
                        help="Choose whether to inject anomalies ('infect') or fix them ('repair').")
    parser.add_argument("--scenarios", type=str, nargs='+', required=True,
                        help="List of scenarios to combine (e.g., A2_top B2_bottleneck).")
    args = parser.parse_args()
    
    base_config_path = Path("config") / f"config_{args.dataset}.yaml"
    if not base_config_path.exists():
        print(f"[ERROR] Config file not found: {base_config_path}")
        sys.exit(1)

    with open(base_config_path, 'r') as file:
        base_config = yaml.safe_load(file)
        
    # Extract relevant parameters mapping for the requested scenarios
    active_keys = []
    for scenario in args.scenarios:
        if scenario in SCENARIO_PARAM_MAP:
            active_keys.extend(SCENARIO_PARAM_MAP[scenario])
            
    # Remove duplicates while preserving required keys
    active_keys = list(set(active_keys))
    
    # Extract only necessary lists from the base config
    grid_values = []
    for key in active_keys:
        val = base_config.get(key)
        if val is None:
            print(f"[WARNING] Parameter '{key}' not found in config. Skipping.")
            val = [1]
        elif not isinstance(val, list):
            val = [val]
        grid_values.append(val)
        
    # Generate Cartesian Product for Grid Search
    combinations = list(itertools.product(*grid_values))
    print(f"ORCHESTRATOR: Found {len(combinations)} parameter combinations to run.")
    
    temp_config_path = Path("config") / f"config_{args.dataset}_temp.yaml"
    combined_scenario_name = "+".join(args.scenarios)
    
    for combo in combinations:
        run_config = sanitize_dict(base_config)
        
        combo_str_parts = []
        for i, key in enumerate(active_keys):
            run_config[key] = combo[i]
            combo_str_parts.append(f"{key}_{combo[i]}")
            
        combo_str = " | ".join(combo_str_parts) if combo_str_parts else "No numeric params"
    
        # Save temporary YAML configuration file for main.py execution
        with open(temp_config_path, 'w') as file:
            yaml.dump(run_config, file)
        
        print("\n" + "="*70)
        print(f"Running combined scenarios: {combined_scenario_name} with config: {combo_str}")
        print("="*70 + "\n")
    
        # Execute main.py, injecting the temporary config and all scenarios
        cmd = [
            "python", "main.py",
            "--dataset", args.dataset,
            "--strategy", args.strategy,
            "--config", str(temp_config_path),
            "--scenario"
        ] + args.scenarios
    
        subprocess.run(cmd, check=True)
    
    # Cleanup temporary files
    if temp_config_path.exists():
        temp_config_path.unlink()
        
    print("\nORCHESTRATOR: All experiments completed successfully.")

if __name__ == "__main__":
    main()