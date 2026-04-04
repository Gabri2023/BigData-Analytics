import argparse
from pathlib import Path
import yaml
import pm4py
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Imports from PHASE 1 ---
from src._1_baseline.bottleneck_extractor import extract_process_metrics
from src._1_baseline.frequencies_extractor import extract_frequencies
from src._1_baseline.ged_mapper import get_features
from src._1_baseline.parser import parse_subelements

# --- Imports from PHASE 2 ---
# from src._2_engine.infect import run_infect
from src._2_engine.repair import run_repair

# --- Imports from PHASE 3 ---
from src._3_scenarios.a_global_frequency import filter_all_anomalies, filter_top_k_frequent, filter_bottom_k_frequent
from src._3_scenarios.b_structural import filter_by_ged, filter_by_bottleneck, filter_by_position

# --- Imports from PHASE 4 ---
from src._4_evaluation.metrics_calculator import evaluate_model
from src._4_evaluation.results_tracker import update_results_matrix, is_baseline_calculated

def main():
    # 1. Command Line Arguments Configuration
    parser = argparse.ArgumentParser(description="Log Alteration Engine - Process Mining")
    
    # Parameter for the dataset name (required)
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset name (e.g., 'fineExp'). Used to resolve all file paths.")
    
    # Parameter for the strategy (Infect or Repair)
    parser.add_argument("--strategy", type=str, required=True, choices=["infect", "repair"],
                        help="Choose whether to inject anomalies ('infect') or fix them ('repair').")

    # Parameter for the scenario
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario code to execute (e.g., A1, A2_top, B2_exact, B2_bottleneck).")
    
    # Optional parameter for metrics
    parser.add_argument("--recalc-baseline", action="store_true",
                        help="If set, recalculates baseline metrics before execution.")
    
    args = parser.parse_args()

    # 2. Dynamic Path Construction
    dataset_name = args.dataset
    base_data_path = Path("data") / dataset_name
    
    log_path = base_data_path / f"{dataset_name}.xes"
    csv_path = base_data_path / f"{dataset_name}_table2_on_file.csv"
    anom_path = base_data_path / "custom" / "anomalous_sub.txt"
    corr_path = base_data_path / "custom" / "correct_sub.txt"
    config_path = Path("config") / f"config_{dataset_name}.yaml"
    pnml_path = base_data_path / "models_raw" / f"petri_net_{dataset_name}.pnml"
    matrix_path = Path("results") / "experiments_matrix.csv"
        
    output_path = base_data_path / "custom" / "processed" / f"{dataset_name}_{args.strategy}_{args.scenario}.xes"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Experiment ---")
    print(f"Dataset: {dataset_name}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Scenario: {args.scenario}")
    print(f"Log Path: {log_path}")
    print(f"---------------------------\n")
    
    # 3. File Existence Checks
    for path, name in [(log_path, "Log"),
                       (csv_path, "CSV"),
                       (anom_path, "Anomalous TXT"),
                       (corr_path, "Correct TXT"),
                       (config_path, "Config YAML"),
                       (pnml_path, "Baseline Petri Net PNML")]:
        if not path.exists():
            print(f"Error: {name} file not found at {path}")
            sys.exit(1)
    
    # 4. Load Config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # -----------------------------------
    # --- PHASE 1: Loading & Baseline ---
    # -----------------------------------
    original_log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)
    auto_bottlenecks, auto_early, auto_late = extract_process_metrics(log_path, top_k=config.get('top_k_bottlenecks', 3))
    freq_dict = extract_frequencies(csv_path)
    anomaly_ids = list(freq_dict.keys())
    anom_graphs = parse_subelements(anom_path, custom_ids=anomaly_ids)
    corr_graphs = parse_subelements(corr_path)
    features_dict = get_features(anom_graphs, corr_graphs, anomaly_ids, freq_dict)

    # -----------------------------------
    # --- PHASE 3: Scenario Selection ---
    # -----------------------------------
    scenario_params = ""
    match args.scenario:
        # --- A SCENARIOS ---
        case "A1":
            target_anomalies = filter_all_anomalies(features_dict)
            scenario_params = "all"
        case "A2_top":
            target_anomalies = filter_top_k_frequent(features_dict, k=config.get('top_k', 5))
            scenario_params = f"top_{config.get('top_k', 5)}"
        case "A2_bottom":
            target_anomalies = filter_bottom_k_frequent(features_dict, k=config.get('bottom_k', 5))
            scenario_params = f"bottom_{config.get('bottom_k', 5)}"

        # --- B SCENARIOS ---
        case "B1_exact":
            target_anomalies = filter_by_ged(features_dict, exact_ged=config['ged_thresholds']['exact'])
            scenario_params = f"exact_ged_{config['ged_thresholds']['exact']}"
        case "B1_extreme_min":
            target_anomalies = filter_by_ged(features_dict, min_ged=config['ged_thresholds']['min_extreme'])
            scenario_params = f"min_extreme_{config['ged_thresholds']['min_extreme']}"
        case "B1_extreme_max":
            target_anomalies = filter_by_ged(features_dict, max_ged=config['ged_thresholds']['max_extreme'])
            scenario_params = f"max_extreme_{config['ged_thresholds']['max_extreme']}"
        case "B2_bottleneck":
            target_anomalies = filter_by_bottleneck(features_dict, anom_graphs, auto_bottlenecks)
            scenario_params = f"bottleneck_top_{config.get('top_k_bottlenecks', 3)}"
        case "B3_early":
            target_anomalies = filter_by_position(features_dict, anom_graphs, auto_early, "Early")
            scenario_params = f"early_{config.get('top_k', 5)}"
        case "B3_late":
            target_anomalies = filter_by_position(features_dict, anom_graphs, auto_late, "Late")
            scenario_params = f"late_{config.get('top_k', 5)}"
            
        # --- C SCENARIOS ---
        
        # --- D SCENARIOS ---   
            
        # --- DEFAULT ---
        case _:
            print(f"[ERROR] Scenario '{args.scenario}' not implemented or invalid.")
            sys.exit(1)
        
    if not target_anomalies:
        print("[WARNING] The chosen scenario resulted in 0 target anomalies. Exiting.")
        sys.exit(0)
        
    # ----------------------------------
    # --- PHASE 2: Alteration Engine ---
    # ----------------------------------
    if args.strategy == "repair":
        altered_log, modified_traces = run_repair(original_log, anom_graphs, corr_graphs, features_dict, target_anomalies)
    elif args.strategy == "infect":
        print("[INFO] Strategy 'infect' not implemented yet.")
        sys.exit(0)
        
    # -----------------------
    # --- PHASE 4: Saving ---
    # -----------------------
    pm4py.write_xes(altered_log, str(output_path))
    if args.recalc_baseline or not is_baseline_calculated(matrix_path, dataset_name):
        print("[INFO] Calculating baseline metrics...")
        baseline_metrics = evaluate_model(log_path, pnml_path)
        update_results_matrix(matrix_path, dataset_name, "BASELINE", "original", 0, baseline_metrics)
        
    print(f"\n[!] Evaluating new {args.strategy.upper()} log for scenario {args.scenario}...")
    new_metrics = evaluate_model(output_path, pnml_path)
    update_results_matrix(matrix_path, dataset_name, args.strategy, args.scenario, modified_traces, new_metrics, parameters=scenario_params)
    print("\nExperiment completed successfully!")
     
if __name__ == "__main__":
    main()
    
    '''
    Example from terminal:
    python main.py --dataset fineExp --strategy repair --scenario A1
    '''