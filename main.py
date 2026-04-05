import argparse
from pathlib import Path
import yaml
import pm4py
import sys
import os
import pickle  # Added for Data Caching

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Imports from PHASE 1 ---
from src._1_baseline.bottleneck_extractor import extract_process_metrics
from src._1_baseline.frequencies_extractor import extract_frequencies
from src._1_baseline.ged_mapper import get_features
from src._1_baseline.parser import parse_subelements

# --- Imports from PHASE 2 ---
from src._2_engine.repair import run_repair

# --- Imports from PHASE 3 ---
from src._3_scenarios.a_global_frequency import filter_all_anomalies, filter_top_k_frequent, filter_bottom_k_frequent
from src._3_scenarios.b_structural import filter_by_ged, filter_by_bottleneck, filter_by_position

# --- Imports from PHASE 4 ---
from src._4_evaluation.metrics_calculator import evaluate_model
from src._4_evaluation.results_tracker import update_results_matrix, is_baseline_calculated

def main():
    parser = argparse.ArgumentParser(description="Log Alteration Engine - Process Mining")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset name (e.g., 'fineExp').")
    parser.add_argument("--strategy", type=str, required=True, choices=["infect", "repair"],
                        help="Choose whether to inject anomalies ('infect') or fix them ('repair').")
    parser.add_argument("--scenario", type=str, nargs='+', required=True,
                        help="Scenario code(s) to execute (e.g., A1, A2_top B2_bottleneck).")
    parser.add_argument("--recalc-baseline", action="store_true",
                        help="If set, recalculates baseline metrics before execution.")
    parser.add_argument("--config", type=str, 
                        help="Optional path to a custom YAML configuration file.")
    
    args = parser.parse_args()

    dataset_name = args.dataset
    base_data_path = Path("data") / dataset_name
    
    log_path = base_data_path / f"{dataset_name}.xes"
    csv_path = base_data_path / f"{dataset_name}_table2_on_file.csv"
    anom_path = base_data_path / "custom" / "anomalous_sub.txt"
    corr_path = base_data_path / "custom" / "correct_sub.txt"
    pnml_path = base_data_path / "models_raw" / f"petri_net_{dataset_name}.pnml"
    matrix_path = Path("results") / "experiments_matrix.csv"
    config_path = Path(args.config) if args.config else Path("config") / f"config_{dataset_name}.yaml"
        
    for path, name in [(log_path, "Log"), (csv_path, "CSV"), (anom_path, "Anomalous TXT"),
                       (corr_path, "Correct TXT"), (config_path, "Config YAML"), (pnml_path, "PNML")]:
        if not path.exists():
            print(f"[ERROR] {name} file not found at {path}")
            sys.exit(1)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    for key in ['top_k', 'bottom_k', 'top_k_bottlenecks', 'exact_ged', 'min_extreme_ged', 'max_extreme_ged']:
        if isinstance(config.get(key), list):
            config[key] = config[key][0]
        
    # -----------------------------------
    # --- PHASE 1: Loading & Caching  ---
    # -----------------------------------
    original_log = pm4py.read_xes(str(log_path), return_legacy_log_object=True)
    freq_dict = extract_frequencies(csv_path)
    anomaly_ids = list(freq_dict.keys())
    anom_graphs = parse_subelements(anom_path, custom_ids=anomaly_ids)
    corr_graphs = parse_subelements(corr_path)

    # 1. CACHING: Prevent GED recalculation on every grid search iteration
    cache_path = base_data_path / "custom" / "features_cache.pkl"
    if cache_path.exists():
        print("[INFO] Loading pre-calculated GED features from cache...")
        with open(cache_path, 'rb') as f:
            features_dict = pickle.load(f)
    else:
        print("[INFO] First run detected: calculating and caching GED features...")
        features_dict = get_features(anom_graphs, corr_graphs, anomaly_ids, freq_dict)
        with open(cache_path, 'wb') as f:
            pickle.dump(features_dict, f)

    # 2. LAZY EVALUATION: Extract bottleneck metrics ONLY if requested by the scenario
    needs_metrics = any(scen in ["B2_bottleneck", "B3_early", "B3_late"] for scen in args.scenario)
    if needs_metrics:
        auto_bottlenecks, auto_early, auto_late = extract_process_metrics(log_path, top_k=config.get('top_k_bottlenecks', 3))

    # -----------------------------------
    # --- PHASE 3: Scenario Selection ---
    # -----------------------------------
    scenario_params_list = []
    combined_anomalies = None
    
    for current_scenario in args.scenario:
        current_anomalies = []
        match current_scenario:
            case "A1":
                current_anomalies = filter_all_anomalies(features_dict)
                scenario_params_list.append("all")
            case "A2_top":
                current_anomalies = filter_top_k_frequent(features_dict, k=config.get('top_k', 5))
                scenario_params_list.append(f"top_{config.get('top_k', 5)}")
            case "A2_bottom":
                current_anomalies = filter_bottom_k_frequent(features_dict, k=config.get('bottom_k', 5))
                scenario_params_list.append(f"bottom_{config.get('bottom_k', 5)}")
            case "B1_exact":
                current_anomalies = filter_by_ged(features_dict, exact_ged=config.get('exact_ged', 2))
                scenario_params_list.append(f"exact_ged_{config.get('exact_ged', 2)}")
            case "B1_extreme_min":
                current_anomalies = filter_by_ged(features_dict, min_ged=config.get('min_extreme_ged', 4))
                scenario_params_list.append(f"min_extreme_{config.get('min_extreme_ged', 4)}")
            case "B1_extreme_max":
                current_anomalies = filter_by_ged(features_dict, max_ged=config.get('max_extreme_ged', 6))
                scenario_params_list.append(f"max_extreme_{config.get('max_extreme_ged', 6)}")
            case "B2_bottleneck":
                current_anomalies = filter_by_bottleneck(features_dict, anom_graphs, auto_bottlenecks)
                scenario_params_list.append(f"bottleneck_top_{config.get('top_k_bottlenecks', 3)}")
            case "B3_early":
                current_anomalies = filter_by_position(features_dict, anom_graphs, auto_early, "Early")
                scenario_params_list.append(f"early")
            case "B3_late":
                current_anomalies = filter_by_position(features_dict, anom_graphs, auto_late, "Late")
                scenario_params_list.append(f"late")
            case _:
                print(f"[ERROR] Scenario '{current_scenario}' not implemented or invalid.")
                sys.exit(1)
                
        if combined_anomalies is None:
            combined_anomalies = set(current_anomalies)
        else:
            combined_anomalies = combined_anomalies.intersection(set(current_anomalies))

    target_anomalies = list(combined_anomalies)
    final_scenario_name = "+".join(args.scenario)
    final_params_string = "+".join(scenario_params_list)

    if not target_anomalies:
        print(f"[WARNING] Combined scenario '{final_scenario_name}' resulted in 0 anomalies. Exiting.")
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
    # 3. DYNAMIC OUTPUT PATH: Prevent overwriting by including the specific parameters
    output_path = base_data_path / "custom" / "processed" / f"{dataset_name}_{args.strategy}_{final_scenario_name}_{final_params_string}.xes"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(altered_log, str(output_path))
    
    if args.recalc_baseline or not is_baseline_calculated(matrix_path, dataset_name):
        print("[INFO] Calculating baseline metrics...")
        baseline_metrics = evaluate_model(log_path, pnml_path)
        update_results_matrix(matrix_path, dataset_name, "BASELINE", "original", 0, baseline_metrics)
        
    print(f"\n[!] Evaluating new {args.strategy.upper()} log for scenario {final_scenario_name}...")
    new_metrics = evaluate_model(output_path, pnml_path)
    update_results_matrix(matrix_path, dataset_name, args.strategy, final_scenario_name, modified_traces, new_metrics, parameters=final_params_string)
    print("\nExperiment completed successfully!")
     
if __name__ == "__main__":
    main()