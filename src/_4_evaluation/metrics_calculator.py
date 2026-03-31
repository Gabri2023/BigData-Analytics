import sys
from pathlib import Path
from typing import Dict, Union

# pm4py imports
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


def evaluate_model(xes_path: Union[str, Path], pnml_path: Union[str, Path]) -> Dict[str, float]:
    """ 
    Calculates the metrics of Fitness, Precision, Generalization, and Simplicity.
    
    To call it from other scripts, just import this function and pass the paths 
    of the log and the model:

    from src.phase4_evaluation.metrics_calculator import evaluate_model
    res = evaluate_model("data/dataset_1/processed/log.xes", "data/dataset_1/raw/model.pnml")
    """
    # Convert paths to string in case pathlib.Path objects are passed
    xes_path_str = str(xes_path)
    pnml_path_str = str(pnml_path)

    # =========================
    # Data Loading
    # =========================
    log = xes_importer.apply(xes_path_str)
    net, im, fm = pnml_importer.apply(pnml_path_str)

    # =========================
    # Fitness (Alignment-based)
    # =========================
    fitness = fitness_evaluator.apply(
        log, net, im, fm,
        variant=fitness_evaluator.Variants.ALIGNMENT_BASED
    )

    # =========================
    # Precision (Alignment-based)
    # =========================
    precision = precision_evaluator.apply(
        log, net, im, fm,
        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE
    )

    # =========================
    # Generalization
    # =========================
    generalization = generalization_evaluator.apply(log, net, im, fm)

    # =========================
    # Simplicity
    # =========================
    simplicity = simplicity_evaluator.apply(net)

    # =========================
    # RETURN RESULTS
    # =========================
    # Using .get() ensures backward/forward compatibility with different pm4py versions
    fit_value = fitness.get("averageFitness", fitness.get("log_fitness", 0.0))

    return {
        "fitness": float(fit_value),
        "precision": float(precision),
        "generalization": float(generalization),
        "simplicity": float(simplicity)
    }

if __name__ == "__main__":
    # Standalone execution via CLI
    if len(sys.argv) != 3:
        print("Usage: python metrics_calculator.py <log.xes> <model.pnml>")
        sys.exit(1)

    cli_xes_path = sys.argv[1]
    cli_pnml_path = sys.argv[2]

    results = evaluate_model(cli_xes_path, cli_pnml_path)

    print("\n=== EVALUATION RESULTS ===")
    print(f"Fitness:        {results['fitness']:.4f}")
    print(f"Precision:      {results['precision']:.4f}")
    print(f"Generalization: {results['generalization']:.4f}")
    print(f"Simplicity:     {results['simplicity']:.4f}")
    print("==========================\n")