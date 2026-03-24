""" 
function to calculate the metrics of fitness, precision, generalization and simplicity
to call it from other scripts, just import this function and pass it the paths of the log and the model

    from src.4_evaluation.metrics_calculator import evaluate_model
    res = evaluate_model("log.xes", "model.pnml") ## paths in quotes

"""


import sys
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


def evaluate_model(xes_path, pnml_path):
    # =========================
    # Caricamento dati
    # =========================
    log = xes_importer.apply(xes_path)
    net, im, fm = pnml_importer.apply(pnml_path)

    # =========================
    # Fitness
    # =========================
    fitness = fitness_evaluator.apply(
        log, net, im, fm,
        variant=fitness_evaluator.Variants.ALIGNMENT_BASED
    )

    # =========================
    # Precision
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
    # RETURN (questa è la parte chiave)
    # =========================
    return {
        "fitness": fitness["averageFitness"],
        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python script.py log.xes model.pnml")
        sys.exit(1)

    xes_path = sys.argv[1]
    pnml_path = sys.argv[2]

    results = evaluate_model(xes_path, pnml_path)

    print("\n=== RISULTATI ===")
    print(f"Fitness: {results['fitness']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Generalization: {results['generalization']:.4f}")
    print(f"Simplicity: {results['simplicity']:.4f}")