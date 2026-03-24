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
    # Output
    # =========================
    print("\n=== RISULTATI ===")
    print(f"Fitness: {fitness['averageFitness']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Generalization: {generalization:.4f}")
    print(f"Simplicity: {simplicity:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python script.py log.xes model.pnml")
        sys.exit(1)

    xes_path = sys.argv[1]
    pnml_path = sys.argv[2]

    evaluate_model(xes_path, pnml_path)