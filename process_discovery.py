import os
import pm4py

# ======== 1️⃣ Input ========
#log_path = "Sepsis Cases - Event Log_log_con_start_end.xes"
log_path ="SEPSIS_Final_Complete.xes"
output_folder = "Modelli_datasets_raw"
os.makedirs(output_folder, exist_ok=True)

# ======== 2️⃣ Caricamento log ========
print("Parsing log...")
# Nota: assicurati che il file .xes esista in questa cartella
log = pm4py.read_xes(log_path)
print(f"Log caricato: {len(log)} trace")

# ======== 3️⃣ Scoperta ProcessTree ========
print("Scoperta ProcessTree con Inductive Miner...")
# Nuova sintassi semplificata per scoprire il Process Tree
ptree = pm4py.discover_process_tree_inductive(log, noise_threshold=0.2)

# ======== 4️⃣ Conversione ProcessTree -> Petri Net ========
print("Conversione in Petri Net...")


# La funzione è ora accessibile direttamente da pm4py.
net, initial_marking, final_marking = pm4py.convert_to_petri_net(ptree)
print("Petri net generata.")

# ======== 5️⃣ Esportazione PNML ========
pnml_path = os.path.join(output_folder, "petri_net_SEPSIS_finale2.pnml")
# La sintassi di write_pnml è leggermente cambiata per essere più diretta
pm4py.write_pnml(net, initial_marking, final_marking, pnml_path)
print(f"PNML salvato in: {pnml_path}")

# ======== 6️⃣ Esportazione PNG ========
png_path = os.path.join(output_folder, "petri_net_SEPSIS_finale2.png")
print("Salvataggio immagine PNG...")
# pm4py ora ha una funzione diretta per salvare la visualizzazione
pm4py.save_vis_petri_net(net, initial_marking, final_marking, png_path)
print(f"Petri net salvato in: {png_path}")