import parse_subelements
import frequences_extraction 
import get_features
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import re

# 0. Carichiamo i file di testo e i csv
csv_file = '../data/fineExp/fineExp_table2_on_file.csv' 
subelements_txt = '../data/fineExp/subelements_fineExp.txt'
anomalous_txt_data = open('../data/fineExp/custom/anomalous_sub.txt', 'r').read()
correct_txt_data = open('../data/fineExp/custom/correct_sub.txt', 'r').read()

# Estrazione frequenze
freq_dict = frequences_extraction.frequence_extraction(csv_file)

# Estraiamo i 27 ID anomali del file (escludendo i 5 finali che aggiungeremo a mano)
tutti_id = list(freq_dict.keys())
anomalous_ids_ordered = tutti_id[:27] 

# Parsiamo i grafi anomali e corretti dal file di testo
anomalous_subgraphs = parse_subelements.parse_subelements(anomalous_txt_data, custom_ids=anomalous_ids_ordered)
correct_subgraphs = parse_subelements.parse_subelements(correct_txt_data)

# Funzione per l'aggiunta manuale dei 5 restanti
def add_manual_sub(sub_dict, sub_id, nodes_dict, edges_list):
    G = nx.DiGraph()
    for nid, lbl in nodes_dict.items():
        G.add_node(nid, label=lbl)
    G.add_edges_from(edges_list)
    sub_dict[sub_id] = G

# Aggiungiamo i 5 grafi mancanti per arrivare a 32
add_manual_sub(anomalous_subgraphs, "Sub174", {1: "AddPenalty", 2: "NotifyOffenders", 3: "ReceiveResults"}, [(1,2), (2,3)])
add_manual_sub(anomalous_subgraphs, "Sub179", {1: "AddPenalty", 2: "NotifyOffenders", 3: "ReceiveResults"}, [(1,2), (2,3)])
add_manual_sub(anomalous_subgraphs, "Sub176", {1: "AddPenalty", 2: "AppealToPrefecture", 3: "AppealToJudge", 4: "SendAppeal"}, [(1,2), (2,3), (3,4)])
add_manual_sub(anomalous_subgraphs, "Sub178", {1: "AddPenalty", 2: "SendAppeal"}, [(1,2)])
add_manual_sub(anomalous_subgraphs, "Sub180", {1: "SendAppeal", 2: "AppealToJudge", 3: "AddPenalty"}, [(1,2), (2,3)])


print(f"Sottografi anomali in memoria: {len(anomalous_subgraphs)}")
print(f"Sottografi normativi in memoria: {len(correct_subgraphs)}")

target_subs = list(anomalous_subgraphs.keys()) # I nostri 32 ID

print(f"\nInizio calcolo feature per {len(target_subs)} sottografi anomali...")

features = get_features(anomalous_subgraphs, correct_subgraphs, target_subs, freq_dict)
