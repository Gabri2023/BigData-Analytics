import pandas as pd

def frequence_extraction(csv_file):
    # Caricamento del dataset
    df = pd.read_csv(csv_file, sep=';')

    # Escludiamo la colonna 'grafo' per sommare solo le occorrenze dei sottografi
    subgraphs_only = df.drop(columns=['grafo'])

    # Sommiamo e filtriamo solo quelli con almeno 1 occorrenza
    occurrences = subgraphs_only.sum()
    occurrences = occurrences[occurrences > 0]

    # Creiamo il dizionario ufficiale delle frequenze
    freq_dict = occurrences.to_dict()

    print(f"Numero di sottografi anomali trovati nel CSV: {len(freq_dict)}")
    print(f"Lista degli ID estratti dal CSV:\n{list(freq_dict.keys())}")
    
    return freq_dict