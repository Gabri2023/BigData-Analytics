import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter
from datetime import timedelta
import math
from copy import deepcopy

# ==========================================
# 1. FUNZIONE PER RIMUOVERE I NAN
# ==========================================
def clean_log_from_nans(log):
    print("Pulizia dei valori NaN e attributi sporchi...")
    for trace in log:
        for event in trace:
            keys_to_remove = []
            for key, value in event.items():
                if isinstance(value, float) and math.isnan(value):
                    keys_to_remove.append(key)
                elif isinstance(value, str) and value.lower() == 'nan':
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del event[key]
    return log

# ==========================================
# 2. FUNZIONE AGGIUNTA START/END (CORRETTA)
# ==========================================
def add_start_end_to_log(log):
    # Convertiamo per sicurezza
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    
    # Creiamo il nuovo log
    new_log = EventLog()

    # [CORREZIONE] STEP 0: COPIARE I METADATI GLOBALI
    # Non usiamo '=', ma .update() perché attributes è read-only
    print("Copia dei metadati del log (Extensions, Classifiers, Globals)...")
    
    if hasattr(log, "attributes"):
        new_log.attributes.update(deepcopy(log.attributes))
        
    if hasattr(log, "extensions"):
        # Anche le extensions vanno aggiornate, non sovrascritte
        new_log.extensions.update(deepcopy(log.extensions))
        
    if hasattr(log, "classifiers"):
        # I classificatori sono un dizionario, usiamo update
        new_log.classifiers.update(deepcopy(log.classifiers))
        
    if hasattr(log, "omni_present"):
        new_log.omni_present.update(deepcopy(log.omni_present))

    # Definiamo le chiavi da mantenere per Start/End
    KEYS_TO_KEEP = ["time:timestamp", "lifecycle:transition", "org:resource", "org:group"]

    for trace in log:
        new_trace = Trace()
        # Copiamo gli attributi della traccia
        if hasattr(trace, "attributes"):
            new_trace.attributes.update(trace.attributes)

        if len(trace) > 0:
            # ===== START =====
            first_event = trace[0]
            start_event = Event()
            start_event["concept:name"] = "Start"
            for key in KEYS_TO_KEEP:
                if key in first_event:
                    start_event[key] = first_event[key]
            # Gestione sicurezza timestamp
            if "time:timestamp" in first_event:
                start_event["time:timestamp"] = first_event["time:timestamp"] - timedelta(seconds=1)
            
            new_trace.append(start_event)

            # ===== EVENTI ORIGINALI =====
            for ev in trace:
                new_trace.append(ev)

            # ===== END =====
            last_event = trace[-1]
            end_event = Event()
            end_event["concept:name"] = "End"
            for key in KEYS_TO_KEEP:
                if key in last_event:
                    end_event[key] = last_event[key]
            if "time:timestamp" in last_event:
                end_event["time:timestamp"] = last_event["time:timestamp"] + timedelta(seconds=1)
                
            new_trace.append(end_event)
        
        else:
            # Tracce vuote vengono mantenute vuote
            pass

        new_log.append(new_trace)

    return new_log

# ==============================
# ESECUZIONE
# ==============================
# Assicurati che il nome del file sia corretto
filename = "Sepsis Cases - Event Log_new.xes"
output_name = "SEPSIS_Final_Complete.xes"

print(f"Caricamento {filename}...")
log = pm4py.read_xes(filename)

# 1. Aggiunta Start/End preservando i metadati
try:
    log = add_start_end_to_log(log)
except Exception as e:
    print(f"ERRORE durante l'aggiunta Start/End: {e}")
    exit()

# 2. Pulizia NaN
log = clean_log_from_nans(log)

# 3. Salvataggio
print(f"Salvataggio in {output_name}...")
pm4py.write_xes(log, output_name)

print("Fatto! Il file è stato generato correttamente con i metadati.")