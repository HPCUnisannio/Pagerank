import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# 1. CONFIGURAZIONE
# ------------------------------------------------------------
# Scegli il criterio per selezionare la riga migliore per ogni numero di processori:
# - 'speedup'   : massimo speedup
# - 'efficienza': massima efficienza
CRITERIO = 'speedup'   # <--- CAMBIA QUI

# Lista dei file CSV da confrontare
files = [
    ('mpi_native/results_MPI_infiniband.csv', 'MPI nativa'),
    ('csr/mpi/results_MPI_CSR_infiniband.csv', 'MPI CSR'),
    ('csr/hybrid/results_hybrid_CSR_infiniband.csv', 'MPI + OpenMP ibrido')
]

# Carica il tempo sequenziale di riferimento
df_seq = pd.read_csv('results_sequential.csv')
T_seq = df_seq['T_Compute (s)'].values[1]   # primo valore
print(f"Tempo sequenziale di riferimento: {T_seq} s\n")

# ------------------------------------------------------------
# 2. FUNZIONE PER CARICARE E PREPARARE I DATI DI UN FILE
# ------------------------------------------------------------
def load_and_prepare(filepath, label, criterion='speedup'):
    """
    Carica un file CSV, calcola le metriche e seleziona per ogni Processori
    la riga migliore secondo il criterio scelto.

    Parametri:
        filepath  : percorso del CSV
        label     : etichetta per la legenda
        criterion : 'speedup', 'efficienza' o 'tempi'
    """
    if not os.path.exists(filepath):
        print(f"[INFO] Salto {filepath} perché non trovato.")
        return None

    df = pd.read_csv(filepath)

    df['Efficienza_float'] = df['Efficienza'].str.rstrip('%').astype(float) / 100

    # --- Seleziona la riga migliore per ogni Processori ---
    def get_best_indices(group):
        if criterion == 'speedup':
            return group['Speedup'].idxmax()
        elif criterion == 'efficienza':
            return group['Efficienza_float'].idxmax()
        else:
            raise ValueError("Criterio non valido: scegli 'speedup', 'efficienza' o 'tempi'")

    # Raggruppa per Processori e trova l'indice della riga migliore
    idx_best = df.groupby('Processori').apply(get_best_indices)
    # Estrai le righe selezionate
    df_best = df.loc[idx_best].reset_index(drop=True)

    # Ordina per Processori
    df_best = df_best.sort_values('Processori')

    # Restituisci un dizionario con i dati
    return {
        'label': label,
        'procs': df_best['Processori'].values,
        'speedup': df_best['Speedup'].values,
        'efficienza': df_best['Efficienza_float'].values,
    }

# ------------------------------------------------------------
# 3. CARICAMENTO DATI PER TUTTI I FILE
# ------------------------------------------------------------
data_list = []
for filepath, label in files:
    data = load_and_prepare(filepath, label, criterion=CRITERIO)
    if data is not None:
        data_list.append(data)

if not data_list:
    print("Nessun file valido trovato. Esco.")
    exit()

# ------------------------------------------------------------
# 4. GRAFICO SPEEDUP (CONFRONTO MULTIPLE VERSIONI)
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Colori e stili
colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
markers = ['s', '^', 'o', 'D', 'v']

for i, data in enumerate(data_list):
    plt.plot(data['procs'], data['speedup'],
             color=colors[i], linestyle='-', linewidth=2,
             marker=markers[i % len(markers)], markersize=6,
             label=data['label'])

max_procs = max(data['procs'].max() for data in data_list)
plt.plot([1, max_procs], [1, max_procs],
         color='k', linestyle='--', linewidth=2,
         label='Speedup Ideale')

#xticks = np.arange(1, 65, 3)
#plt.xticks(xticks)
#plt.xlim(0, max_procs)

plt.title('Confronto Speedup tra versioni', fontsize=14)
plt.xlabel('Processors', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
#plt.savefig('speedup_confronto.png', dpi=300, bbox_inches='tight')
#plt.savefig('speedup_confronto.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 5. GRAFICO EFFICIENZA (CONFRONTO MULTIPLE VERSIONI)
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

for i, data in enumerate(data_list):
    plt.plot(data['procs'], data['efficienza'],
             color=colors[i], linestyle='-', linewidth=2,
             marker=markers[i % len(markers)], markersize=6,
             label=data['label'])

plt.axhline(y=1.0, color='k', linestyle='--', linewidth=2,
            label='Efficienza Ideale')

#plt.xticks(xticks)
#plt.xlim(0, max_procs)

#yticks = np.arange(0.0, 2.05, 0.25)
#plt.yticks(yticks)
#plt.ylim(0, 2.0)

plt.title('Confronto Efficienza tra versioni', fontsize=14)
plt.xlabel('Processors', fontsize=12)
plt.ylabel('Efficienza', fontsize=12)
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
#plt.savefig('efficienza_confronto.png', dpi=300, bbox_inches='tight')
#plt.savefig('efficienza_confronto.pdf', bbox_inches='tight')
plt.show()


print("[OK] Grafici di confronto generati con criterio =", CRITERIO)