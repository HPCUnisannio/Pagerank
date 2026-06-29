from operator import index

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ------------------------------------------------------------
# 1. LETTURA DATI
# ------------------------------------------------------------
# Prendo il tempo di esecuzione del programma sequenziale
df_base = pd.read_csv('results_sequential.csv')
T_seq = df_base['T_Compute (s)'].values[1]
print(f"Tempo sequenziale di riferimento: {T_seq} s")

# Lista dei file da ciclare
files_csv = ['csr/mpi/results_MPI_CSR_infiniband.csv',
             'mpi_native/results_MPI_infiniband.csv']

names = ['MPI CSR', 'MPI nativa']

for i,file in enumerate(files_csv):
    if not os.path.exists(file):
        print(f"[INFO] Salto {file} perché non ancora creato.")
        continue

    # Carica i dati
    df = pd.read_csv(file)

    df['Efficienza_float'] = df['Efficienza'].str.rstrip('%').astype(float) / 100



    # Ordina per numero di processori
    df = df.sort_values(by='Processori')
    X_procs = df['Processori'].values
    max_procs = X_procs.max()

    # Nome della versione per i titoli
    versione_nome = file.replace('risultati_', '').replace('.csv', '').upper()

    # Tick sull'asse X (1,4,7,...,64)
    #xticks = np.arange(1, 65, 3)

    # ------------------------------------------------------------
    # GRAFICO 1: SPEEDUP (stile unificato)
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Linea ideale
    plt.plot(X_procs, X_procs,
             'k--', linewidth=2,
             label='Speedup Ideale')

    # Linea reale
    plt.plot(X_procs, df['Speedup'].values, 'ro-',
             color='blue', linestyle='-', linewidth=2,markersize=8,
             label='Speedup Reale')

    plt.title('Analisi Speedup '+names[i], fontsize=14)
    plt.xlabel('Processors', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)

    #plt.xticks(xticks)
    #plt.xlim(0, max_procs)

    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()

    # Salvataggio (opzionale)
    # plt.savefig(f'speedup_{versione_nome.lower()}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'speedup_{versione_nome.lower()}.pdf', bbox_inches='tight')

    plt.show()

    # ------------------------------------------------------------
    # GRAFICO 2: EFFICIENZA (stile unificato)
    # ------------------------------------------------------------
    # Raggruppa per Processori (anche se un solo valore, è coerente)
    plt.figure(figsize=(12, 6))

    # Linea reale (miglior caso)
    plt.plot(X_procs, df['Efficienza_float'].values,
             'ro-', label='Efficienza Reale', linewidth=2, markersize=8)


    # Linea ideale
    plt.axhline(y=1.0, color='k', linestyle='--', label='Efficienza ideale', linewidth=2)

    plt.title('Analisi Efficienza '+names[i] , fontsize=14)
    plt.xlabel('Processors', fontsize=12)
    plt.ylabel('Efficienza', fontsize=12)

    # Tick asse X
    #xticks = np.arange(1, 65, 3)
    #plt.xticks(xticks)
   # plt.xlim(0, max_procs+1)

    # Tick asse Y (0.00, 0.25, ..., 2.00)
    #yticks = np.arange(0.0, 2.05, 0.25)
    #plt.yticks(yticks)
   # plt.ylim(0, 2.0)

    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()

    # Salvataggio (opzionale)
    # plt.savefig(f'efficienza_{versione_nome.lower()}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'efficienza_{versione_nome.lower()}.pdf', bbox_inches='tight')

    plt.show()



print("[OK] Grafici generati e visualizzati correttamente!")