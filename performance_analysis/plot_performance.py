import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Prendo il tempo di esecuzione del programma sequenziale
df_base = pd.read_csv('results_sequential.csv')
T_seq = df_base['T_Compute (s)'].values[1]

# Lista dei file da ciclare
files_csv = ['mpi_native/results_MPI_infiniband.csv']

for file in files_csv:
    if not os.path.exists(file):
        print(f"[INFO] Salto {file} perché non ancora creato.")
        continue

    # Carica i dati ed esegue i calcoli
    df = pd.read_csv(file)

    # Assicurati che le colonne 'Speedup' ed 'Efficienza' esistano
    # (se non presenti, calcolale; qui assumiamo già presenti)
    # df['Speedup'] = T_seq / df['T_Compute (s)']   # se necessario
    df['Efficienza'] = df['Speedup'] / df['Processori']

    # Ordina per numero di processori
    df = df.sort_values(by='Processori')
    X_procs = df['Processori'].values

    versione_nome = file.replace('risultati_', '').replace('.csv', '').upper()

    # -----------------------------------------------------------------
    # GRAFICO 1: SPEEDUP
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # Linea ideale (Speedup = Numero di Processori)
    plt.plot(X_procs, X_procs,
             label='Speedup Ideale',
             color='gray', linestyle='--', marker='o', markersize=6)

    # Linea reale
    plt.plot(X_procs, df['Speedup'].values,
             label='Speedup Reale',
             color='blue', linestyle='-', marker='s', linewidth=2, markersize=6)

    plt.title('Analisi Speedup')
    plt.xlabel('Processors')
    plt.ylabel('Speedup')

    # Imposta i tick dell'asse X come nell'immagine (1,4,7,...,64)
    xticks = np.arange(1, 65, 3)
    plt.xticks(xticks)
    # Eventualmente limita l'asse X al range dei dati
    plt.xlim(min(X_procs), max(X_procs))

    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Salvataggio (opzionale)
    # plt.savefig(f'grafico_{versione_nome.lower()}_speedup.pdf', bbox_inches='tight')
    # plt.savefig(f'grafico_{versione_nome.lower()}_speedup.png', bbox_inches='tight')
    # plt.close()

    # -----------------------------------------------------------------
    # GRAFICO 2: EFFICIENZA
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # Linea ideale (Efficienza = 1)
    plt.axhline(y=1, label='Efficienza Ideale', color='gray', linestyle='--')

    # Linea reale
    plt.plot(X_procs, df['Efficienza'].values,
             label='Efficienza Reale',
             color='red', linestyle='-', marker='^', linewidth=2, markersize=6)

    plt.title('Analisi Efficienza')
    plt.xlabel('Processors')
    plt.ylabel('Efficienza')
    plt.ylim(0, 2.0)   # margine superiore per leggibilità

    # Stessi tick sull'asse X
    plt.xticks(xticks)
    plt.xlim(min(X_procs), max(X_procs))

    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # Salvataggio (opzionale)
    # plt.savefig(f'grafico_{versione_nome.lower()}_efficienza.pdf', bbox_inches='tight')
    # plt.savefig(f'grafico_{versione_nome.lower()}_efficienza.png', bbox_inches='tight')
    # plt.close()

print("[OK] Grafici generati e visualizzati correttamente!")