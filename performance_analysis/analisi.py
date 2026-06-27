import pandas as pd

# 1. Configurazione visualizzazione PyCharm
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# 2. PRENDIAMO IL RIFERIMENTO SEQUENZIALE
# Prendo il tempo di esecuzione del programma sequenziale
df_base = pd.read_csv('results_sequential.csv')
T_seq = df_base['T_Compute (s)'].values[1]
print(f"=== TEMPO SEQUENZIALE DI RIFERIMENTO: {T_seq} s ===\n")


# 3. ELABORAZIONE SINGOLA DEI DATASET
# Lista dei datasets CSV
datasets = ['mpi_native/results_MPI_infiniband.csv']

for file in datasets:
    # Carica il singolo dataset
    df = pd.read_csv(file)

    # Calcolo Speedup ed Efficienza per il dataset in questione: uso tempo prettamente di calcolo
    df['Speedup'] = (T_seq / df['T_Compute (s)']).round(4)
    df['Efficienza'] = ((df['Speedup'] / df['Processori']) * 100).round(1).astype(str) + '%'

    # Stampa i risultati di questo specifico dataset
    print(f"---------------------------------------------------------------------------")
    print(f" ANALISI DATASET: {file}")
    print(f"---------------------------------------------------------------------------")
    print(df.to_string(index=False))
    print(f"---------------------------------------------------------------------------\n")

    # Per salvare i risultati in un file CSV:
    df.to_csv(file, index=False)