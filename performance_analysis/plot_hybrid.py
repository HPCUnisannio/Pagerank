import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. CONFIGURAZIONE
# ------------------------------------------------------------
# Scegli il criterio per selezionare la riga migliore per ogni numero di processori:
# - 'speedup'   : massimo speedup
# - 'efficienza': massima efficienza
CRITERIO = 'speedup'   # <--- CAMBIA QUI per cambiare il criterio

# ------------------------------------------------------------
# 2. LETTURA DATI
# ------------------------------------------------------------
# Carica il dataset ibrido
df = pd.read_csv('csr/hybrid/results_hybrid_CSR_infiniband.csv')

# Carica il tempo sequenziale di riferimento
df_seq = pd.read_csv('results_sequential.csv')
T_seq = df_seq['T_Compute (s)'].values[1]   # primo valore
print(f"Tempo sequenziale di riferimento: {T_seq} s")

# ------------------------------------------------------------
# 3. PULIZIA DATI
# ------------------------------------------------------------
# Se Efficienza è stringa con '%', converti in float decimale
df['Efficienza_float'] = df['Efficienza'].str.rstrip('%').astype(float) / 100

# Crea una colonna 'Config' per etichettare le configurazioni
df['Config'] = df.apply(
    lambda row: f"{int(row['Nodi'])} nodi, {int(row['Processi'])} MPI, {int(row['Thread'])} OMP",
    axis=1
)

# Ordina per Processori (core fisici totali)
df = df.sort_values('Processori')
max_procs = df['Processori'].max()



# Colori per gli scatter
configs = df['Config'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

# ------------------------------------------------------------
# 4. FUNZIONE PER SELEZIONARE LA RIGA MIGLIORE PER OGNI PROCESSORI
# ------------------------------------------------------------
def get_best_rows(df, criterion='speedup'):
    """
    Per ogni Processori, seleziona la riga che ottimizza il criterio scelto.
    Restituisce un DataFrame con le righe selezionate.
    """
    def get_best_indices(group):
        if criterion == 'speedup':
            return group['Speedup'].idxmax()
        elif criterion == 'efficienza':
            return group['Efficienza_float'].idxmax()
        else:
            raise ValueError("Criterio non valido: scegli 'speedup', 'efficienza' o 'tempi'")

    idx_best = df.groupby('Processori').apply(get_best_indices)
    df_best = df.loc[idx_best].reset_index(drop=True)
    return df_best.sort_values('Processori')

# Seleziona le righe migliori secondo il criterio scelto
df_best = get_best_rows(df, CRITERIO)


# Raggruppa per Processori per gli intervalli (min/max indipendenti dal criterio)
grouped = df.groupby('Processori')
procs_unique = df['Processori'].unique()

# ------------------------------------------------------------
# 5. GRAFICI SPEEDUP
# ------------------------------------------------------------
# 5a. Scatter (tutte le configurazioni)
plt.figure(figsize=(12, 6))
for i, config in enumerate(configs):
    subset = df[df['Config'] == config]
    plt.scatter(subset['Processori'], subset['Speedup'],
                label=config, color=colors[i], s=80, alpha=0.7)
plt.plot([1, max_procs], [1, max_procs],
         'k--', label='Speedup ideale', linewidth=2)
plt.title('Speedup per diverse configurazioni ibride MPI+OpenMP')
plt.xlabel('Processori (core fisici totali)')
plt.ylabel('Speedup')
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.savefig('speedup_scatter.png', dpi=300, bbox_inches='tight')
#plt.savefig('speedup_scatter.pdf', bbox_inches='tight')
plt.show()

# 5b. Best case + intervallo (secondo il criterio scelto)
plt.figure(figsize=(12, 6))

# Linea principale: la migliore secondo il criterio
plt.plot(df_best['Processori'], df_best['Speedup'],
         'bo-', label=f'Speedup ',
         linewidth=2, markersize=8)

# Intervallo di variazione (min/max di tutte le configurazioni)
#min_speedup = grouped['Speedup'].min().values
#max_speedup = grouped['Speedup'].max().values
#plt.fill_between(procs_unique, min_speedup, max_speedup,
 #                color='blue', alpha=0.2, label='Intervallo di variazione')

# Linea ideale
plt.plot([1, max_procs], [1, max_procs],
         'k--', label='Speedup ideale', linewidth=2)

plt.title(f'Analisi Speedup MPI + OpenMP')
plt.xlabel('Processori (core fisici totali)')
plt.ylabel('Speedup')
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.legend()
plt.tight_layout()
#plt.savefig('speedup_bestcase.png', dpi=300, bbox_inches='tight')
#plt.savefig('speedup_bestcase.pdf', bbox_inches='tight')
plt.show()


# ------------------------------------------------------------
# 6. GRAFICI EFFICIENZA
# ------------------------------------------------------------
# 6a. Scatter (tutte le configurazioni)
plt.figure(figsize=(12, 6))
for i, config in enumerate(configs):
    subset = df[df['Config'] == config]
    plt.scatter(subset['Processori'], subset['Efficienza_float'],
                label=config, color=colors[i], s=80, alpha=0.7)
plt.axhline(y=1.0, color='k', linestyle='--', label='Efficienza ideale', linewidth=2)
plt.title('Efficienza per diverse configurazioni ibride')
plt.xlabel('Processori (core fisici totali)')
plt.ylabel('Efficienza')
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.savefig('efficienza_scatter.png', dpi=300, bbox_inches='tight')
#plt.savefig('efficienza_scatter.pdf', bbox_inches='tight')
plt.show()

# 6b. Best case + intervallo (secondo il criterio scelto)
plt.figure(figsize=(12, 6))

# Linea principale: la migliore secondo il criterio
plt.plot(df_best['Processori'], df_best['Efficienza_float'],
         'ro-', label=f'Efficienza',
         linewidth=2, markersize=8)

# Intervallo di variazione
#min_eff = grouped['Efficienza_float'].min().values
#max_eff = grouped['Efficienza_float'].max().values
#plt.fill_between(procs_unique, min_eff, max_eff,
 #                color='red', alpha=0.2, label='Intervallo di variazione')

# Linea ideale
plt.axhline(y=1.0, color='k', linestyle='--', label='Efficienza ideale', linewidth=2)

plt.title(f'Analisi Efficienza MPI + OpenMP')
plt.xlabel('Processori (core fisici totali)')
plt.ylabel('Efficienza')
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.legend()
plt.tight_layout()
#plt.savefig('efficienza_bestcase.png', dpi=300, bbox_inches='tight')
#plt.savefig('efficienza_bestcase.pdf', bbox_inches='tight')
plt.show()



print(f"[OK] Tutti i grafici generati con criterio = '{CRITERIO}'.")