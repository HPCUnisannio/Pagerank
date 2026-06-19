# PageRank OpenMP Privatizzato: Scelte Progettuali

## Filosofia Progettuale: Architettura Zero-Atomic

L'implementazione adotta un approccio di privatizzazione della memoria che elimina completamente l'uso di direttive `#pragma omp atomic` durante la fase di moltiplicazione matrice-vettore, sostituendole con buffer privati per thread e una fase di riduzione esplicita.

## Struttura della Memoria Privatizzata

### Array `local_prnew`
La memoria privatizzata è implementata come array bidimensionale allocato dinamicamente: `local_prnew[num_threads][NODES]`. Ogni thread possiede un vettore privato di dimensione NODES:

- Thread 0: `local_prnew[0][0]` fino a `local_prnew[0][NODES-1]`
- Thread 1: `local_prnew[1][0]` fino a `local_prnew[1][NODES-1]`
- ...

Questa strategia rappresenta un Memory-Compute Tradeoff: si scambia RAM aggiuntiva (~33 MB per 6 thread su grafo medium) in cambio dell'eliminazione totale delle contese di scrittura.

## Direttive OpenMP Utilizzate

| Direttiva | Scopo |
|-----------|-------|
| `#pragma omp parallel` | Crea il team di thread per l'intera power iteration |
| `#pragma omp for` | Distribuisce le iterazioni del ciclo tra i thread (SpMV e riduzione) |
| `#pragma omp single nowait` | Un solo thread esegue l'operazione, gli altri non attendono |
| `#pragma omp single` | Un solo thread esegue l'operazione, gli altri attendono (barriera implicita) |
| `reduction(+:norm_sq)` | Ogni thread accumula la propria norma parziale, OpenMP somma automaticamente |

## Flusso di Esecuzione per Iterazione

### Fase 1: Azzeramento Privato
Ogni thread azzera il proprio array `local_prnew[tid]` senza necessità di sincronizzazione.

### Fase 2: Moltiplicazione Zero-Atomic (SpMV)
```c
#pragma omp for
for (col_local = 0; col_local < NODES; col_local++) {
    for (j_local = colptr[col_local]; j_local < colptr[col_local + 1]; j_local++) {
        local_prnew[tid][rowind[j_local]] += val[j_local] * prold[col_local];
    }
}
```
Ogni thread scrive esclusivamente nel proprio array privato. Nessuna direttiva atomic necessaria.

### Fase 3: Reset Norma
```c
#pragma omp single nowait
{
    norm_sq = 0.0;
}
```
Un solo thread azzera norm_sq. La clausola nowait evita una barriera implicita.

### Fase 4: Riduzione, Damping e Norma
```c
#pragma omp for reduction(+:norm_sq)
for (i_local = 0; i_local < NODES; i_local++) {
    double sum_local = 0.0;
    for (t_local = 0; t_local < num_threads; t_local++) {
        sum_local += local_prnew[t_local][i_local];
    }
    prnew[i_local] = sum_local * damp1[i_local] + damp2[i_local];
    diff[i_local] = prnew[i_local] - prold[i_local];
    norm_sq += diff[i_local] * diff[i_local];
    prold[i_local] = prnew[i_local];
}
```
I thread si dividono i nodi e leggono i contributi da tutti gli array privati. Il damping viene applicato dopo la somma completa. La clausola reduction(+:norm_sq) gestisce automaticamente l'accumulo thread-safe della norma.

### Fase 5: Aggiornamento Norma Finale
```c
#pragma omp single
{
    norm = sqrt(norm_sq);
}
```
Un solo thread calcola la radice quadrata. La barriera implicita garantisce che norm_sq sia completa.

## Il Fenomeno del "Memory Wall"

### Osservazione Empirica
Riducendo il numero di thread, le prestazioni migliorano. Con il massimo numero di core disponibili, il tempo di esecuzione subisce un degrado.

### Motivazione Tecnica
Rimuovendo le istruzioni atomic, si elimina la contesa sulla CPU ma si sposta il collo di bottiglia sul bus della RAM. L'algoritmo diventa memory-bound.

| Causa | Descrizione |
|-------|-------------|
| Overhead di Riduzione | La fusione richiede T letture per nodo. Con T=16 e 685k nodi: ~11 milioni di accessi RAM per iterazione |
| Saturazione del Bus | Troppi thread richiedono vettori enormi simultaneamente, creando un ingorgo sul bus della RAM |
| Cache Thrashing | La memoria locale di T thread eccede la capacità della Cache L3, forzando ricaricamenti continui dalla RAM |

### Sweet Spot
Il tempo ottimale si ottiene con un numero di thread inferiore ai core fisici (tipicamente 4), che bilancia la parallelizzazione senza saturare la larghezza di banda della RAM.

## Vantaggi e Svantaggi

### Vantaggi
- Zero contesa atomica durante la SpMV
- Massima velocità hardware, nessuno stallo
- Località della cache massimizzata nella fase di riduzione
- Codice relativamente semplice grazie alle direttive OpenMP

### Svantaggi
- Occupazione memoria proporzionale al numero di thread
- La fase di riduzione richiede T letture per ogni nodo
- Sensibilità al Memory Wall su molti core
- La memoria privata deve essere azzerata a ogni iterazione

## Comando di Compilazione ed Esecuzione
gcc -fopenmp -O2 -o openmp/pr_openmp_opt openmp/pr_openmp_opt.c libraries/data.c libraries/measure.c -Ilibraries -lm
./openmp/pr_openmp_opt 4

## Note sulle Prestazioni
- L'eliminazione delle direttive atomic rimuove la contesa sulla CPU
- Il collo di bottiglia si sposta sulla banda di memoria RAM
- La curva delle prestazioni segue un andamento a "U" rispetto al numero di thread
- Lo scheduling statico nativo di OpenMP è ideale per carichi bilanciati
- La clausola nowait su omp single evita barriere non necessarie
- La reduction(+:norm_sq) è più efficiente di accumuli manuali con atomic



