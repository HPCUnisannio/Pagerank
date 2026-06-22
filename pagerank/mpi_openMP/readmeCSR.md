# Introduzione

Il codice implementa l'algoritmo PageRank in ambiente parallelo
utilizzando un approccio ibrido MPI+OpenMP. L'obiettivo è sfruttare sia
la parallelizzazione a livello di nodo (MPI) che quella a livello di
core (OpenMP) per massimizzare le prestazioni su cluster HPC.

# Funzionamento del Codice

## Struttura Dati

Il grafo è rappresentato in formato CSR (Compressed Sparse Row):

``` {style="codestyle"}
typedef struct {
    int N;          // numero di nodi
    int E;          // numero di archi
    int *row_ptr;   // puntatori di riga (size N+1)
    int *col_idx;   // indici di colonna (size E)
    double *val;    // valori (size E)
    int *outdeg;    // grado uscente (size N)
} CSRGraph;
```

Il formato CSR è efficiente per la memorizzazione di grafi sparsi e
consente un rapido accesso agli archi durante il calcolo del PageRank.

## Algoritmo PageRank

L'algoritmo implementa la formula iterativa:

$$PR(i)^{(k+1)} = d \cdot \sum_{j \in B_i} \frac{PR(j)^{(k)}}{L(j)} + \frac{1-d}{N}$$

Il processo iterativo continua fino a quando la norma della differenza
tra due iterazioni consecutive è minore di una tolleranza predefinita
($10^{-6}$).

## Flusso di Esecuzione

1.  **Inizializzazione MPI**: Avvio dei processi e determinazione di
    rank e size

2.  **Caricamento del grafo**: Solo il processo master carica il grafo
    dal file

3.  **Broadcast**: Il master trasmette il grafo a tutti i processi

4.  **Partizionamento**: I nodi vengono distribuiti equamente tra i
    processi

5.  **Inizializzazione PageRank**: Ogni nodo riceve un valore iniziale
    $1/N$

6.  **Ciclo iterativo**:

    -   Calcolo locale del contributo di PageRank

    -   Riduzione globale tramite MPI_Allreduce

    -   Aggiornamento dei valori di PageRank

    -   Calcolo della norma per verificare la convergenza

7.  **Verifica somma**: Controllo che la somma dei PageRank sia 1.0

8.  **Output**: Il master stampa i risultati e le metriche di
    performance

# Evoluzione: Da MPI Puro a Ibrido MPI+OpenMP

## Versione MPI Pura

Nella versione puramente MPI, ogni processo esegue il calcolo in modo
sequenziale sulla propria porzione di nodi:

``` {style="codestyle"}
// Calcolo CSR - Versione MPI pura
for (i = start; i < end; i++) {
    for (j = g.row_ptr[i]; j < g.row_ptr[i+1]; j++) {
        local[g.col_idx[j]] += g.val[j] * pr[i];
    }
}
```

Questa versione:

-   Utilizza solo MPI per la parallelizzazione

-   Ogni processo lavora su un singolo core

-   Non sfrutta il parallelismo intra-nodo

-   Può lasciare inutilizzati i core multipli disponibili su ciascun
    nodo

## Transizione a Ibrido MPI+OpenMP

Il passaggio a MPI+OpenMP ha richiesto le seguenti modifiche:

### 1. Inclusione dell'Header OpenMP

``` {style="codestyle"}
#include <omp.h>
```

### 2. Impostazione del Numero di Thread

Il numero di thread OpenMP viene letto come parametro da riga di
comando:

``` {style="codestyle"}
int omp_threads = atoi(argv[1]);
omp_set_num_threads(omp_threads);
```

### 3. Parallelizzazione del Calcolo CSR

Il calcolo principale viene parallelizzato con OpenMP:

``` {style="codestyle"}
// Calcolo CSR - Versione Ibrida
#pragma omp parallel for
for (i = start; i < end; i++) {
    for (j = g.row_ptr[i]; j < g.row_ptr[i+1]; j++) {
        #pragma omp atomic
        local[g.col_idx[j]] += g.val[j] * pr[i];
    }
}
```

**Punto critico**: L'uso di `#pragma omp atomic` è necessario per
garantire che gli aggiornamenti all'array `local` siano thread-safe,
poiché più thread possono tentare di aggiornare la stessa posizione.

### 4. Parallelizzazione della Fase di Aggiornamento

L'aggiornamento dei PageRank e il calcolo della norma utilizzano una
riduzione OpenMP:

``` {style="codestyle"}
#pragma omp parallel for reduction(+:norm_local)
for (i = 0; i < N; i++) {
    double newv = DAMPING * global[i] + (1.0 - DAMPING) / N;
    norm_local += (newv - pr[i]) * (newv - pr[i]);
    pr[i] = newv;
}
```

### 5. Parallelizzazione della Verifica della Somma

Anche la verifica finale della somma dei PageRank viene parallelizzata:

``` {style="codestyle"}
#pragma omp parallel for reduction(+:local_pr_sum)
for (i = start; i < end; i++) {
    local_pr_sum += pr[i];
}
```

## Confronto tra le Due Versioni

  **Aspetto**         **MPI Puro**                  **MPI+OpenMP Ibrido**
  ------------------- ----------------------------- ---------------------------------------------
  Parallelizzazione   Solo tra processi             Tra processi e tra thread
  Utilizzo CPU        1 core per processo           Multi-core per processo
  Overhead            Comunicazione MPI             Comunicazione MPI + sincronizzazione thread
  Scalabilità         Limitata dal numero di nodi   Sfrutta tutti i core disponibili
  Complessità         Minore                        Maggiore (gestione thread)

  : Confronto tra Versioni MPI Puro e Ibrido

# Dettagli Implementativi

## Partizionamento Bilanciato

Il codice implementa un partizionamento che distribuisce i nodi in modo
il più equo possibile:

:::: algorithm
::: algorithmic
$base \gets N / size$ $rem \gets N \mod size$
$counts[r] \gets base + (r < rem ? 1 : 0)$ $displs[0] \gets 0$
$displs[r] \gets displs[r-1] + counts[r-1]$ $start \gets displs[rank]$
$end \gets start + counts[rank]$
:::
::::

Questo approccio garantisce che ogni processo riceva approssimativamente
lo stesso numero di nodi, bilanciando il carico di lavoro.

## Gestione della Comunicazione

La comunicazione tra processi avviene tramite due principali chiamate
MPI:

1.  `MPI_Allreduce` per la somma dei contributi locali

2.  `MPI_Allreduce` per il calcolo globale della norma

Queste operazioni di riduzione globale sono il principale costo di
comunicazione e limitano la scalabilità su un numero elevato di
processi.

## Gestione della Concorrenza OpenMP

Le sfide principali nella gestione di OpenMP sono:

1.  **Aggiornamenti atomici**: Le operazioni atomiche garantiscono la
    correttezza ma introducono overhead

2.  **Condizioni di race**: La riduzione OpenMP evita race conditions
    nel calcolo della norma

3.  **False sharing**: L'allocazione degli array deve considerare
    l'allineamento della cache

# Vantaggi dell'Approccio Ibrido

## Vantaggi Specifici

1.  **Utilizzo efficiente delle risorse**: Sfrutta tutti i core
    disponibili su ciascun nodo

2.  **Riduzione della comunicazione**: Meno processi MPI significa meno
    comunicazione di rete

3.  **Flessibilità**: È possibile bilanciare il numero di processi MPI e
    thread OpenMP in base all'architettura

4.  **Scalabilità**: Migliore scaling su cluster con molti core per nodo

## Configurazione Ottimale

La configurazione ottimale dipende dall'architettura del cluster:

-   **Nodi con molti core**: Più thread OpenMP, meno processi MPI

-   **Nodi con pochi core**: Più processi MPI, meno thread OpenMP

-   **Regola generale**: Numero totale di thread = Numero di core
    disponibili

# Compilazione ed Esecuzione

## Compilazione

``` {style="codestyle"}
mpicc -o pr_hybrid pr_hybrid.c -lm -fopenmp
```

L'opzione `-fopenmp` è necessaria per abilitare il supporto OpenMP.

## Esecuzione

``` {style="codestyle"}
mpirun -np 4 -machinefile machinefile.txt pr_hybrid 2
```

Dove:

-   `-np 4`: numero di processi MPI

-   `-machinefile`: file con la lista dei nodi

-   `2`: numero di thread OpenMP per processo

# Limitazioni e Sfide

## Limitazioni Attuali

1.  **Sovraccarico delle operazioni atomiche**: Le operazioni atomiche
    possono diventare un collo di bottiglia

2.  **Memoria**: Ogni processo mantiene una copia completa del grafo

3.  **Comunicazione**: MPI_Allreduce su N elementi può essere costoso
    per grafi grandi

## Miglioramenti Futuri Possibili

1.  **Buffer locali per ridurre atomiche**: Ogni thread può accumulare
    in un buffer privato

2.  **Partizionamento basato su METIS**: Ridurre il taglio degli archi
    tra processi

3.  **Comunicazione asincrona**: Sovrapporre comunicazione e calcolo

# Verifica della Correttezza

Il codice include due meccanismi di verifica:

## 1. Verifica della Somma

La somma di tutti i PageRank deve essere 1.0 (entro una tolleranza):

``` {style="codestyle"}
#pragma omp parallel for reduction(+:local_pr_sum)
for (i = start; i < end; i++) {
    local_pr_sum += pr[i];
}
MPI_Reduce(&local_pr_sum, &global_pr_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER);
```

## 2. Verifica della Convergenza

L'algoritmo termina quando la norma della differenza tra iterazioni
consecutive è minore di $10^{-6}$:

$$\sqrt{\sum_{i=1}^{N} (PR^{(k+1)}(i) - PR^{(k)}(i))^2} < 10^{-6}$$

# Conclusioni

L'implementazione ibrida MPI+OpenMP del PageRank offre un buon
compromesso tra scalabilità ed efficienza. Il passaggio da MPI puro a
ibrido ha richiesto:

1.  L'aggiunta di direttive OpenMP per parallelizzare i loop

2.  La gestione della concorrenza tramite atomiche e riduzioni

3.  La configurazione flessibile del numero di thread

L'approccio ibrido permette di sfruttare al meglio le architetture
moderne con multi-core, migliorando significativamente le performance
rispetto alla versione puramente MPI.

# Guida Rapida

## Preparazione Dataset

Il dataset deve essere nel formato:

``` {style="codestyle"}
source_node destination_node
source_node destination_node
...
```

## Comandi Utili

``` {style="codestyle"}
# Compilazione
mpicc -o pr_hybrid pr_hybrid.c -lm -fopenmp

# Esecuzione con 2 processi MPI e 4 thread OpenMP per processo
mpirun -np 2 -machinefile machinefile.txt pr_hybrid 4

# Verifica delle configurazioni
for threads in 1 2 4 8; do
    mpirun -np 4 -machinefile machinefile.txt pr_hybrid $threads
done
```
