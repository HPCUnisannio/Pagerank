# PageRank Ibrido MPI+OpenMP: Analisi Comparativa e Scelte Progettuali

## Panoramica delle Tre Implementazioni

Questo documento presenta l'analisi comparativa di tre implementazioni ibride MPI+OpenMP dell'algoritmo PageRank Power Iteration:

1. Versione Base: Architettura a I/O Multi-Reader con post-processing centralizzato sul Master
2. Versione Centralizzata: Architettura a I/O Centralizzato con post-processing distribuito simmetrico
3. Versione Replicata: Architettura a Dati Replicati con post-processing distribuito e zero comunicazioni di setup

Tutte e tre condividono l'approccio a grana grossa (coarse-grain) con pool di thread OpenMP creato una sola volta all'esterno del ciclo do-while e la privatizzazione della memoria per eliminare le direttive atomic.

---

# PARTE 1: VERSIONE BASE (Post-Processing Centralizzato)

## Filosofia Progettuale: Parallelismo Ibrido a Grana Grossa

L'implementazione combina MPI per il parallelismo a memoria distribuita tra nodi e OpenMP per il parallelismo a memoria condivisa all'interno di ciascun nodo. L'architettura adotta un approccio a grana grossa (coarse-grain): il pool di thread OpenMP viene creato una sola volta all'esterno del ciclo do-while principale, eliminando l'overhead di creazione/distruzione thread a ogni iterazione.

## Modello di Threading MPI: MPI_THREAD_FUNNELED

Viene utilizzato il livello di supporto MPI_THREAD_FUNNELED:

MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_support);

Questo garantisce che solo il thread master di ciascun processo possa effettuare chiamate MPI, mentre gli altri thread sono dedicati esclusivamente al calcolo. Tutte le comunicazioni MPI sono protette dalla direttiva #pragma omp master.

## Privatizzazione della Memoria OpenMP

### Array thread_sums

Ogni thread OpenMP alloca il proprio vettore sum privato di dimensione NODES:

thread_sums[tid] = (double *)calloc(NODES, sizeof(double));
double *local_sum = thread_sums[tid];

Durante la SpMV, ogni thread scrive esclusivamente nel proprio local_sum, eliminando la necessità di direttive atomic e garantendo scritture a contesa zero.

### Riduzione Locale dei Thread

Dopo la SpMV, i vettori privati vengono sommati nel vettore sum del processo tramite un ciclo parallelo:

#pragma omp for
for (i = 0; i < NODES; i++) {
    double total_row_sum = 0.0;
    int t;
    for (t = 0; t < num_threads; t++) {
        if (thread_sums[t] != NULL)
            total_row_sum += thread_sums[t][i];
    }
    sum[i] = total_row_sum;
}

## Sincronizzazione e Barriere (Versione Base)

Il flusso di esecuzione richiede 7 punti di sincronizzazione per iterazione:

| Barriera | Tipo | Scopo |
|----------|------|-------|
| Barriera 1 | #pragma omp barrier (esplicita) | Attende che il master thread abbia completato MPI_Scatterv |
| Barriera 2 | #pragma omp for (implicita) | Attende che tutti i thread abbiano finito la SpMV |
| Barriera 3 | #pragma omp for (implicita) | Attende che tutti i thread abbiano finito la riduzione locale |
| Barriera 4 | #pragma omp barrier (esplicita) | Attende che il master thread abbia completato MPI_Reduce |
| Barriera 5 | #pragma omp for (implicita) | Attende il completamento del calcolo dangling mass |
| Barriera 6 | #pragma omp for (implicita) | Attende il completamento di damping e norma |
| Barriera 7 | #pragma omp barrier (esplicita) | Attende che il master thread abbia completato MPI_Bcast |

## Ruolo dei Thread nelle Comunicazioni MPI

Solo il master thread di ogni processo partecipa alle comunicazioni MPI:

- MPI_Scatterv: Il master thread del processo 0 invia, i master thread dei worker ricevono
- MPI_Reduce: I master thread dei worker inviano, il master thread del processo 0 riceve e somma
- MPI_Bcast: Il master thread del processo 0 invia norm, i master thread dei worker ricevono

Tutti gli altri thread OpenMP rimangono in attesa alla barriera corrispondente.

## Post-Processing Centralizzato sul Master

Solo il processo master (rank == 0) esegue il post-processing, parallelizzandolo con OpenMP:

- Calcolo della dangling_mass con reduction(+:dangling_mass)
- Applicazione del damping e calcolo della norma con reduction(+:norm_sq) in un unico ciclo fuso
- Calcolo della norma finale norm = sqrt(norm_sq) eseguito dal solo master thread

## Ottimizzazioni Implementate

### Eliminazione Array Densi

Gli array damp1, damp2 e diff sono sostituiti da costanti scalari:

const double DAMP1 = DAMPING;
const double DAMP2 = (1.0 - DAMPING) / NODES;

### Fusione dei Cicli di Post-Processing

Damping, calcolo diff e aggiornamento prold sono fusi in un unico ciclo:

#pragma omp for reduction(+ : norm_sq)
for (i = 0; i < NODES; i++) {
    prnew[i] = prnew[i] * DAMP1 + DAMP2 + redistribution;
    double diff = prnew[i] - prold[i];
    norm_sq += diff * diff;
    prold[i] = prnew[i];
}

Questo riduce il numero di barriere implicite e migliora la località della cache.

### Scheduling Dinamico

La SpMV utilizza schedule(dynamic, 512) per bilanciare il carico tra i thread in caso di distribuzione irregolare degli archi.

## Comando di Compilazione ed Esecuzione (Versione Base)

Compilazione:

mpicc mpi_openMP/pr_mpi_openmp.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp -Ilibraries -fopenmp -lm

Esecuzione con 4 processi e 2 thread per processo:

mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp 2

---

# PARTE 2: VERSIONE CENTRALIZZATA (I/O Centralizzato, Post-Processing Distribuito)

## Filosofia Progettuale: I/O Centralizzato e Post-Processing Distribuito

Questa variante introduce un modello a memoria distribuita pura per la gestione della matrice del grafo. A differenza della versione base in cui ogni processo leggeva ridondantemente il file intero, la gestione dell'I/O e la costruzione della struttura dati iniziale sono completamente centralizzate sul processo MASTER, mentre il post-processing è distribuito simmetricamente tra tutti i processi.

## Innovazioni Rispetto alla Versione Base

### 1. I/O Centralizzato e Isolamento dei Worker

Solo il processo MASTER (rank == 0) apre il file, esegue il parsing e costruisce la struttura CSC. I nodi di calcolo non accedono al disco, eliminando i colli di bottiglia legati all'I/O concorrente su cluster.

### 2. Massima Efficienza della Memoria (RAM)

Nella versione base, ogni processo manteneva in memoria gli interi array globali val e rowind di dimensione EDGES. In questa versione, gli array globali vengono allocati esclusivamente dal Master. I Worker allocano solo i buffer locali rec_val e rec_row di dimensione my_cnt. Il Master, subito dopo la distribuzione, libera la matrice globale.

### 3. Distribuzione Intelligente in Due Fasi

Fase 1 (Metadati): MPI_Bcast degli array di controllo colptr e readsum. Ogni processo calcola autonomamente le dimensioni dei blocchi di colonne (pcols) e il numero di non-zeri associati (sendcnts e displs).

Fase 2 (Dati Pesanti): MPI_Scatterv per distribuire i valori della matrice (val) e gli indici di riga (rowind). Ogni processo riceve solo ciò che deve effettivamente computare.

### 4. Eliminazione MPI_Scatterv a Ogni Iterazione

A differenza della versione base, non è più necessario distribuire il vettore prold a ogni iterazione. Poiché ogni processo conosce la propria porzione di colonne tramite displs_pr, può leggere direttamente da prold utilizzando l'indice globale global_col.

## Flusso di Esecuzione per Iterazione (Versione Centralizzata)

### Fase A: Prodotto Matrice-Vettore (SpMV) Parallelo

I thread OpenMP eseguono la SpMV ciclando sulle colonne locali (rec_col). L'accumulo avviene sui vettori privati local_sum. prold[global_col] viene letto direttamente senza necessità di un buffer rec_pr.

### Fase B: Riduzione Thread Locali

I vettori privati dei thread vengono ridotti nel vettore sum locale al processo.

### Fase C: Overlap Rete/CPU con Iallreduce

Viene lanciata MPI_Iallreduce asincrona sul vettore sum. Mentre la rete scambia i dati globali, la CPU calcola in parallelo la dangling mass locale. Segue MPI_Allreduce per dm_global e MPI_Wait per attendere prnew.

### Fase D: Aggiornamento Distribuito e Norma Locale

Ogni processo applica il damping e calcola lo scarto quadratico solo per le proprie colonne (rec_col), non su tutti i NODES:

#pragma omp for reduction(+ : norm_sq_local)
for (i = 0; i < rec_col; i++) {
    int global_col = global_col_start + i;
    prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;
    double diff = prnew[global_col] - prold[global_col];
    norm_sq_local += diff * diff;
    prold[global_col] = prnew[global_col];
}

### Fase E: Convergenza Globale

MPI_Allreduce somma le norme parziali scalari di tutti i processi:

MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
norm = sqrt(norm_sq_global);

## Sincronizzazione e Barriere (Versione Centralizzata)

| Barriera | Tipo | Scopo |
|----------|------|-------|
| Barriera 1 | #pragma omp barrier (esplicita) | Attende completamento MPI_Iallreduce e MPI_Allreduce |
| Barriera 2 | #pragma omp for (implicita) | Attende completamento aggiornamento e norma |
| Barriera 3 | #pragma omp barrier (esplicita) | Attende completamento MPI_Allreduce della norma |

La riduzione da 7 a 3 barriere per iterazione è resa possibile da:
- Eliminazione di MPI_Scatterv per iterazione (non più necessario)
- Utilizzo di MPI_Iallreduce che non richiede barriera esplicita prima del damping
- Fusione di damping e calcolo norma in un unico ciclo

## Comando di Compilazione ed Esecuzione (Versione Centralizzata)

Compilazione:

mpicc mpi_openMP/pr_mpi_openmp_centralized.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp_centralized -Ilibraries -fopenmp -lm

Esecuzione con 4 processi e 2 thread per processo:

mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp_centralized 2

---

# PARTE 3: VERSIONE REPLICATA (Dati Replicati, Zero Comunicazioni di Setup)

## Filosofia Progettuale: Dati Replicati e Post-Processing Distribuito

Questa variante adotta un approccio a dati replicati: ogni processo legge indipendentemente l'intero dataset e mantiene in memoria l'intera matrice CSC. In cambio, elimina completamente la fase di distribuzione iniziale dei dati (MPI_Scatterv) e la distribuzione del vettore PageRank a ogni iterazione, puntando direttamente alla propria porzione tramite aritmetica dei puntatori.

## Innovazioni Rispetto alle Versioni Precedenti

### 1. Eliminazione Totale delle Comunicazioni di Distribuzione

Nessuna MPI_Scatterv iniziale per val e rowind. Ogni processo usa direttamente i puntatori locali:

double *rec_val = val + displs[rank];
int    *rec_row = rowind + displs[rank];
rec_col = pcols[rank];

Nessuna MPI_Scatterv a ogni iterazione per prold. Ogni processo legge direttamente prold[global_col] usando l'indice globale.

### 2. Post-Processing Distribuito Simmetrico al 100%

Il calcolo del damping, la redistribuzione della massa, l'aggiornamento di prold e il calcolo della norma vengono eseguiti da ogni processo solo sulla propria porzione di colonne (rec_col). Questo garantisce scalabilità perfetta della fase di update.

### 3. Overlap Comunicazione-Calcolo

MPI_Iallreduce asincrona per la riduzione del vettore sum. Mentre la rete scambia i dati, le CPU calcolano la dangling mass locale. La MPI_Wait viene chiamata solo prima di applicare la redistribuzione.

## Flusso di Esecuzione per Iterazione (7 Blocchi)

### Blocco 1: Reset Locale
Azzera i vettori privati local_sum e gli accumulatori dm_local e norm_sq_local.

### Blocco 2: SpMV Parallela
Prodotto matrice-vettore con vettori privatizzati. Ogni thread scrive sul proprio local_sum.

### Blocco 3: Riduzione Thread Locali
Fonde i vettori privati dei thread nel vettore sum del processo.

### Blocco 4: Riduzione MPI Asincrona
Il master thread avvia MPI_Iallreduce per sommare i vettori sum di tutti i processi in prnew.

### Blocco 5: Overlap - Calcolo Dangling Mass
Mentre la rete lavora, i thread calcolano la dangling mass locale. Segue MPI_Allreduce per sommare i valori e MPI_Wait per garantire l'arrivo di prnew.

### Blocco 6: Aggiornamento Distribuito
Ogni processo aggiorna prnew e prold solo sulle proprie colonne, calcolando la norma parziale.

### Blocco 7: Convergenza Globale
MPI_Allreduce somma le norme parziali. Tutti i processi calcolano norm = sqrt(norm_sq_global).

## Sincronizzazione e Barriere (Versione Replicata)

| Barriera | Tipo | Scopo |
|----------|------|-------|
| Barriera 1 | #pragma omp for (implicita) | Attende completamento SpMV |
| Barriera 2 | #pragma omp for (implicita) | Attende completamento riduzione thread |
| Barriera 3 | #pragma omp for (implicita) | Attende completamento calcolo dangling mass |
| Barriera 4 | #pragma omp barrier (esplicita) | Attende MPI_Wait e dm_global |
| Barriera 5 | #pragma omp for (implicita) | Attende completamento aggiornamento |
| Barriera 6 | #pragma omp barrier (esplicita) | Attende MPI_Allreduce della norma |

Totale: 6 barriere per iterazione.

## Comando di Compilazione ed Esecuzione (Versione Replicata)

Compilazione:

mpicc mpi_openMP/pr_mpi_openmp_replicated.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp_replicated -Ilibraries -fopenmp -lm

Esecuzione con 4 processi e 2 thread per processo:

mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp_replicated 2

---

# PARTE 4: CONFRONTO TRA LE TRE VERSIONI

## Tabella Comparativa Globale

| Parametro | Versione Base | Versione Centralizzata | Versione Replicata |
|:---|:---|:---|:---|
| I/O file | Multi-Reader (tutti i processi) | Solo Master | Multi-Reader (tutti i processi) |
| Memoria val/rowind | Su tutti i processi | Solo Master, poi freed | Su tutti i processi |
| MPI_Scatterv iniziale | Si | Si | No (puntatori locali) |
| Distribuzione prold a ogni iter | MPI_Scatterv | Non necessaria | Non necessaria |
| Post-processing | Centralizzato sul Master | Distribuito simmetrico | Distribuito simmetrico |
| Riduzione vettore | MPI_Reduce + MPI_Bcast | MPI_Iallreduce | MPI_Iallreduce |
| Riduzione norma | MPI_Bcast dal Master | MPI_Allreduce | MPI_Allreduce |
| Overlap calcolo-comunicazione | No | Si (dangling mass) | Si (dangling mass) |
| Barriere per iterazione | 7 | 3 | 6 |
| Collo di bottiglia | Master (post-processing seriale) | Master (solo I/O iniziale) | Nessuno (solo I/O iniziale) |

## Confronto Memoria per Processo

| Struttura | Versione Base | Versione Centralizzata | Versione Replicata |
|-----------|---------------|----------------------|---------------------|
| val | EDGES double (tutti) | Solo Master, poi freed | EDGES double (tutti) |
| rowind | EDGES int (tutti) | Solo Master, poi freed | EDGES int (tutti) |
| colptr | NODES+1 int (tutti) | NODES+1 int (tutti) | NODES+1 int (tutti) |
| readsum | NODES int (tutti) | NODES int (tutti) | NODES int (tutti) |
| rec_val | sendcnts[rank] double | sendcnts[rank] double | Puntatore (no alloc) |
| rec_row | sendcnts[rank] int | sendcnts[rank] int | Puntatore (no alloc) |
| rec_pr | pcols[rank] double | Non allocato | Non allocato |
| thread_sums | num_threads * NODES double | num_threads * NODES double | num_threads * NODES double |
| full_pr | Non allocato | NODES double (finale) | NODES double (finale) |

## Confronto Comunicazioni per Iterazione

| Comunicazione | Versione Base | Versione Centralizzata | Versione Replicata |
|---------------|---------------|----------------------|---------------------|
| Distribuzione prold | MPI_Scatterv (pcols[rank] double) | Nessuna | Nessuna |
| Riduzione vettore | MPI_Reduce (NODES double) | MPI_Iallreduce (NODES double) | MPI_Iallreduce (NODES double) |
| Dangling mass | Calcolo solo Master | MPI_Allreduce (1 double) | MPI_Allreduce (1 double) |
| Norma | MPI_Bcast (1 double) | MPI_Allreduce (1 double) | MPI_Allreduce (1 double) |
| Raccolta finale | Non necessaria | MPI_Allgatherv | MPI_Allgatherv |

## Vantaggi e Svantaggi

### Versione Base

Vantaggi:
- Codice più semplice e manutenibile
- Flusso lineare facile da debuggare
- Minore numero di allocazioni dinamiche
- Recupero memoria rec_val/rec_row possibile

Svantaggi:
- I/O ridondante: tutti i processi leggono lo stesso file
- MPI_Scatterv a ogni iterazione per distribuire prold
- Collo di bottiglia seriale sul master per post-processing
- Worker in idle durante post-processing
- 7 barriere per iterazione
- Operazioni di rete completamente bloccanti

### Versione Centralizzata

Vantaggi:
- I/O centralizzato: elimina accesso ridondante al file system
- Memoria ridotta: worker allocano solo buffer locali
- Nessuna distribuzione prold a ogni iterazione
- Post-processing distribuito: carico bilanciato
- Overlap calcolo-comunicazione con MPI_Iallreduce
- Solo 3 barriere per iterazione
- Scalabilità in memoria: il grafo è distribuito

Svantaggi:
- Collo di bottiglia sul Master in fase di setup (lettura sequenziale + MPI_Scatterv)
- MPI_Scatterv iniziale grava interamente sul Master
- Per grafi enormi, la fase di inizializzazione può dominare il tempo totale
- Maggiore complessità del codice
- MPI_Allgatherv finale aggiunge comunicazione extra

### Versione Replicata

Vantaggi:
- Zero comunicazioni di distribuzione dati (né iniziali né per iterazione)
- Post-processing completamente distribuito
- Overlap calcolo-comunicazione con MPI_Iallreduce
- Setup di rete immediato (nessuna scatterv)
- Aritmetica dei puntatori a costo zero per accesso ai dati locali

Svantaggi:
- Memoria replicata: ogni processo alloca l'intera matrice CSC
- I/O ridondante: tutti i processi leggono lo stesso file
- Non scala in memoria: il grafo deve entrare nella RAM di ogni nodo
- 6 barriere per iterazione (vs 3 della versione centralizzata)
- Recupero memoria rec_val/rec_row non possibile (puntatori, non allocazioni)

## Profiling Integrato (Comune a Tutte)

Tutte le versioni includono la misurazione dettagliata dei tempi:

- SpMV locale: tempo del prodotto matrice-vettore parallelizzato
- Riduzione thread: tempo di fusione dei vettori privati dei thread
- Allreduce+attesa: tempo di comunicazione MPI (inclusa attesa)
- Aggiornamento: tempo di damping e calcolo norma locale
- Norma MPI: tempo di riduzione globale della norma

## Criteri di Scelta

| Scenario | Versione Consigliata |
|----------|---------------------|
| RAM limitata sui nodi worker | Versione Centralizzata |
| File system lento/condiviso | Versione Centralizzata |
| Rete ad alta latenza | Versione Replicata (zero scatterv per iterazione) |
| Grafi che entrano in RAM su ogni nodo | Versione Replicata (massime prestazioni) |
| Massima scalabilità in memoria | Versione Centralizzata |
| Debugging e semplicità | Versione Base |
| Bilanciamento generale | Versione Centralizzata |

## Ottimizzazioni Comuni a Tutte le Versioni

- Regione parallela a grana grossa: pool di thread creato una sola volta
- Privatizzazione memoria OpenMP: nessuna direttiva atomic nella SpMV
- Eliminazione array densi: costanti scalari DAMP1 e DAMP2
- Scheduling dinamico con chunk=512 per bilanciamento carico
- Fusione dei cicli di post-processing per località della cache
- Profiling integrato per analisi prestazioni

## Correttezza Matematica (Comune a Tutte)

Tutte le implementazioni garantiscono:

1. Inizializzazione stocastica: somma iniziale = 1.0
2. Gestione dangling nodes: redistribuzione uniforme della massa persa
3. Applicazione corretta del damping: formula eseguita dopo accumulo completo
4. Verifica convergenza: norma L2 calcolata su tutto il vettore
5. Validazione finale: somma PageRank = 1.0