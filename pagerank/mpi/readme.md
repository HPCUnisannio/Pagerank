# PageRank MPI Distribuito: Analisi Comparativa e Scelte Progettuali

## Panoramica delle Due Implementazioni

Questo documento presenta l'analisi comparativa di due implementazioni parallele dell'algoritmo PageRank Power Iteration utilizzando MPI puro (nessun multithreading):

1. Versione Base: Architettura Multi-Reader con riduzione al master e post-processing seriale
2. Versione Ottimizzata: Architettura a I/O Centralizzato con post-processing simmetrico e overlap calcolo-comunicazione

---

# PARTE 1: VERSIONE BASE MPI

## Filosofia Progettuale: Parallelismo a Memoria Distribuita

L'implementazione sfrutta il Message Passing Interface (MPI) per distribuire il calcolo del PageRank su più processi potenzialmente eseguiti su nodi fisici diversi. La strategia di partizionamento adottata è il 1D Column Partitioning.

## Strategia di Partizionamento: 1D Column Partitioning

La matrice sparsa del grafo è salvata in formato CSC (Compressed Sparse Column). Il parallelismo si ottiene dividendo le colonne tra i processi MPI. Poiché le colonne rappresentano i link uscenti dei nodi, ogni processo diventa responsabile di calcolare l'impatto di un sottoinsieme di nodi sul resto della rete.

## Strutture di Partizionamento

### Array di Mappatura

Per gestire la distribuzione irregolare degli archi, vengono calcolati quattro array:

- pcols: Numero di colonne (nodi) assegnate a ciascun processo
- sendcnts: Numero di archi non-zero contenuti nelle colonne assegnate
- displs: Offset nell'array globale da cui iniziare l'estrazione per ogni processo
- displs_pr: Offset per la distribuzione del vettore PageRank

### Calcolo dei Chunk Irregolari

Sfruttando l'array cumulativo colptr del formato CSC:

sendcnts[i] = colptr[j] - colptr[k]

dove j è la fine blocco e k è l'inizio blocco del processo i. Una singola sottrazione determina il numero esatto di archi, senza iterare.

## Primitive MPI Utilizzate (Versione Base)

- MPI_Scatter: Distribuisce array in parti uguali (numero colonne per processo)
- MPI_Scatterv: Distribuisce array in parti di dimensioni diverse (archi e indici)
- MPI_Reduce: Somma i risultati parziali di tutti i processi sul master
- MPI_Bcast: Trasmette la norma a tutti i processi per la decisione di convergenza
- MPI_Barrier: Sincronizza tutti i processi (solo per benchmarking)
- MPI_Wtime: Misura il tempo reale (wall-clock) trascorso

## Flusso di Esecuzione per Iterazione (Versione Base)

### Fase 1: Distribuzione Vettore Corrente

Il master invia a ogni processo solo la porzione del vettore PageRank associata ai nodi di competenza:

MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE, rec_pr, pcols[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

### Fase 2: Moltiplicazione Sparsa Locale

Ogni processo calcola i contributi dei propri nodi sorgente, accumulandoli nell'array locale sum. L'indice locale viene tradotto in indice globale sottraendo displs[rank]:

for (local_col = 0; local_col < rec_col; local_col++) {
    int global_col = global_col_start + local_col;
    int start_idx  = colptr[global_col] - displs[rank];
    int end_idx    = colptr[global_col + 1] - displs[rank];
    for (j = start_idx; j < end_idx; j++) {
        sum[rec_row[j]] += rec_val[j] * rec_pr[local_col];
    }
}

### Fase 3: Riduzione Globale

Somma gli array sum di tutti i processi. Il risultato prnew risiede solo sul master:

MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

### Fase 4: Computazione Master

Solo il master esegue:
- Calcolo della dangling_mass dai nodi senza link uscenti (readsum[i] == 0)
- Applicazione del damping: prnew[i] = prnew[i] * damp1[i] + damp2[i] + redistribution
- Calcolo della norma L2 per la convergenza: norm_sq = somma(diff[i] * diff[i])
- Aggiornamento del vettore prold = prnew

### Fase 5: Broadcast Convergenza

Tutti i processi ricevono norm e valutano la condizione while(norm > ERROR) all'unisono:

MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

## Correttezza Matematica

### Inizializzazione Stocastica

Il vettore PageRank iniziale è 1.0 / NODES, garantendo somma uguale a 1.0.

### Gestione Dangling Nodes

I nodi senza link uscenti (out-degree = 0) causerebbero perdita di massa di probabilità. Il master calcola la massa totale persa e la ridistribuisce uniformemente:

dangling_mass = somma prold[i] per readsum[i] == 0
redistribution = (dangling_mass * DAMPING) / NODES

### Applicazione Centralizzata del Damping

Il damping viene applicato dal master dopo la riduzione globale, garantendo che la formula sia eseguita una sola volta per nodo.

## Fix Critici Rispetto alla Versione Originale

- Buffer Overflow (buffer fissi 100k): Allocazione dinamica basata su sendcnts[rank] e pcols[rank]
- Memory Explosion (matrice densa): Moltiplicazione puramente sparsa iterando su rec_val e rec_row
- Heap Corruption (parser fragile): Parser robusto con gestione "buchi" negli indici dei nodi
- Probabilità iniziale errata (0.25 fisso): Inizializzazione 1.0 / NODES
- Perdita massa dangling nodes: Calcolo e redistribuzione della dangling_mass
- Misurazione tempo errata (clock()): Sostituita con MPI_Wtime() con MPI_Barrier

## Comando di Compilazione ed Esecuzione (Versione Base)

Compilazione:

mpicc mpi/pr_mpi.c libraries/data.c libraries/measure.c -o mpi/pr_mpi -Ilibraries -lm

Esecuzione con 4 processi:

mpirun -np 4 -machinefile mpi/machinefile.txt mpi/pr_mpi

---

# PARTE 2: VERSIONE OTTIMIZZATA MPI

## Filosofia Progettuale: I/O Centralizzato e Post-Processing Simmetrico

L'implementazione ottimizzata introduce quattro miglioramenti chiave rispetto alla versione base: I/O centralizzato sul master, eliminazione di array densi in favore di costanti scalari, post-processing distribuito simmetrico, e overlap tra calcolo e comunicazione tramite operazioni asincrone.

## Ottimizzazione 1: I/O Centralizzato

### Versione Base (Multi-Reader)
Tutti i processi aprivano e leggevano simultaneamente l'intero file dataset, generando overhead I/O su file system condivisi. Ogni processo allocava e manteneva in RAM gli array globali completi val e rowind.

### Versione Ottimizzata (Master-Worker)
Solo il master accede al disco, esegue il parsing sequenziale e costruisce la CSC. Il master distribuisce tramite MPI_Scatterv solo i frammenti necessari a ogni processo. I worker ricevono solo rec_val e rec_row di dimensione sendcnts[rank], una frazione della RAM rispetto a prima.

## Ottimizzazione 2: Eliminazione Array Densi

### Versione Base
Allocava array densi damp1, damp2 e diff, ognuno da NODES elementi double, solo per contenere valori costanti ripetuti o delta temporanei.

### Versione Ottimizzata
Gli array inutili sono sostituiti da costanti scalari calcolate a tempo di compilazione:

const double DAMP1 = DAMPING;
const double DAMP2 = (1.0 - DAMPING) / NODES;

Risparmio di decine di Megabyte e riduzione dei cache miss della CPU.

## Ottimizzazione 3: Post-Processing Simmetrico

### Versione Base
Dopo la SpMV parallela, tutti i processi inviavano la somma parziale al master tramite MPI_Reduce. Il master, da solo e in modo seriale, applicava damping, calcolava la massa dei nodi pozzo e la norma, mentre gli altri processi rimanevano in idle.

### Versione Ottimizzata
Nessun processo resta inattivo. Utilizzando MPI_Allreduce, tutti ottengono il vettore grezzo prnew. Ogni processo applica il damping, la redistribuzione della massa pozzo e calcola la norma solo sulla propria porzione di colonne (rec_col). Carico perfettamente bilanciato.

## Ottimizzazione 4: Overlap Calcolo-Comunicazione

### Versione Base
Operazioni strettamente sequenziali e bloccanti: calcolo, poi rete (MPI_Reduce), poi attesa (MPI_Bcast).

### Versione Ottimizzata
Viene introdotta l'istruzione asincrona MPI_Iallreduce. Mentre le schede di rete gestiscono lo scambio e la somma dei voti PageRank in background, le CPU calcolano la dangling mass locale, nascondendo i tempi morti di rete.

MPI_Request request;
MPI_Iallreduce(local_sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

// Calcolo dangling mass mentre la rete lavora
for (local_col = 0; local_col < rec_col; local_col++) {
    int global_col = global_col_start + local_col;
    if (readsum[global_col] == 0) {
        dm_local += prold[global_col];
    }
}

MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Wait(&request, MPI_STATUS_IGNORE);

## Flusso di Esecuzione per Iterazione (Versione Ottimizzata)

### Fase A: Moltiplicazione Sparsa Locale
Ogni processo calcola i contributi dei propri nodi sorgente nell'array locale local_sum, usando solo i propri frammenti rec_val e rec_row.

### Fase B: Riduzione Asincrona e Overlap
MPI_Iallreduce viene lanciata in modalità non bloccante. Mentre la rete somma i local_sum di tutti i processi nel vettore prnew, ogni processo calcola la propria dangling mass locale.

### Fase C: Post-Processing Distribuito
Dopo aver ricevuto dm_global e atteso il completamento di prnew, ogni processo applica damping e redistribuzione solo sulle proprie colonne:

prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

Calcola inoltre la norma parziale norm_sq_local.

### Fase D: Convergenza Globale
MPI_Allreduce somma le norme parziali di tutti i processi. Ogni processo calcola norm = sqrt(norm_sq_global) e valuta la condizione di uscita.

## Comando di Compilazione ed Esecuzione (Versione Ottimizzata)

Compilazione:

mpicc mpi/pr_mpi_opt.c libraries/data.c libraries/measure.c -o mpi/pr_mpi_opt -Ilibraries -lm

Esecuzione con 4 processi:

mpirun -np 4 -machinefile mpi/machinefile.txt mpi/pr_mpi_opt

---

# PARTE 3: CONFRONTO TRA LE DUE VERSIONI

## Tabella Comparativa Globale

| Parametro | Versione Base | Versione Ottimizzata |
|:---|:---|:---|
| Modello I/O | Multi-Reader (tutti leggono il file) | Master-Worker (solo master legge) |
| Post-Processing | Seriale sul master | Distribuito simmetrico |
| Comunicazione | Bloccante (Reduce + Bcast) | Asincrona con overlap (Iallreduce) |
| Array Densi | damp1[NODES], damp2[NODES], diff[NODES] | Costanti scalari DAMP1, DAMP2 |
| Memoria Worker | val[NODES], rowind[NODES] completi | Solo rec_val e rec_row locali |
| Carico Master | Elevato (damping e norma seriali) | Bilanciato (stesso carico dei worker) |
| Tempi Morti | Worker in idle durante fase master | Nessun idle (overlap calcolo-comunicazione) |

## Confronto Primitive MPI

| Operazione | Versione Base | Versione Ottimizzata |
|------------|---------------|----------------------|
| Distribuzione iniziale val | MPI_Scatterv | MPI_Scatterv |
| Distribuzione iniziale rowind | MPI_Scatterv | MPI_Scatterv |
| Distribuzione vettore PR a ogni iter | MPI_Scatterv | Non necessaria (prold locale) |
| Riduzione vettore | MPI_Reduce (solo master riceve) | MPI_Iallreduce (tutti ricevono, asincrono) |
| Riduzione norma | MPI_Bcast dal master | MPI_Allreduce (tutti calcolano) |
| Riduzione dangling | Calcolo solo master | MPI_Allreduce (tutti contribuiscono) |
| Raccolta finale | Non necessaria (master ha tutto) | MPI_Allgatherv |

## Confronto Memoria per Processo

| Struttura | Versione Base | Versione Ottimizzata |
|-----------|---------------|----------------------|
| val | NODES double (su ogni processo) | Solo master, poi freed |
| rowind | NODES int (su ogni processo) | Solo master, poi freed |
| damp1 | NODES double | Eliminato (costante scalare) |
| damp2 | NODES double | Eliminato (costante scalare) |
| diff | NODES double | Eliminato (variabile locale) |
| rec_val | sendcnts[rank] double | sendcnts[rank] double |
| rec_row | sendcnts[rank] int | sendcnts[rank] int |
| local_sum / sum | NODES double | NODES double |
| rec_pr | pcols[rank] double | Non necessario (usa prold direttamente) |
| full_pr | Non allocato | NODES double (solo raccolta finale) |

## Vantaggi e Svantaggi

### Versione Base

Vantaggi:
- Codice più semplice e manutenibile
- Flusso lineare facile da debuggare
- Non richiede gestione di richieste asincrone
- Minore numero di comunicazioni collettive per iterazione

Svantaggi:
- I/O ridondante: tutti i processi leggono lo stesso file
- Memoria sprecata: array globali su ogni processo
- Collo di bottiglia seriale sul master
- Worker in idle durante post-processing
- Operazioni di rete completamente bloccanti

### Versione Ottimizzata

Vantaggi:
- I/O centralizzato: elimina accesso ridondante al file system
- Memoria ridotta: niente array damp1, damp2, diff, val, rowind sui worker
- Carico bilanciato: tutti i processi partecipano al post-processing
- Overlap calcolo-comunicazione: latenza di rete nascosta
- Scalabilità migliorata: nessun collo di bottiglia seriale sul master

Svantaggi:
- Maggiore complessità del codice
- MPI_Iallreduce richiede gestione esplicita delle richieste
- MPI_Allgatherv finale aggiunge una comunicazione collettiva extra
- La memoria local_sum[NODES] rimane allocata su ogni processo

## Note sulle Comunicazioni per Iterazione

### Versione Base

- Scatterv iniziale (val): sendcnts[rank] double
- Scatterv iniziale (rowind): sendcnts[rank] int
- Scatter (pcols): 1 int
- Scatterv iterazione (prold): pcols[rank] double
- Reduce (sum): NODES double
- Bcast (norm): 1 double

### Versione Ottimizzata

- Scatterv iniziale (val): sendcnts[rank] double
- Scatterv iniziale (rowind): sendcnts[rank] int
- Bcast (colptr): NODES+1 int (solo iniziale)
- Bcast (readsum): NODES int (solo iniziale)
- Iallreduce (local_sum): NODES double (asincrono)
- Allreduce (dm): 1 double
- Allreduce (norm_sq): 1 double
- Allgatherv finale (prold): pcols[rank] double

## Criteri di Scelta

| Scenario | Versione Consigliata |
|----------|---------------------|
| Pochi processi (2-4) | Versione Base (overhead comunicazione accettabile) |
| Molti processi (8+) | Versione Ottimizzata (colli di bottiglia amplificati) |
| File system lento/condiviso | Versione Ottimizzata (I/O centralizzato) |
| RAM limitata sui nodi | Versione Ottimizzata (minor footprint) |
| Rete ad alta latenza | Versione Ottimizzata (overlap nasconde latenza) |
| Debugging e manutenzione | Versione Base (codice più semplice) |
| Massime prestazioni | Versione Ottimizzata |

## Correttezza Matematica (Comune a Entrambe)

Entrambe le implementazioni garantiscono:

1. Inizializzazione stocastica: somma iniziale = 1.0
2. Gestione dangling nodes: redistribuzione uniforme della massa persa
3. Applicazione corretta del damping: formula eseguita dopo accumulo completo
4. Verifica convergenza: norma L2 calcolata su tutto il vettore
5. Validazione finale: somma PageRank = 1.0

## Possibili Ottimizzazioni Future (Comuni)

- Lettura file solo dal master con successiva distribuzione (già implementata nella versione ottimizzata)
- Overlap di comunicazione e calcolo con operazioni non bloccanti (già implementato nella versione ottimizzata)
- Parallelizzazione della fase di damping e convergenza (già implementato nella versione ottimizzata)
- Utilizzo di MPI-IO per lettura parallela del file
- Compressione dei dati trasmessi per ridurre volume comunicazione
- Topologia a griglia per ridurre il costo di Allreduce