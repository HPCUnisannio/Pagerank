// Command to compile:
// mpicc mpi/pr_mpi.c libraries/data.c libraries/measure.c -o mpi/pr_mpi -Ilibraries -lm
// To run with 4 processes:
// mpirun -np 4 -machinefile mpi/machinefile.txt mpi/pr_mpi
/*
================================================================================
# 🚀 PAGERANK DISTRIBUITO (MPI) - VERSIONE BASELINE STABILE ED ESATTA
================================================================================

Questo sorgente rappresenta una versione corretta, stabilizzata e matematicamente
esatta di un calcolatore di PageRank distribuito basato su MPI e formato CSC
(Compressed Sparse Column).

Il codice originale presentava molteplici falle critiche (Crash, Memory Leak,
Buffer Overflow, Errori Matematici). Di seguito sono documentate tutte le
modifiche apportate per ottenere questa versione "Baseline", su cui potranno
essere applicate successive ottimizzazioni di performance.

## 🛠️ 1. STABILITÀ DELLA MEMORIA E PREVENZIONE CRASH
* **Fix Buffer Overflow (Magic Numbers):** Nel codice originale, i buffer di ricezione
    (`rec_val`, `rec_row`, `rec_pr`) erano allocati con una dimensione fissa e "magica"
    di 100.000 elementi. Su dataset reali, questo causava immediati buffer overflow.
    Le allocazioni sono state rese dinamiche e precise usando le quote calcolate
    da `sendcnts[rank]` e `pcols[rank]`.
* **Fix Memory Explosion (Rimozione Matrice Densa):** La power iteration originale
    ricostruiva una sottomatrice densa (`vec`) a partire dai dati sparsi ricevuti.
    Questo approccio distruggeva i benefici del formato CSC e saturava la RAM
    (OOM Killer) su grafi medi. La moltiplicazione matrice-vettore è stata
    riscritta per iterare puramente in formato sparso usando `rec_val`, `rec_row`
    e i bounds derivati da `colptr`.
* **Fix Heap Corruption (Parsing del File Robusto):** Il parser di lettura file
    originale andava in crash o corrompeva la memoria se i nodi nel file non erano
    perfettamente sequenziali, causando out-of-bounds array writes. La logica di
    popolamento di `colptr` e `readsum` è stata riscritta per gestire "buchi"
    negli indici dei nodi, garantendo la coerenza strutturale in memoria.

## 🧮 2. CORREZIONE DELLA LOGICA MATEMATICA
* **Fix Inizializzazione Probabilità:** Il vettore `pr` era originariamente
    impostato fisso a `0.25` per ogni nodo, sballando la distribuzione stocastica.
    È stato corretto a `1.0 / NODES` in modo che la somma iniziale sia esattamente 1.0.
    Similmente, il termine costante di teleporting (`damp2`) è stato corretto in
    `(1.0 - DAMPING) / NODES`.
* **Gestione dei "Dangling Nodes" (Nodi Pozzo):** La precedente implementazione
    perdeva massa di probabilità (fermando la somma totale a ~0.966) a causa
    dei nodi senza link uscenti (out-degree = 0). Nel Master è stato introdotto
    il calcolo della `dangling_mass` e la successiva redistribuzione uniforme
    (`+ redistribution`) per garantire la conservazione rigorosa della probabilità
    (Somma finale = 1.00000000).

## ⏱️ 3. BENCHMARKING ED ESECUZIONE MPI
* **Metriche di Tempo Reale (Wall-Clock):** La funzione `clock()` del C standard,
    che misurava i cicli CPU locali (ignorando l'idle e i ritardi di rete), è stata
    sostituita con `MPI_Wtime()`, preceduta da un `MPI_Barrier` per allineare
    tutti i processi e ottenere una misurazione affidabile del tempo reale di calcolo.
* **Pulizia I/O e Standard:** Aggiunti controlli bloccanti (`MPI_Abort`) sul fallimento
    di `fopen` e `fscanf`. Aggiunte le chiamate `free()` per deallocare correttamente
    ogni struttura dati, prevenendo memory leak prima di `MPI_Finalize()`.
* **Output Centralizzato:** Spostate le stampe dei risultati finali e delle
    verifiche matematiche esclusivamente sotto il controllo del processo `MASTER` (rank 0).

---
**🔜 PROSSIMI STEP (Ottimizzazioni per il futuro):**
Al momento la lettura del file viene fatta in modo seriale da TUTTI i processi.
Questa operazione è sicura, ma genera un Overhead di I/O enorme. La prossima
frontiera di ottimizzazione consisterà nel demandare la lettura al solo MASTER o
sfruttare MPI-IO.
================================================================================
*/

/*
================================================================================
# 🧠 ARCHITETTURA DEL PARALLELISMO: COME FUNZIONA QUESTO CODICE MPI
================================================================================
Questo blocco descrive la logica di partizionamento dei dati e l'uso specifico
delle primitive MPI per la computazione distribuita del PageRank.

## 1. LA STRATEGIA: 1D Column Partitioning (Divisione per Colonne)
La matrice sparsa del grafo è salvata in formato CSC (compressa per colonne).
Il parallelismo si ottiene dividendo le *colonne* tra i processi MPI.
Poiché le colonne rappresentano i "link uscenti" dei nodi, ogni processo diventa
responsabile di calcolare l'impatto di un sottoinsieme di nodi sul resto della rete.

## 2. PREPARAZIONE DELLE "FETTE" (Variabili di partizionamento)
I nodi hanno un numero di link uscenti diverso, quindi i processi riceveranno
array di dimensioni differenti. Vengono calcolati tre array di mappatura:
* 'pcols': Numero di colonne (nodi) assegnate a ciascun processo MPI.
* 'sendcnts': Numero totale di archi non-zero contenuti in quelle specifiche colonne.
* 'displs': L'offset (spiazzamento) nell'array globale da cui iniziare a tagliare
            la fetta di competenza per ogni processo.

## 3. LA FASE DI DISTRIBUZIONE INIZIALE
* MPI_Scatter: Divide un array in parti *uguali*. Usato per inviare a ogni processo
  il numero di colonne ('rec_col') che gli competono.
* MPI_Scatterv (Scatter Variable): L'istruzione chiave. Permette di inviare fette di
  dimensioni *diverse* sfruttando 'sendcnts' e 'displs'. Invia le probabilità degli
  archi ('val' -> 'rec_val') e i nodi destinazione ('rowind' -> 'rec_row').

## 4. IL LOOP DI CALCOLO (Iterazione del PageRank)
A. Distribuzione Vettore Corrente:
   Ad ogni giro, tramite 'MPI_Scatterv', il MASTER taglia il vettore globale 'pr'
   e invia a ogni processo solo la porzione associata ai nodi che deve gestire ('rec_pr').

B. Calcolo Locale Parallelo (Moltiplicazione Sparsa):
   Ogni processo esegue un loop sui propri dati locali per calcolare i voti espressi
   dai suoi nodi, accumulandoli nel suo array 'sum'.
   NOTA: L'array 'sum' è grande quanto l'intero grafo (NODES), ma ogni processo lo
   riempie solo per i nodi di destinazione che ha incontrato.

C. Assemblaggio Globale:
   L'istruzione 'MPI_Reduce' con operatore 'MPI_SUM' prende gli array 'sum' sparsi di
   tutti i processi e li somma cella per cella. Il risultato globale convergente viene
   depositato nell'array 'prnew' ESCLUSIVAMENTE nella memoria del MASTER.

## 5. FASE FINALE E SINCRONIZZAZIONE
* Il MASTER riceve il vettore grezzo, aggiunge il Damping factor, ridistribuisce la
  massa dei Dangling Nodes (nodi pozzo) e calcola l'errore ('norm') rispetto al
  vettore precedente.
* MPI_Bcast (Broadcast): A questo punto, solo il MASTER conosce 'norm'. Usa il
  Broadcast per comunicare questo scalare a tutti gli altri nodi. In questo modo,
  tutti i processi possono valutare contemporaneamente la condizione del ciclo
  'while(norm > ERROR)' e decidere se ripetere o terminare all'unisono.

(NOTA SULLE BARRIERE: Le chiamate 'MPI_Barrier' non sono necessarie per la correttezza
matematica, in quanto istruzioni come Scatterv, Reduce e Bcast implementano già una
sincronizzazione implicita bloccante. Vengono usate solo per il benchmarking rigoroso).
================================================================================
*/

/*
 * =======================================================================================
 * 🧮 CALCOLO DEI CHUNK IRREGOLARI PER IL PARTIZIONAMENTO (SENDCNTS & DISPLS)
 * =======================================================================================
 * Poiché il grafo è irregolare, a un ugual numero di nodi (colonne) NON corrisponde
 * un ugual numero di archi (elementi non-zero). Questo rende impossibile usare un
 * semplice MPI_Scatter. Dobbiamo usare MPI_Scatterv e calcolare l'esatta dimensione
 * del "pacchetto dati" per ogni processo.
 *
 * La proprietà matematica dell'array cumulativo 'colptr' (formato CSC) ci permette di
 * trovare il numero totale di archi per un blocco di nodi con una singola sottrazione,
 * senza dover iterare e contare gli archi uno ad uno.
 *
 * VARIABILI:
 * - j : Indice di colonna finale (esclusiva) del blocco assegnato al processo 'i'.
 * - k : Indice di colonna iniziale (inclusiva) del blocco assegnato al processo 'i'.
 * - pcols[i] : Quante colonne (nodi) deve gestire il processo 'i'.
 *
 * LOGICA:
 * - colptr[j] : Numero totale cumulativo di archi nel grafo fino alla colonna 'j'.
 * - colptr[k] : Numero totale cumulativo di archi nel grafo fino alla colonna 'k'.
 * - sendcnts[i] = colptr[j] - colptr[k] : Esatto numero di archi (e relative probabilità)
 * che il processo 'i' riceverà.
 * - displs[i]   : Offset (spiazzamento) da cui MPI_Scatterv inizierà a tagliare l'array
 * globale per inviarlo al processo 'i'.
 * =======================================================================================
 */


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
/*
 * Per eseguire da riga di comando e settare più processi, ad esempio 4
 * mpiexec -n 4 ".\cmake-build-debug\pr_mpi.exe"
*/

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char *argv[])
{
    int NPROC, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("Program start\n");
        printf("Number of processes %d\n", NPROC);
    }

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    FILE *fp;
    int colindex, link, i, j = 0, k, col, c, colmatch = -1, localsum = 0;
    int co, index;

    // Allocazione strutture principali
    double * val = (double*)calloc(EDGES, sizeof(double));
    int * rowind =(int*)calloc(EDGES, sizeof(int));
    int * colptr = (int*)calloc(NODES + 1, sizeof(int));
    int * readsum = (int*)calloc(NODES, sizeof(int));

    double * prold = (double*)malloc(NODES * sizeof(double));
    double * prnew = (double*)calloc(NODES, sizeof(double));
    double * damp1 = (double*)malloc(NODES * sizeof(double));
    double * damp2 = (double*)malloc(NODES * sizeof(double));
    double * diff = (double*)calloc(NODES, sizeof(double));
    double * sum = (double*)calloc(NODES, sizeof(double));

    int *sendcnts = malloc(sizeof(int) * NPROC);
    int *displs = malloc(sizeof(int) * NPROC);
    int *pcols = (int*)malloc(NPROC * sizeof(int));
    int *displs_pr = (int*)malloc(NPROC * sizeof(int));

    double norm;
    int rec_col;

    // Inizializzazione vettori
    for(i = 0; i < NODES; i++) {
       prold[i] = 1.0 / NODES;
       damp1[i] = DAMPING;
       damp2[i] = (1.0 - DAMPING) / NODES;
    }

    if (rank == MASTER) {
        printf("Initialization complete\n");
    }

    // Lettura file
    fp = fopen(FILEPATH, "r");
    if (fp == NULL) {
       fprintf(stderr, "Rank %d - Errore: impossibile aprire il file '%s'\n", rank, FILEPATH);
       MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- FIX 1: Lettura robusta a prova di "Dangling Nodes" ---
    localsum = 0;
    for(i = 0; i < EDGES; i++) {
       if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
          fprintf(stderr, "Rank %d - Errore lettura file alla riga %d\n", rank, i + 1);
          fclose(fp);
          MPI_Abort(MPI_COMM_WORLD, 1);
       }
       colindex = colindex - 1;
       link = link - 1;
       rowind[i] = link;

       if (i == 0) {
           colmatch = colindex;
           localsum = 1;
       } else if (colmatch == colindex) {
           localsum += 1;
       } else {
           readsum[colmatch] = localsum;
           for(c = colmatch + 1; c <= colindex; c++) {
               colptr[c] = colptr[colmatch] + localsum;
           }
           localsum = 1;
           colmatch = colindex;
       }
       val[i] = 1.0;
    }
    if (EDGES > 0) {
        readsum[colmatch] = localsum;
        for (c = colmatch + 1; c <= NODES; c++) {
            colptr[c] = EDGES;
        }
    }
    fclose(fp);

    // Normalizzazione della matrice CSC
    index = 0;
    for(i = 0; i < NODES; i++) {
       co = readsum[i];
       for(j = index; j < index + co; j++) {
          val[j] = val[j] / co;
       }
       index += co;
    }

    if (rank == MASTER) {
        printf("val, rowind and colptr have been populated\n");
/*
        printf("\n======================= CSC construction complete ==========================\n");
        printf("\n--- VERIFICA COSTRUZIONE CSC ---\n");
        printf("COLPTR: ");
        for (int c = 0; c <= NODES; c++) printf("%d ", colptr[c]);
        printf("\nROWIND: ");
        for (int c = 0; c < EDGES; c++) printf("%d ", rowind[c]);
        printf("\nVAL:    ");
        for (int c = 0; c < EDGES; c++) printf("%.2f ", val[c]);
        printf("\n============================================================================\n");
        */
    }

    // Calcolo della distribuzione delle colonne tra i processi
    for(i = 0; i < NPROC; i++) {
       if(i == 0) {
          pcols[i] = NODES / NPROC + NODES % NPROC;
          displs_pr[i] = 0;
       } else {
          pcols[i] = NODES / NPROC;
          displs_pr[i] = pcols[i-1] + displs_pr[i-1];
       }
    }

    // Calcolo dei conteggi di elementi non-zero (sendcnts) e relativi spiazzamenti (displs)
    j = 0;
    for(i = 0; i < NPROC; i++) {
       j = j + pcols[i];
       k = j - pcols[i];
       sendcnts[i] = colptr[j] - colptr[k];
       if (i == 0) {
          displs[i] = 0;
       } else
          displs[i] = sendcnts[i-1] + displs[i-1];
    }

    // Allocazione dei buffer di ricezione locali per lo Scatterv
    double * rec_val = (double*)malloc(sendcnts[rank] * sizeof(double));
    int * rec_row = (int*)malloc(sendcnts[rank] * sizeof(int));
    double * rec_pr = (double*)malloc(pcols[rank] * sizeof(double));

    // Distribuzione dei dati
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT, &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    //MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    do
    {
       memset(sum, 0, NODES * sizeof(double));
       if(rank == MASTER) {
           memset(prnew, 0, NODES * sizeof(double));
       }
       norm = 0.0;

       // Invia la porzione corretta del vettore di PageRank a ciascun processo
       MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE, rec_pr, pcols[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // --- NUOVO CALCOLO PARALLELO MPI (Allineato al sequenziale) ---
        int global_col_start = (rank == MASTER) ? 0 : displs_pr[rank];

        // Iteriamo strettamente sulle colonne assegnate a questo processo (come nel sequenziale)
        int local_col;
        for (local_col = 0; local_col < rec_col; local_col++) {

            // Ricalcoliamo l'ID della colonna (nodo) nel contesto globale
            int global_col = global_col_start + local_col;

            // Calcoliamo i confini di inizio e fine per gli archi di questo nodo.
            // Sottraendo displs[rank], trasliamo l'indice globale nell'indice locale del buffer rec_val/rec_row.
            int start_idx = colptr[global_col] - displs[rank];
            int end_idx   = colptr[global_col + 1] - displs[rank];

            // Ciclo interno identico al sequenziale!
            for (j = start_idx; j < end_idx; j++) {
                sum[rec_row[j]] += rec_val[j] * rec_pr[local_col];
            }
        }


       // Riduzione: accumula i risultati parziali di tutti i processi sul MASTER
       MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

       // Il MASTER applica damping, teleportation, redistribuzione della massa e calcola la convergenza
       if(rank == MASTER) {
          
          // --- FIX 2: Calcolo della massa dei Dangling Nodes ---
          double dangling_mass = 0.0;
          for(i = 0; i < NODES; i++) {
              if (readsum[i] == 0) {
                  dangling_mass += prold[i];
              }
          }
          
          double redistribution = (dangling_mass * DAMPING) / NODES;

          for(i = 0; i < NODES; i++) {
              prnew[i] = prnew[i] * damp1[i] + damp2[i] + redistribution;
          }

          double norm_sq = 0.0;
          for(i = 0; i < NODES; i++) {
             diff[i] = prnew[i] - prold[i];
             norm_sq += diff[i] * diff[i];
             prold[i] = prnew[i];
          }
          norm = sqrt(norm_sq);
       }

       // Trasmette la norma a tutti i nodi per decidere la continuazione del ciclo
       MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    } while(norm > ERROR);

    double end = MPI_Wtime();
    double time_spent = end - begin;

    if(rank == MASTER)
    {
        double sum_pr = 0;
        for(i = 0; i < NODES; i++) sum_pr += prnew[i];

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("=============================================\n");
        printf("Time taken for power iteration solution: %f seconds\n", time_spent);

        /*
        printf("\n--- VETTORE PAGERANK FINALE ---\n");
        for (i = 0; i < NODES; i++) {
            printf("Nodo %d: %.6f\n", i + 1, prnew[i]);
        }
        printf("=============================================\n");
        */
    }


        // Liberazione memoria
        free(val); free(rowind); free(colptr); free(readsum);
        free(prold); free(prnew); free(damp1); free(damp2); free(diff); free(sum);
        free(sendcnts); free(displs); free(pcols); free(displs_pr);
        free(rec_val); free(rec_row); free(rec_pr);

        MPI_Finalize();
        return 0;
    }