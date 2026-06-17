// Command to compile:
// mpicc mpi/pr_mpi_opt.c libraries/data.c libraries/measure.c -o mpi/pr_mpi_opt -Ilibraries -lm
// To run with 4 processes:
// mpirun -np 4 -machinefile mpi/machinefile.txt mpi/pr_mpi_opt
/*
================================================================================
# PAGERANK DISTRIBUITO (Pure MPI) - VERSIONE OTTIMIZZATA
# Architettura: I/O Centralizzato, Memoria Distribuita, Post-Processing Simmetrico
================================================================================
*/

/*
================================================================================
# CONFRONTO: VERSIONE MPI NATIVA vs MPI OTTIMIZZATA
================================================================================
Questo codice rappresenta una versione ottimizzata puramente MPI della versione base con MPI puro.
Dunque non abbiamo il multithreading.


## 1. I/O E IMPRONTA DI MEMORIA (Da Multi-Reader a Centralizzato)
- [NATIVA]: Tutti i processi (rank) aprivano e leggevano simultaneamente
  l'intero file dataset (generando un immenso overhead I/O su file system
  condivisi). Inoltre, ogni processo allocava e manteneva in RAM gli array
  globali completi ('val' e 'rowind', milioni di elementi).
- [OTTIMIZZATA]: Modello Master-Worker puro. Solo il MASTER accede al disco,
  esegue il parsing sequenziale e costruisce la CSC. Il Master distribuisce
  poi tramite 'MPI_Scatterv' SOLO i frammenti strettamente necessari a ogni
  processo. I worker usano una frazione infinitesima della RAM rispetto a prima.

## 2. COLLI DI BOTTIGLIA SULLA CPU (Post-Processing Seriale vs Simmetrico)
- [NATIVA]: Dopo la SpMV parallela, tutti i processi inviavano la loro somma parziale
  al MASTER ('MPI_Reduce'). Il Master, *completamente da solo e in modo seriale*,
  applicava il damping, sommava i pozzi e calcolava la norma su centinaia di
  migliaia di nodi, mentre gli altri processi rimanevano "congelati" in attesa.
- [OTTIMIZZATA]: Nessun processo resta in idle. Utilizzando 'MPI_Allreduce',
  tutti ottengono il vettore grezzo. Ognuno poi applica il damping, la
  redistribuzione della massa pozzo e calcola la norma (errore) ESCLUSIVAMENTE
  sulla propria porzione di colonne ('rec_col'). Carico perfettamente bilanciato.

## 3. ASINCRONIA (Overlap tra Calcolo e Comunicazione)
- [NATIVA]: Le operazioni erano strettamente sequenziali e bloccanti. Prima
  il calcolo, poi la rete ('MPI_Reduce'), poi l'attesa ('MPI_Bcast').
- [OTTIMIZZATA]: Viene introdotta l'istruzione asincrona 'MPI_Iallreduce'.
  Mentre le schede di rete gestiscono lo scambio e la somma dei milioni di
  voti del PageRank in background, le CPU dei processi elaborano la propria
  'dangling mass' locale, nascondendo efficacemente i tempi morti di rete.

## 4. SPRECO DI MEMORIA SU VARIABILI SCALARI
- [NATIVA]: Allocava enormi array come 'damp1', 'damp2' e 'diff' (ognuno da
  NODES elementi in double) solo per contenere valori costanti (es. 0.85
  ripetuto N volte, per esempio 685.000 volte) o delta(diff) temporanei.
- [OTTIMIZZATA]: Gli array inutili sono stati letteralmente rimossi. Sono
  stati sostituiti da costanti scalari ('DAMP1', 'DAMP2') calcolate a tempo
  di compilazione/avvio e variabili temporanee di ciclo, azzerando i "cache miss"
  della CPU e risparmiando decine di Megabyte di memoria preziosa.
================================================================================
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"


#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char *argv[])
{
    int NPROC, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    if (rank == MASTER) {
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK (PURE MPI - I/O Centralizzato)\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali : %d\n", NPROC);
        printf("=============================================\n\n");
        fflush(stdout);
    }

    int i, j, c;
    int colindex, link, colmatch = -1, localsum = 0;
    int co, index;

    // Strutture dati allocate solo dal Master per l'I/O
    double *val = NULL;
    int *rowind = NULL;

    // Strutture condivise (metadati di struttura del grafo)
    int *colptr = (int*)calloc(NODES + 1, sizeof(int));
    int *readsum = (int*)calloc(NODES, sizeof(int));

    // Vettori centrali del PageRank
    double *prold = (double*)malloc(NODES * sizeof(double));
    double *prnew = (double*)calloc(NODES, sizeof(double));
    double *local_sum = (double*)calloc(NODES, sizeof(double)); // Accumulatore locale

    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    // Vettori di utility per il partizionamento MPI
    int *sendcnts = malloc(sizeof(int) * NPROC);
    int *displs = malloc(sizeof(int) * NPROC);
    int *pcols = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_col;

    // INIZIALIZZAZIONE VETTORE PR
    for(i = 0; i < NODES; i++) {
       prold[i] = 1.0 / NODES;
    }

    // ========================================================================
    // 1. LETTURA FILE E COSTRUZIONE CSC (SOLO MASTER)
    // ========================================================================
    if (rank == MASTER) {
        FILE *fp = fopen(FILEPATH, "r");
        if (fp == NULL) {
           fprintf(stderr, "Rank %d - Errore: impossibile aprire il file '%s'\n", rank, FILEPATH);
           MPI_Abort(MPI_COMM_WORLD, 1);
        }

        val = (double*)calloc(EDGES, sizeof(double));
        rowind = (int*)calloc(EDGES, sizeof(int));

        localsum = 0;
        for(i = 0; i < EDGES; i++) {
           if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
              fprintf(stderr, "Rank %d - Errore lettura file\n", rank);
              fclose(fp);
              MPI_Abort(MPI_COMM_WORLD, 1);
           }
           colindex--; link--;
           rowind[i] = link;

           if (i == 0) {
               colmatch = colindex;
               localsum = 1;
           } else if (colmatch == colindex) {
               localsum++;
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

        // Normalizzazione Pesi CSC
        index = 0;
        for(i = 0; i < NODES; i++) {
           co = readsum[i];
           for(j = index; j < index + co; j++) val[j] /= co;
           index += co;
        }
        printf("CSC costruita e normalizzata --> Inizio Distribuzione Rete\n");
    }

    // ========================================================================
    // 2. DISTRIBUZIONE DATI GLOBALI (METADATI E PORZIONI CSC)
    // ========================================================================
    MPI_Bcast(colptr, NODES + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(readsum, NODES, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Calcolo delle colonne spettanti a ciascun processo
    // Ogni processo fa questa operazione, cioè si calcola la porzione di colonne che gli spettano
    // e che processerà poi.
    for(i = 0; i < NPROC; i++) {
       if(i == 0) {
          pcols[i] = NODES / NPROC + NODES % NPROC;
          displs_pr[i] = 0;
       } else {
          pcols[i] = NODES / NPROC;
          displs_pr[i] = pcols[i-1] + displs_pr[i-1];
       }
    }

    // Calcolo del numero di archi (elementi non-zero) per ogni processo
    j = 0;
    for(i = 0; i < NPROC; i++) {
       j += pcols[i];
       int k = j - pcols[i];
       sendcnts[i] = colptr[j] - colptr[k];
       if (i == 0) displs[i] = 0;
       else        displs[i] = sendcnts[i-1] + displs[i-1];
    }

    int my_cnt = sendcnts[rank];
    rec_col = pcols[rank];

    double *rec_val = (double*)malloc(my_cnt * sizeof(double));
    int *rec_row = (int*)malloc(my_cnt * sizeof(int));

    // Il master smista i pacchetti di matrice ai worker
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        free(val);
        free(rowind);
    }

    // ========================================================================
    // 3. CORE COMPUTAZIONALE (POWER ITERATION)
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();
    double norm = 0.0;

    // Variabili per il post-processing distribuito
    double dm_local, dm_global, norm_sq_local, norm_sq_global;

    do {
       memset(local_sum, 0, NODES * sizeof(double));
       dm_local = 0.0;
       norm_sq_local = 0.0;

       int global_col_start = displs_pr[rank];

       // --- FASE A: Prodotto Matrice-Vettore (SpMV) Locale ---
       // Ogni processo calcola l'impatto dei soli nodi che gestisce
       int local_col;
       for (local_col = 0; local_col < rec_col; local_col++) {
           int global_col = global_col_start + local_col;
           int start_idx = colptr[global_col] - displs[rank];
           int end_idx   = colptr[global_col + 1] - displs[rank];

           for (j = start_idx; j < end_idx; j++) {
               local_sum[rec_row[j]] += rec_val[j] * prold[global_col];
           }
       }

       // --- FASE B: Rete Asincrona e Overlap (Dangling Mass) ---
       MPI_Request request;
       // Lancio l'accumulo totale dei voti verso prnew in background
       MPI_Iallreduce(local_sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

       // Contemporaneamente calcolo la massa dispersa dai miei nodi pozzo
       for (local_col = 0; local_col < rec_col; local_col++) {
           int global_col = global_col_start + local_col;
           if (readsum[global_col] == 0) {
               dm_local += prold[global_col];
           }
       }

       // Sincronizzo la Dangling Mass e aspetto che arrivi tutto prnew
       MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       MPI_Wait(&request, MPI_STATUS_IGNORE);

       double redistribution = dm_global * DAMP1 / NODES;

       // --- FASE C: Post-Processing Distribuito Simmetrico ---
       // Ogni processo aggiorna *solo* le proprie colonne
       for (local_col = 0; local_col < rec_col; local_col++) {
           int global_col = global_col_start + local_col;

           prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

           double diff = prnew[global_col] - prold[global_col];
           norm_sq_local += diff * diff;

           prold[global_col] = prnew[global_col];
       }

       // --- FASE D: Valutazione Convergenza ---
       MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       norm = sqrt(norm_sq_global);

    } while(norm > ERROR);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;

    // ========================================================================
    // 4. VERIFICA MATEMATICA E CHIUSURA
    // Raccogliamo i pezzi finali di prold (pulito) sul Master
    // ========================================================================
    double *full_pr = (double*)malloc(NODES * sizeof(double));

    MPI_Allgatherv(prold + displs_pr[rank], pcols[rank], MPI_DOUBLE,
                   full_pr, pcols, displs_pr, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    if(rank == MASTER) {
        double sum_pr = 0;
        for(i = 0; i < NODES; i++) sum_pr += full_pr[i];

        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("Tempo puro di Iterazione: %f secondi\n", time_spent);
        printf("=============================================\n");

/*
     printf("\n--- VETTORE PAGERANK FINALE ---\n");
     for (i = 0; i < NODES; i++) {
         printf("Nodo %d: %.6f\n", i + 1, full_pr[i]);
     }
     printf("=============================================\n");
*/
    }



    // Liberazione Memoria
    free(full_pr);
    free(colptr); free(readsum);
    free(prold); free(prnew); free(local_sum);
    free(sendcnts); free(displs); free(pcols); free(displs_pr);
    free(rec_val); free(rec_row);

    MPI_Finalize();
    return 0;
}