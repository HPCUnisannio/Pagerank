// Command to compile:
// mpicc mpi_openMP/pr_mpi_openmp.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp -Ilibraries -fopenmp -lm
// To run with 4 processes:
// mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp 2

/*
 * CONFIGURAZIONE CON REGIONE PARALLALE A GRANA GROSSA
 * POOL DI THREAD CREATO UNA SOLA VOLTA FUORI DA DO-WHILE
 * POST PROCESSING FATTO FARE ESCLUSIVAMENTE AL PROCESSO MASTER
 * RIMOZIONE DI atomic E PRIVATIZZAZIONE DEL VETTORE sum
 *
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char **argv)
{
    int NPROC, rank, num_threads;
    // Se l'utente passa un argomento extra, lo usiamo come numero di thread
    if (argc > 1)
    {
        num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    }
    else
    {
        num_threads = omp_get_max_threads();
    }

    int mpi_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    // 2. Stampa delle informazioni di setup (Solo MASTER)
    if (rank == 0)
    { // Usa 'MASTER' se hai definito una macro per lo 0
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK IBRIDO [ MPI + OpenMP(grana grossa + privatizzazione + master thread) ]\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf("=============================================\n\n");
        // QUESTA RIGA FORZA LA STAMPA IMMEDIATA A SCHERMO
        fflush(stdout);
    }

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph *graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char *FILEPATH = graph->filepath;

    FILE *fp;
    int colindex, link, i, j = 0, col, c, colmatch = -1, localsum = 0;
    int co, index;

    // Allocazione strutture principali
    double *val = (double *)calloc(EDGES, sizeof(double));
    int *rowind = (int *)calloc(EDGES, sizeof(int));
    int *colptr = (int *)calloc(NODES + 1, sizeof(int));
    int *readsum = (int *)calloc(NODES, sizeof(int));

    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)calloc(NODES, sizeof(double));

    // double * damp1 = (double*)malloc(NODES * sizeof(double));
    // double * damp2 = (double*)malloc(NODES * sizeof(double));
    // double * diff = (double*)calloc(NODES, sizeof(double));
    // Al posto delle 2 malloc e del loop di inizializzazione
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    double *sum = (double *)calloc(NODES, sizeof(double));

    int *sendcnts = malloc(sizeof(int) * NPROC);
    int *displs = malloc(sizeof(int) * NPROC);
    int *pcols = (int *)malloc(NPROC * sizeof(int));
    int *displs_pr = (int *)malloc(NPROC * sizeof(int));

    double norm;
    int rec_col;

    // Inizializzazione vettori
    for (i = 0; i < NODES; i++)
    {
        prold[i] = 1.0 / NODES;
        // damp1[i] = DAMPING;
        // damp2[i] = (1.0 - DAMPING) / NODES;
    }

    if (rank == MASTER)
    {
        printf("Initialization complete\n");
    }

    // Lettura file
    fp = fopen(FILEPATH, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Rank %d - Errore: impossibile aprire il file '%s'\n", rank, FILEPATH);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- FIX 1: Lettura robusta a prova di "Dangling Nodes" ---
    localsum = 0;
    for (i = 0; i < EDGES; i++)
    {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2)
        {
            fprintf(stderr, "Rank %d - Errore lettura file alla riga %d\n", rank, i + 1);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        colindex = colindex - 1;
        link = link - 1;
        rowind[i] = link;

        if (i == 0)
        {
            colmatch = colindex;
            localsum = 1;
        }
        else if (colmatch == colindex)
        {
            localsum += 1;
        }
        else
        {
            readsum[colmatch] = localsum;
            for (c = colmatch + 1; c <= colindex; c++)
            {
                colptr[c] = colptr[colmatch] + localsum;
            }
            localsum = 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }
    if (EDGES > 0)
    {
        readsum[colmatch] = localsum;
        for (c = colmatch + 1; c <= NODES; c++)
        {
            colptr[c] = EDGES;
        }
    }
    fclose(fp);

    // Normalizzazione della matrice CSC
    index = 0;
    for (i = 0; i < NODES; i++)
    {
        co = readsum[i];
        for (j = index; j < index + co; j++)
        {
            val[j] = val[j] / co;
        }
        index += co;
    }

    if (rank == MASTER)
    {
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
    for (i = 0; i < NPROC; i++)
    {
        if (i == 0)
        {
            pcols[i] = NODES / NPROC + NODES % NPROC;
            displs_pr[i] = 0;
        }
        else
        {
            pcols[i] = NODES / NPROC;
            displs_pr[i] = pcols[i - 1] + displs_pr[i - 1];
        }
    }

    // Calcolo dei conteggi di elementi non-zero (sendcnts) e relativi spiazzamenti (displs)
    j = 0;
    for (i = 0; i < NPROC; i++)
    {
        j = j + pcols[i];
        int k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0)
        {
            displs[i] = 0;
        }
        else
            displs[i] = sendcnts[i - 1] + displs[i - 1];
    }

    // Allocazione dei buffer di ricezione locali per lo Scatterv
    double *rec_val = (double *)malloc(sendcnts[rank] * sizeof(double));
    int *rec_row = (int *)malloc(sendcnts[rank] * sizeof(int));
    double *rec_pr = (double *)malloc(pcols[rank] * sizeof(double));

    // Distribuzione dei dati
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE,
                 rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT,
                &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT,
                 rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    // double redistribution = 0.0;

    double dangling_mass = 0.0;
    double norm_sq = 0.0;

    /*
     * VARIABILI PER MISURARE IL TEMPO DI COMUNICAZIONE E DI CALCOLO
     */
    double t_scatter = 0.0, t_spmv = 0.0, t_reduce_sum = 0.0;
    double t_reduce_mpi = 0.0, t_postproc = 0.0, t_bcast = 0.0;

    // --- CONFIGURAZIONE STRUTTURE PER PRIVATIZZAZIONE OPENMP (No MPI) ---
    // Creiamo un array di puntatori condiviso: conterrà il vettore sum di ogni thread
    // int actual_threads = num_threads;
    double **thread_sums = (double **)calloc(num_threads, sizeof(double *));
    // ============================================================================================================
    // REGIONE PARALLELA A GRANA GROSSA --> CREIAMO QUI UNA SOLA VOLTA IL POOL DI THREAD
    // ============================================================================================================
    #pragma omp parallel private(i, j)
    {

        // Ogni thread alloca una volta sola il proprio vettore sum privato sulla Heap
        int tid = omp_get_thread_num();
        thread_sums[tid] = (double *)calloc(NODES, sizeof(double));
        double *local_sum = thread_sums[tid];

        // ============================================================================================================
        // REGIONE PARALLELA A GRANA GROSSA (Versione Corretta e Standard-Compliant)
        // ============================================================================================================
        do
        {
            /* LOGICA COMUNICAZIONE MPI
            1) In testa al ciclo (MPI_Scatterv):
            Il Thread 0 del Processo 0 prende i dati e li spedisce.
            I Thread 0 dei processi Worker (Rank 1, 2, 3...) si mettono in ascolto e li ricevono.
            I thread 1, 2, 3 di tutti i processi non fanno nulla riguardo la rete, aspettano e basta alla barriera.

            2) A metà ciclo (MPI_Reduce):
            I Thread 0 dei processi Worker prendono il vettore sum del loro processo e lo spediscono via rete.
            Il Thread 0 del Processo 0 si mette in ricezione, raccoglie i dati dei Worker e li somma dentro prnew.

            3) Alla fine del ciclo (MPI_Bcast):
            Il Thread 0 del Processo 0 (che ha appena finito di calcolare la radice quadrata) spedisce il valore norm.
            I Thread 0 dei Worker lo ricevono, in modo che tutti i processi abbiano lo stesso valore per valutare il while(norm > ERROR).
       */

            memset(local_sum, 0, NODES * sizeof(double));

            // SICUREZZA MPI: Solo il thread MASTER di ogni processo avvia la comunicazione di rete
            #pragma omp master
            {
                if (rank == MASTER)
                {
                    memset(prnew, 0, NODES * sizeof(double));
                    dangling_mass = 0.0;
                    norm_sq = 0.0;
                }
                norm = 0.0;

                double t0 = MPI_Wtime();
                MPI_Scatterv(prold, pcols, displs_pr, MPI_DOUBLE,
                             rec_pr, pcols[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                t_scatter += MPI_Wtime() - t0;
            }
            // [BARRIERA 1 Esplicita -> FONDAMENTALE]
            // I thread dei nodi worker devono aspettare che il thread master abbia ricevuto i dati via MPI
            // prima di procedere al calcolo.
            #pragma omp barrier

            double t1 = MPI_Wtime(); // Misurazione tempo di calcolo e riduzione

            // --- CALCOLO PARALLELO ---
            int global_col_start = (rank == MASTER) ? 0 : displs_pr[rank];

            int local_col;
            #pragma omp for schedule(dynamic, 512)
            for (local_col = 0; local_col < rec_col; local_col++)
            {
                int global_col = global_col_start + local_col;
                int start_idx = colptr[global_col] - displs[rank];
                int end_idx = colptr[global_col + 1] - displs[rank];

                for (j = start_idx; j < end_idx; j++)
                {
                    local_sum[rec_row[j]] += rec_val[j] * rec_pr[local_col];
                }
            } // [BARRIERA 2 Implicita -> FONDAMENTALE]
            // I thread devono aspettare che tutti abbiano finito di scrivere nei propri local_sum
            // prima di procedere alla riduzione

            // --- RIDUZIONE PARALLELA DEI THREAD ---
            #pragma omp for
            for (i = 0; i < NODES; i++)
            {
                double total_row_sum = 0.0;
                int t;
                for (t = 0; t < num_threads; t++)
                {
                    if (thread_sums[t] != NULL)
                        total_row_sum += thread_sums[t][i];
                }
                sum[i] = total_row_sum;
            } // [BARRIERA 3 Implicita]

            t_spmv += MPI_Wtime() - t1;

            // SICUREZZA MPI: Solo il thread master di ciascun processo esegue la riduzione collettiva
            #pragma omp master
            {
                double t2 = MPI_Wtime();
                MPI_Reduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
                t_reduce_mpi += MPI_Wtime() - t2;
            }
            #pragma omp barrier // [BARRIERA 4 Esplicita]
            // Tutti i thread devo aspettare che la Reduce sia completata

            // ============================================================================================================
            // FASE DI POST PROCESSING CENTRALIZZATO --> Solo sul Processo MASTER
            // ============================================================================================================
            if (rank == MASTER)
            {
                double t_post = MPI_Wtime(); // Inizio misurazione Post processing centralizzato

                #pragma omp for reduction(+ : dangling_mass)
                for (i = 0; i < NODES; i++)
                {
                    if (readsum[i] == 0)
                    {
                        dangling_mass += prold[i];
                    }
                } // [BARRIERA 5 Implicita]
                // Tutti i thread del master devono aspettare di aver finito di calcolare la massa dei dangling nodes
                /*
                                #pragma omp master
                                {
                                    redistribution = dangling_mass * DAMPING / NODES;
                                }
                                #pragma omp barrier // Aspetta il calcolo di redistribution
                */
                double redistribution = dangling_mass * DAMPING / NODES;
                // FUSIONE DEI DUE CICLI
                /*
                #pragma omp for
                for(i = 0; i < NODES; i++) {
                    prnew[i] = prnew[i] * damp1[i] + damp2[i] + redistribution;
                }*/

#pragma omp for reduction(+ : norm_sq)
                for (i = 0; i < NODES; i++)
                {
                    prnew[i] = prnew[i] * DAMP1 + DAMP2 + redistribution; // aggiorna prnew in-place
                    double diff = prnew[i] - prold[i];                    // diff sul valore dampato
                    norm_sq += diff * diff;
                    prold[i] = prnew[i];
                    // accorpando i cicli abbiamo una unica barriera qui
                } //[BARRIERA 6]

                t_postproc += MPI_Wtime() - t_post; // Fine misurazione post processing
            }

            // I thread secondari del MASTER che passano di qui vanno dritti alla barriera finale.
            // Il Thread 0 dei Worker si ferma nella Bcast. Il Thread 0 del Master fa la Bcast e li sblocca.
            #pragma omp master
            {
                if (rank == MASTER) // in questa variante facciamo fare l'aggiornamento della norma al processo MASTER
                {
                    norm = sqrt(norm_sq);
                }
                double t3 = MPI_Wtime();
                MPI_Bcast(&norm, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                t_bcast += MPI_Wtime() - t3;
            }
            #pragma omp barrier //[BARRIERA 7 Esplicita]
            // Tutti i thread aspettano che il proprio thread master riceva la nuova 'norm' per poter
            // ripartire con l'iterazione successiva di calcolo(con norm aggiornata)

        } while (norm > ERROR);
        #pragma omp master
        if (rank == MASTER)
        {
            printf("\n--- PROFILO TEMPO (totale su tutte le iterazioni) ---\n");
            printf("  MPI_Scatterv  : %.4f s\n", t_scatter);
            printf("  SpMV+riduzione: %.4f s\n", t_spmv);
            printf("  MPI_Reduce    : %.4f s\n", t_reduce_mpi);
            printf("  Post-proc     : %.4f s\n", t_postproc);
            printf("  MPI_Bcast     : %.4f s\n", t_bcast);
        }

        free(local_sum);
    }
    free(thread_sums);
    // ==============================================================================================================
    // FINE REGIONE PARALLELA A GRANA GROSSA
    // ==============================================================================================================

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;

    if (rank == MASTER)
    {
        double sum_pr = 0;
        for (i = 0; i < NODES; i++)
            sum_pr += prnew[i];

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
    free(val);
    free(rowind);
    free(colptr);
    free(readsum);
    free(prold);
    free(prnew);
    free(sum);
    // free(damp1); free(damp2); free(diff);

    free(sendcnts);
    free(displs);
    free(pcols);
    free(displs_pr);
    free(rec_val);
    free(rec_row);
    free(rec_pr);

    MPI_Finalize();
    return 0;
}