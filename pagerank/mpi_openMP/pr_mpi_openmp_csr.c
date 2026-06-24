#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char **argv)
{
    GraphType graph_type = GRAPH_MEDIUM;
    const Graph *graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char *FILEPATH = graph->filepath;

    int NPROC, rank, num_threads;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    // Inizializzazione MPI con supporto Funneled
    int mpi_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("\n======================================================\n");
        printf(" AVVIO PAGERANK IBRIDO OPT (CSR - Lettura Centralizzata)\n");
        printf("======================================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf(" -> Struttura Dati         : Compressed Sparse Row (CSR)\n");
        printf("======================================================\n\n");
        fflush(stdout);
    }

    int j;

    // Strutture dati globali della matrice (Allocate solo dal MASTER)
    double *val = NULL;
    int *colind = NULL;

    // Allocazione contigua dei metadati strutturali (Ottimizzazione Bcast)
    int *metadata_buffer = (int *)calloc((NODES + 1) + NODES, sizeof(int));
    int *rowptr     = metadata_buffer;
    int *out_degree = metadata_buffer + (NODES + 1);

    // Vettori PageRank (Dimensione globale su tutti i nodi)
    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)calloc(NODES, sizeof(double));

    // Costanti scalari per il Damping
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    // Array per il partizionamento delle RIGHE tra processi MPI
    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *prows     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_row;

    // Inizializzazione uniforme del vettore PageRank
    pagerank_init_vector(NODES, prold);

    // ========================================================================
    // 1. LETTURA FILE E COSTRUZIONE MATRICE CSR (SOLO MASTER)
    // ========================================================================
    // Variabili per la misurazione dei tempi
    double setup_start = 0.0, setup_end = 0.0;
    double compute_start = 0.0, compute_end = 0.0;
    double setup_time = 0.0, compute_time = 0.0, total_time = 0.0;
    if (rank == MASTER) {
        setup_start = MPI_Wtime();
    }

    if (rank == MASTER) {
        val    = (double *)calloc(EDGES, sizeof(double));
        colind = (int *)calloc(EDGES, sizeof(int));

        if (csr_build_from_file(FILEPATH, NODES, EDGES, val, colind, rowptr, out_degree) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("CSR costruita e normalizzata centralmente --> Distribuzione dati\n");
        fflush(stdout);
    }

    // ========================================================================
    // 2. DISTRIBUZIONE DEI METADATI E DELLE PORZIONI CSR VIA MPI
    // ========================================================================
    // Broadcast ACCORPATO dei metadati strutturali (Unica chiamata di rete)
    MPI_Bcast(metadata_buffer, (2 * NODES + 1), MPI_INT, MASTER, MPI_COMM_WORLD);

    // Calcolo della distribuzione bilanciata delle righe del grafo
    compute_column_distribution(NODES, NPROC, prows, displs_pr);
    compute_nnz_distribution(NPROC, prows, rowptr, sendcnts, displs);

    int my_cnt = sendcnts[rank];
    rec_row = prows[rank];

    // Allocazione delle strutture locali per la porzione CSR assegnata al processo
    double *rec_val    = (double *)malloc(my_cnt * sizeof(double));
    int    *rec_colind = (int *)malloc(my_cnt * sizeof(int));

    // Distribuzione dei vettori della matrice tramite Scatterv
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(colind, sendcnts, displs, MPI_INT, rec_colind, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);


    if (rank == MASTER) {
        free(val);
        free(colind);

        setup_end = MPI_Wtime();                     // timestamp di fine setup
        setup_time = setup_end - setup_start;        // durata effettiva del setup
        printf("Setup completato in %.6f secondi.\n", setup_time);
        printf("Inizio computazione parallela iterativa...\n");
        fflush(stdout);
    }

    // ========================================================================
    // 3. REGIONE COMPUTAZIONALE IBRIDA (POWER ITERATION CORE)
    // ========================================================================
    // Sincronizzazione: tutti i processi sono pronti per la computazione
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        compute_start = MPI_Wtime();   // inizio della computazione (dopo la barriera)
    }


    double norm = 0.0;
    double dm_local = 0.0, dm_global = 0.0, norm_sq_local = 0.0;
    int iteration_count = 0;

    int global_row_start = displs_pr[rank];
    int global_row_end   = global_row_start + rec_row;

    // Bootstrap iniziale fuori dal loop per la Delayed Dangling Mass
    int i;
    for (i = global_row_start; i < global_row_end; i++) {
        if (out_degree[i] == 0) {
            dm_local += prold[i];
        }
    }
    MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // APERTURA REGIONE PARALLELA OPENMP
    #pragma omp parallel
    {
        int r, j;
        do {
            // Fase A: SpMV parallela su righe CSR (Nessun conflitto di scrittura)
            #pragma omp for schedule(dynamic, 64)
            for (r = global_row_start; r < global_row_end; r++) {
                int start_idx = rowptr[r] - displs[rank];
                int end_idx   = rowptr[r + 1] - displs[rank];
                double row_accum = 0.0;

                for (j = start_idx; j < end_idx; j++) {
                    row_accum += rec_val[j] * prold[rec_colind[j]];
                }
                prnew[r] = row_accum;
            }

            // Fase B: Update PageRank e Riduzioni native OpenMP
            #pragma omp for reduction(+:norm_sq_local, dm_local) schedule(static)
            for (r = global_row_start; r < global_row_end; r++) {
                prnew[r] = prnew[r] * DAMP1 + DAMP2 + (dm_global * DAMP1 / NODES);

                double diff = prnew[r] - prold[r];
                norm_sq_local += diff * diff;

                if (out_degree[r] == 0) {
                    dm_local += prnew[r];
                }
            }

            // Fase C: Sincronizzazione di Rete Inter-Nodo (Solo il thread Master OpenMP comunica)
            #pragma omp master
            {
                // Message Coalescing
                double local_data[2]  = {dm_local, norm_sq_local};
                double global_data[2] = {0.0, 0.0};

                MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                dm_global = global_data[0];
                norm      = sqrt(global_data[1]);

                // Allgatherv: Scambio leggero dei pezzi di vettore aggiornati
                MPI_Allgatherv(prnew + global_row_start, rec_row, MPI_DOUBLE,
                               prold, prows, displs_pr, MPI_DOUBLE,
                               MPI_COMM_WORLD);

                dm_local = 0.0;
                norm_sq_local = 0.0;
                iteration_count++;
            }
            #pragma omp barrier

        } while (norm > ERROR);
    } // CHIUSURA REGIONE PARALLELA OPENMP

    // Sincronizzazione finale per garantire che tutti i processi abbiano terminato il ciclo
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        compute_end = MPI_Wtime();
        compute_time = compute_end - compute_start;   // durata della computazione
    }

    // ========================================================================
    // 4. PROFILING E VALIDAZIONE CON CRUSCOTTO AVANZATO
    // ========================================================================
    if (rank == MASTER) {
        total_time = setup_time + compute_time;

        double sum_pr = pagerank_validate(NODES, prold);

        printf("\n===== FINAL PAGERANK =====\n");
        printf("Number of nodes: %d\n", NODES);
        printf("Number of edges: %d\n", EDGES);
        printf("Iterations: %d\n", iteration_count);
        printf("MPI Processes: %d\n", NPROC);
        printf("OpenMP Threads per MPI process: %d\n", num_threads);
        printf("Total Threads: %d\n", NPROC * num_threads);
        printf("\n");

        /* ----------------------
            PRINT PR SUM CHECK
        ---------------------- */
        printf("╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    PR SUM CHECK                                ║\n");
        printf("╠════════════════════════════════════════════════════════════════╣\n");
        printf("║  Total PR Sum:        %12.10f                               ║\n", sum_pr);
        printf("║  Expected Sum:        %12.10f (should be 1.0)             ║\n", 1.0);

        double pr_diff = fabs(sum_pr - 1.0);
        printf("║  Difference:          %12.10f                               ║\n", pr_diff);

        if (pr_diff < 1e-9) {
            printf("║  Status:              ✓ PASSED (within tolerance)          ║\n");
        } else if (pr_diff < 1e-6) {
            printf("║  Status:              ⚠ WARNING (slightly off)            ║\n");
        } else {
            printf("║  Status:              ✗ FAILED (significant error)        ║\n");
        }
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        printf("\n");


        /* ----------------------
            READ SEQUENTIAL TIME & METRICS DISPLAY
        ---------------------- */
        double seq_time = 0.0;
        FILE *seq_file = fopen("sequential/sequential_time.txt", "r");
        if (seq_file != NULL) {
            if (fscanf(seq_file, "%lf", &seq_time) == 1) {
                printf("\n");
                printf("╔════════════════════════════════════════════════════════════════╗\n");
                printf("║                    PERFORMANCE METRICS                         ║\n");
                printf("╠════════════════════════════════════════════════════════════════╣\n");
                printf("║  Setup Time:          %12.6f seconds                       ║\n", setup_time);
                printf("║  Compute Time:        %12.6f seconds                       ║\n", compute_time);
                printf("║  Total Time:          %12.6f seconds                       ║\n", total_time);
                printf("║  Sequential Time:     %12.6f seconds                       ║\n", seq_time);
                printf("║  MPI Processes:       %12d                                 ║\n", NPROC);
                printf("║  OpenMP Threads:      %12d                                 ║\n", num_threads);
                printf("║  Total Threads:       %12d                                 ║\n", NPROC * num_threads);
                printf("╠════════════════════════════════════════════════════════════════╣\n");

                // Calcolo Speedup ed Efficienza basati sulle tue funzioni di libreria
                double speedup = measure_speedup(seq_time, total_time);
                printf("║  Speedup:             %12.4f x                             ║\n", speedup);

                double efficiency = measure_efficiency(speedup, NPROC * num_threads);
                printf("║  Efficiency:          %12.2f%%                             ║\n", efficiency * 100.0);

                double improvement = ((seq_time - total_time) / seq_time) * 100.0;
                if (improvement > 0) {
                    printf("║  Improvement:         %+12.2f%%                             ║\n", improvement);
                } else {
                    printf("║  Improvement:         %12.2f%% (slower)                    ║\n", improvement);
                }

                double comm_overhead = measure_communication_overhead(total_time, compute_time);
                printf("║  Communication Overhead: %10.2f%%                           ║\n", comm_overhead);

                // Bilanciamento del carico reale calcolato sugli archi (NNZ) assegnati ai processi
                double max_nnz = sendcnts[0];
                double avg_nnz = (double)EDGES / NPROC;
                for (int r = 1; r < NPROC; r++) {
                    if (sendcnts[r] > max_nnz) max_nnz = sendcnts[r];
                }
                double load_balance = measure_load_balance(max_nnz, avg_nnz);
                printf("║  Load Balance (NNZ):  %12.2f%%                             ║\n", load_balance * 100.0);
                printf("║  Partition Size (NNZ): avg=%.1f, max=%.0f                  ║\n", avg_nnz, max_nnz);

                printf("╚════════════════════════════════════════════════════════════════╝\n");
                printf("\n");

                fclose(seq_file);
            } else {
                printf("Warning: Could not read sequential time from file\n");
                fclose(seq_file);
            }
        } else {
            printf("Warning: Could not open sequential/sequential_time.txt\n");
        }
        fflush(stdout);
    }

    // Deallocazione memoria locale
    free(metadata_buffer);
    free(prold); free(prnew);
    free(sendcnts); free(displs); free(prows); free(displs_pr);
    free(rec_val); free(rec_colind);

    MPI_Finalize();
    return 0;
}