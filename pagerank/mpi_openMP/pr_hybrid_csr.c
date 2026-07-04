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
    GraphType graph_type = GRAPH_BIGGEST;
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

    // Inizializzazione MPI
    int mpi_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  AVVIO PAGERANK MPI + OpenMP (VERSIONE HARDWARE-AWARE)\n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Nodi: %d - Archi: %d\n", NODES, EDGES);
        printf("║  Processi MPI totali    : %d\n", NPROC);
        printf("║  Thread OpenMP/processo : %d\n", num_threads);
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }

    double *val = NULL;
    int *colind = NULL;

    // Allocazione contigua dei metadati strutturali
    int *metadata_buffer = (int *)calloc((NODES + 1) + NODES, sizeof(int));
    int *rowptr     = metadata_buffer;
    int *out_degree = metadata_buffer + (NODES + 1);

    // Vettori PageRank allocati con malloc (nessun page-fault sequenziale come in calloc)
    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)malloc(NODES * sizeof(double));

    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *prows     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_row;
    double dm_global = 0.0;

    // --- OTTIMIZZAZIONE 1: NUMA FIRST-TOUCH ---
    // Alloca fisicamente le pagine di memoria RAM in modo distribuito sui socket
    #pragma omp parallel
    {
        int i;
        #pragma omp for schedule(static)
        for (i = 0; i < NODES; i++) {
            prnew[i] = 0.0;
            prold[i] = 1.0 / NODES;
        }
    }

    // ========================================================================
    // 1. LETTURA FILE E COSTRUZIONE MATRICE CSR (SOLO MASTER)
    // ========================================================================
    double setup_start = 0.0, setup_end = 0.0;
    double compute_start = 0.0, compute_end = 0.0;
    double setup_time = 0.0, compute_time = 0.0, total_time = 0.0;

    if (rank == MASTER) {
        setup_start = MPI_Wtime();
        val    = (double *)calloc(EDGES, sizeof(double));
        colind = (int *)calloc(EDGES, sizeof(int));

        if (csr_build_from_file(FILEPATH, NODES, EDGES, val, colind, rowptr, out_degree) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int i;
        #pragma omp parallel for reduction(+:dm_global)
        for (i = 0; i < NODES; i++) {
            if (out_degree[i] == 0) dm_global += 1.0;
        }
        dm_global /= NODES;
    }

    // ========================================================================
    // 2. DISTRIBUZIONE DEI METADATI VIA MPI
    // ========================================================================
    MPI_Bcast(metadata_buffer, (2 * NODES + 1), MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&dm_global, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    compute_column_distribution(NODES, NPROC, prows, displs_pr);
    compute_nnz_distribution(NPROC, prows, rowptr, sendcnts, displs);

    int my_cnt = sendcnts[rank];
    rec_row = prows[rank];

    double *rec_val    = (double *)malloc(my_cnt * sizeof(double));
    int    *rec_colind = (int *)malloc(my_cnt * sizeof(int));

    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(colind, sendcnts, displs, MPI_INT, rec_colind, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        free(val); free(colind);
        setup_end = MPI_Wtime();
        setup_time = setup_end - setup_start;
        printf("Setup completato in %.6f secondi.\n", setup_time);
        fflush(stdout);
    }

    // ========================================================================
    // 3. REGIONE COMPUTAZIONALE IBRIDA (POWER ITERATION CORE)
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MASTER) compute_start = MPI_Wtime();

    double norm = 0.0;
    double dm_local = 0.0, norm_sq_local = 0.0;
    int iteration_count = 0;

    int global_row_start = displs_pr[rank];
    int global_row_end   = global_row_start + rec_row;

    // APERTURA REGIONE PARALLELA OPENMP
    #pragma omp parallel
    {
        int r, j;
        do {
            double local_dm = 0.0;
            double local_norm = 0.0;
            
            #pragma omp for schedule(static) nowait
            for (r = global_row_start; r < global_row_end; r++) {
                int start_idx = rowptr[r] - displs[rank];
                int end_idx   = rowptr[r + 1] - displs[rank];
                double row_accum = 0.0;
                
                // SIMD vectorization without reduction clause
                #pragma omp simd
                for (j = start_idx; j < end_idx; j++) {
                    row_accum += rec_val[j] * prold[rec_colind[j]];
                }
                
                prnew[r] = row_accum * DAMP1 + DAMP2 + (dm_global * DAMP1 / NODES);
                
                double diff = prnew[r] - prold[r];
                local_norm += diff * diff;
                if(out_degree[r] == 0) local_dm += prnew[r];
            }
            
            // Manual reduction to avoid false sharing
            #pragma omp atomic
            norm_sq_local += local_norm;
            
            #pragma omp atomic
            dm_local += local_dm;
            
            #pragma omp barrier  // Ensure all threads complete before master

            // Sincronizzazione MPI
            #pragma omp master
            {
                double local_data[2]  = {dm_local, norm_sq_local};
                double global_data[2] = {0.0, 0.0};

                MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                dm_global = global_data[0];
                norm      = sqrt(global_data[1]);

                MPI_Allgatherv(prnew + global_row_start, rec_row, MPI_DOUBLE,
                               prold, prows, displs_pr, MPI_DOUBLE,
                               MPI_COMM_WORLD);

                dm_local = 0.0;
                norm_sq_local = 0.0;
                iteration_count++;
            }
            #pragma omp barrier

        } while (norm > ERROR);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MASTER) {
        compute_end = MPI_Wtime();
        compute_time = compute_end - compute_start;
    }

    // ========================================================================
    // 4. PROFILING E VALIDAZIONE CON CRUSCOTTO AVANZATO
    // ========================================================================
    if (rank == MASTER) {
        total_time = setup_time + compute_time;

        double sum_pr = pagerank_validate(NODES, prold);

        /* ----------------------
            PRINT PR SUM CHECK
        ---------------------- */
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  PR SUM CHECK                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Total PR Sum:        %12.10f                              \n", sum_pr);
        printf("║  Expected Sum:        %12.10f              \n", 1.0);

        double pr_diff = fabs(sum_pr - 1.0);
        printf("║  Difference:          %12.10f                               \n", pr_diff);

        if (pr_diff < 1e-9) {
            printf("║  Status:              ✓ PASSED (within tolerance)         \n");
        } else if (pr_diff < 1e-6) {
            printf("║  Status:              ⚠ WARNING (slightly off)            \n");
        } else {
            printf("║  Status:              ✗ FAILED (significant error)        \n");
        }
        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        /* ----------------------
            TIME PERFORMANCE & METRICS DISPLAY
        ---------------------- */
        // Print execution summary
        char label[100];
        sprintf(label, "MPI (%d processes) + OpenMP (%d threads)", NPROC, num_threads);
        measure_print_summary(label, setup_time, compute_time, total_time, iteration_count);

        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  LOAD BALANCE                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");

        // Bilanciamento del carico reale calcolato sugli archi (NNZ) assegnati ai processi
        double max_nnz = sendcnts[0];
        double avg_nnz = (double)EDGES / NPROC;
        int r;
        for (r = 1; r < NPROC; r++) {
            if (sendcnts[r] > max_nnz) max_nnz = sendcnts[r];
        }
        double load_balance = measure_load_balance(max_nnz, avg_nnz);
        printf("║  Load Balance (NNZ):  %12.2f%%                             \n", load_balance * 100.0);
        printf("║  Partition Size (NNZ): avg=%.1f, max=%.0f                  \n", avg_nnz, max_nnz);

        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        /* ----------------------
            READ SEQUENTIAL TIME & METRICS DISPLAY
        ---------------------- */
        double sequential_time = 0.0;
        FILE *seq_file = fopen("sequential/sequential_time.txt", "r");
        if (seq_file != NULL) {
            fscanf(seq_file, "%lf", &sequential_time);
            fclose(seq_file);

            char label1[30];
            sprintf(label1, "Sequenziale");
            char label2[30];
            sprintf(label2, "MPI (%d processes) + OpenMP (%d threads)", NPROC, num_threads);

            measure_print_comparison(label1, sequential_time, label2, compute_time);
        } else {
            printf("Warning: Could not open sequential/sequential_time.txt\n");
        }
        fflush(stdout);
    }

    free(metadata_buffer); free(prold); free(prnew);
    free(sendcnts); free(displs); free(prows); free(displs_pr);
    free(rec_val); free(rec_colind);

    MPI_Finalize();
    return 0;
}