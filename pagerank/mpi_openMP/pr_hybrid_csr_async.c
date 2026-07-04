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

    int mpi_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  AVVIO PAGERANK (VERSIONE ASINCRONA / GHOST NODES)\n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Nodi: %d - Archi: %d\n", NODES, EDGES);
        printf("║  Processi MPI totali    : %d\n", NPROC);
        printf("║  Thread OpenMP/processo : %d\n", num_threads);
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }

    double *val = NULL;
    int *colind = NULL;
    int *metadata_buffer = (int *)calloc((NODES + 1) + NODES, sizeof(int));
    int *rowptr     = metadata_buffer;
    int *out_degree = metadata_buffer + (NODES + 1);

    double *prold = (double *)malloc(NODES * sizeof(double));
    double *prnew = (double *)malloc(NODES * sizeof(double));

    // Buffer per memorizzare i risultati parziali (solo archi locali)
    double *partial_prnew = (double *)malloc(NODES * sizeof(double));

    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *prows     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_row;
    double dm_global = 0.0;

    // ========================================================================
    // 1. SETUP STANDARD E DISTRIBUZIONE
    // ========================================================================
    double setup_start = 0.0, setup_end = 0.0;
    double compute_start = 0.0, compute_end = 0.0;
    double setup_time = 0.0, compute_time = 0.0, total_time = 0.0;

    if (rank == MASTER) {
        setup_start = MPI_Wtime();
        val = (double *)calloc(EDGES, sizeof(double));
        colind = (int *)calloc(EDGES, sizeof(int));
        csr_build_from_file(FILEPATH, NODES, EDGES, val, colind, rowptr, out_degree);

        int i;
        #pragma omp parallel for reduction(+:dm_global)
        for (i = 0; i < NODES; i++) {
            if (out_degree[i] == 0) dm_global += 1.0;
        }
        dm_global /= NODES;
    }

    MPI_Bcast(metadata_buffer, (2 * NODES + 1), MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&dm_global, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    compute_column_distribution(NODES, NPROC, prows, displs_pr);
    compute_nnz_distribution(NPROC, prows, rowptr, sendcnts, displs);

    int my_cnt = sendcnts[rank];
    rec_row = prows[rank];

    double *rec_val    = (double *)malloc(my_cnt * sizeof(double));
    int    *rec_colind = (int *)malloc(my_cnt * sizeof(int));

    // Distribuzione dei vettori della matrice tramite Scatterv
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(colind, sendcnts, displs, MPI_INT, rec_colind, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        free(val);
        free(colind);
    }

    // ========================================================================
    // OTTIMIZZAZIONE ACCORPATA: UNICO BLOCCO OPENMP PER VETTORI E MASCHERA
    // ========================================================================
    // Spostato qui dopo il Bcast per combinare First-Touch NUMA e Maschera Dangling
    #pragma omp parallel
    {
        int i;
        #pragma omp for schedule(static)
        for (i = 0; i < NODES; i++) {
            prnew[i] = 0.0;
            partial_prnew[i] = 0.0;
            prold[i] = 1.0 / NODES;
        }
    }

    // ========================================================================
    // PRE-PROCESSING PARALLELO: DIVISIONE ARCHI LOCALI E REMOTI (GHOST NODES)
    // ========================================================================
    int global_row_start = displs_pr[rank];
    int global_row_end   = global_row_start + rec_row;

    int *loc_rowptr = (int *)calloc(rec_row + 1, sizeof(int));
    int *rem_rowptr = (int *)calloc(rec_row + 1, sizeof(int));

    // 1. CONTEGGIO IN PARALLELO (Nessuna dipendenza tra righe r)
    int r, j;
    #pragma omp parallel for schedule(static) private(j)
    for (r = 0; r < rec_row; r++) {
        int start_idx = rowptr[global_row_start + r] - displs[rank];
        int end_idx   = rowptr[global_row_start + r + 1] - displs[rank];
        for (j = start_idx; j < end_idx; j++) {
            int target = rec_colind[j];
            if (target >= global_row_start && target < global_row_end) {
                loc_rowptr[r + 1]++;
            } else {
                rem_rowptr[r + 1]++;
            }
        }
    }

    // 2. PREFIX SUM (Deve restare sequenziale: loop-carried dependency)
    for (r = 0; r < rec_row; r++) {
        loc_rowptr[r + 1] += loc_rowptr[r];
        rem_rowptr[r + 1] += rem_rowptr[r];
    }

    // Allocazione delle strutture locali splittate
    double *loc_val    = (double *)malloc(loc_rowptr[rec_row] * sizeof(double));
    int    *loc_colind = (int *)malloc(loc_rowptr[rec_row] * sizeof(int));
    double *rem_val    = (double *)malloc(rem_rowptr[rec_row] * sizeof(double));
    int    *rem_colind = (int *)malloc(rem_rowptr[rec_row] * sizeof(int));

    int *loc_offset = (int *)calloc(rec_row, sizeof(int));
    int *rem_offset = (int *)calloc(rec_row, sizeof(int));

    // 3. POPOLAMENTO IN PARALLELO (Ogni thread scrive nella propria fetta contigua senza conflitti)
    #pragma omp parallel for schedule(static) private(j)
    for (r = 0; r < rec_row; r++) {
        int start_idx = rowptr[global_row_start + r] - displs[rank];
        int end_idx   = rowptr[global_row_start + r + 1] - displs[rank];
        for (j = start_idx; j < end_idx; j++) {
            int target = rec_colind[j];
            double weight = rec_val[j];
            if (target >= global_row_start && target < global_row_end) {
                int idx = loc_rowptr[r] + loc_offset[r]++;
                loc_colind[idx] = target;
                loc_val[idx]    = weight;
            } else {
                int idx = rem_rowptr[r] + rem_offset[r]++;
                rem_colind[idx] = target;
                rem_val[idx]    = weight;
            }
        }
    }

    // Liberiamo la memoria temporanea non più necessaria
    free(loc_offset);
    free(rem_offset);
    free(rec_val);
    free(rec_colind);

    if (rank == MASTER) {
        setup_end = MPI_Wtime();
        setup_time = setup_end - setup_start;
    }

    // ========================================================================
    // 3. REGIONE COMPUTAZIONALE OVERLAPPATA
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) compute_start = MPI_Wtime();

    double norm = 0.0;
    double dm_local = 0.0, norm_sq_local = 0.0;
    int iteration_count = 0;
    MPI_Request req_pr;

    #pragma omp parallel
    {
        int r, j;
        do {
            double local_dm = 0.0;
            double local_norm = 0.0;

            #pragma omp master
            {
                if (iteration_count > 0) {
                    double *tmp = prold;
                    prold = prnew;
                    prnew = tmp;
                }
                
                // Uso di MPI_IN_PLACE per evitare Data Race con la Fase 1
                MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                prold, prows, displs_pr, MPI_DOUBLE,
                                MPI_COMM_WORLD, &req_pr);
            }
            // FIX 2: Barriera post-swap per far partire i thread all'unisono
            #pragma omp barrier

            // --- FASE 1: CALCOLO LOCALE (Mentre la rete trasmette in background) ---
            #pragma omp for schedule(static) nowait
            for (r = 0; r < rec_row; r++) {
                int global_r = global_row_start + r;
                double row_accum = 0.0;
                #pragma omp simd reduction(+:row_accum)
                for (j = loc_rowptr[r]; j < loc_rowptr[r + 1]; j++) {
                    row_accum += loc_val[j] * prold[loc_colind[j]];
                }
                partial_prnew[global_r] = row_accum;
            }

            // --- FASE 2: ATTESA RETE ---
            #pragma omp master
            {
                MPI_Wait(&req_pr, MPI_STATUS_IGNORE);
            }
            #pragma omp barrier // Tutti i thread aspettano i dati

            // --- FASE 3: CALCOLO REMOTO E FINALIZZAZIONE ---
            #pragma omp for schedule(static) nowait
            for (r = 0; r < rec_row; r++) {
                int global_r = global_row_start + r;
                double row_accum = partial_prnew[global_r];  // Start with local sum

                // Add remote contributions
                #pragma omp simd reduction(+:row_accum)
                for (j = rem_rowptr[r]; j < rem_rowptr[r + 1]; j++) {
                    row_accum += rem_val[j] * prold[rem_colind[j]];
                }

                // FIX: Apply the formula ONCE with the complete sum
                prnew[global_r] = row_accum * DAMP1 + DAMP2 + (dm_global * DAMP1 / NODES);

                double diff = prnew[global_r] - prold[global_r];
                local_norm += diff * diff;
                
                // FIX: Use prold for dangling mass computation
                if(out_degree[global_r] == 0) {
                    local_dm += prnew[global_r];
                }
            }

            // Manual reduction to avoid false sharing
            #pragma omp atomic
            norm_sq_local += local_norm;
            
            #pragma omp atomic
            dm_local += local_dm;

            #pragma omp barrier  // Ensure all threads complete before master

            // Aggiornamento Metriche
            #pragma omp master
            {
                double local_data[2]  = {dm_local, norm_sq_local};
                double global_data[2] = {0.0, 0.0};

                MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                dm_global = global_data[0];
                norm      = sqrt(global_data[1]);

                dm_local = 0.0;
                norm_sq_local = 0.0;
                iteration_count++;
            }
            #pragma omp barrier

        } while (norm > ERROR);
    }

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

    free(metadata_buffer); free(prold); free(prnew); free(partial_prnew);
    free(sendcnts); free(displs); free(prows); free(displs_pr);
    free(loc_rowptr); free(loc_colind); free(loc_val);
    free(rem_rowptr); free(rem_colind); free(rem_val);

    MPI_Finalize();
    return 0;
}
