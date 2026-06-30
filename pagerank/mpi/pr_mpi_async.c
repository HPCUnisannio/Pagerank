#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char *argv[])
{
    int NPROC, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    // Caricamento metadati del grafo
    GraphType graph_type = GRAPH_BIGGEST;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    if (rank == MASTER) {
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  AVVIO PAGERANK MPI Puro (ASINCRONO / OVERLAP)             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Nodi: %d - Archi: %d\n", NODES, EDGES);
        printf("║  Processi MPI totali : %d\n", NPROC);
        printf("║  Modello Computazione: Non-Blocking Collectives (Ghost)\n");
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }

    double *val = NULL;
    int *colind = NULL;

    // Allocazione contigua metadati
    int *metadata_buffer = (int*)calloc((NODES + 1) + NODES, sizeof(int));
    int *rowptr     = metadata_buffer;
    int *out_degree = metadata_buffer + (NODES + 1);

    double *prold = (double*)malloc(NODES * sizeof(double));
    double *prnew = (double*)calloc(NODES, sizeof(double));

    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *prows     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_row;

    double dm_global = 0.0;

    pagerank_init_vector(NODES, prold);

    // ========================================================================
    // FASE 1: SETUP STANDARD
    // ========================================================================
    double setup_start = 0.0, setup_end = 0.0;
    double compute_start = 0.0, compute_end = 0.0;
    double setup_time = 0.0, compute_time = 0.0, total_time = 0.0;

    if (rank == MASTER) setup_start = MPI_Wtime();

    if (rank == MASTER) {
        val    = (double*)calloc(EDGES, sizeof(double));
        colind = (int*)calloc(EDGES, sizeof(int));

        if (csr_build_from_file(FILEPATH, NODES, EDGES, val, colind, rowptr, out_degree) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int total_dangling_nodes = 0;
        int i;
        for (i = 0; i < NODES; i++) {
            if (out_degree[i] == 0) total_dangling_nodes++;
        }
        dm_global = (double) total_dangling_nodes / NODES;
    }

    MPI_Bcast(metadata_buffer, (2 * NODES + 1), MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&dm_global, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    compute_column_distribution(NODES, NPROC, prows, displs_pr);
    compute_nnz_distribution(NPROC, prows, rowptr, sendcnts, displs);

    int my_cnt = sendcnts[rank];
    rec_row = prows[rank];

    double *rec_val    = (double*)malloc(my_cnt * sizeof(double));
    int    *rec_colind = (int*)malloc(my_cnt * sizeof(int));

    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(colind, sendcnts, displs, MPI_INT, rec_colind, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        free(val);
        free(colind);
    }

    // ========================================================================
    // FASE 1.5: PRE-PROCESSING (SPLITTING ARCHI LOCALI/REMOTI)
    // ========================================================================
    int global_row_start = displs_pr[rank];
    int global_row_end   = global_row_start + rec_row;

    int *loc_rowptr = (int *)calloc(rec_row + 1, sizeof(int));
    int *rem_rowptr = (int *)calloc(rec_row + 1, sizeof(int));

    // 1. Conteggio
    int r, j;
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

    // 2. Prefix sum
    for (r = 0; r < rec_row; r++) {
        loc_rowptr[r + 1] += loc_rowptr[r];
        rem_rowptr[r + 1] += rem_rowptr[r];
    }

    // 3. Allocazione e popolamento
    double *loc_val    = (double *)malloc(loc_rowptr[rec_row] * sizeof(double));
    int    *loc_colind = (int *)malloc(loc_rowptr[rec_row] * sizeof(int));
    double *rem_val    = (double *)malloc(rem_rowptr[rec_row] * sizeof(double));
    int    *rem_colind = (int *)malloc(rem_rowptr[rec_row] * sizeof(int));

    int *loc_offset = (int *)calloc(rec_row, sizeof(int));
    int *rem_offset = (int *)calloc(rec_row, sizeof(int));

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

    free(loc_offset); free(rem_offset); free(rec_val); free(rec_colind);

    // Buffer per i risultati parziali degli archi locali
    double *partial_prnew = (double *)malloc(rec_row * sizeof(double));

    if (rank == MASTER) {
        setup_end = MPI_Wtime();
        setup_time = setup_end - setup_start;
        printf("Setup completato in %.6f secondi.\n", setup_time);
        printf("Inizio computazione asincrona iterativa...\n");
        fflush(stdout);
    }

    // ========================================================================
    // FASE 2: COMPUTAZIONE ITERATIVA (NON-BLOCKING IALLGATHERV)
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MASTER) compute_start = MPI_Wtime();

    double norm = 0.0;
    double dm_local = 0.0, norm_sq_local = 0.0;
    int iteration_count = 0;
    MPI_Request req_pr;

    do {
        // Se non è la prima iterazione, trasferiamo i nuovi valori nel buffer sorgente
        if (iteration_count > 0) {
            memcpy(prold + global_row_start, prnew + global_row_start, rec_row * sizeof(double));
        }

        // START RETE: Lanciamo la comunicazione asincrona. MPI_IN_PLACE indica che
        // i nostri dati da spedire si trovano già al posto giusto dentro 'prold'
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                        prold, prows, displs_pr, MPI_DOUBLE,
                        MPI_COMM_WORLD, &req_pr);

        // START CPU: Calcolo degli archi locali (non necessitano della rete)
        for (r = 0; r < rec_row; r++) {
            double accum = 0.0;
            for (j = loc_rowptr[r]; j < loc_rowptr[r + 1]; j++) {
                accum += loc_val[j] * prold[loc_colind[j]];
            }
            partial_prnew[r] = accum;
        }

        // SINCRONIZZAZIONE: Attendiamo che la rete abbia consegnato i vettori degli altri processi
        MPI_Wait(&req_pr, MPI_STATUS_IGNORE);

        // RIPRESA CPU: Calcolo degli archi remoti e chiusura del vettore
        dm_local = 0.0;
        norm_sq_local = 0.0;
        double redistribution = dm_global * DAMP1 / NODES;

        for (r = 0; r < rec_row; r++) {
            double accum = 0.0;
            for (j = rem_rowptr[r]; j < rem_rowptr[r + 1]; j++) {
                accum += rem_val[j] * prold[rem_colind[j]];
            }

            int global_r = global_row_start + r;

            // Fusione: Parziale Locale + Parziale Remoto + Damping
            prnew[global_r] = (partial_prnew[r] + accum) * DAMP1 + DAMP2 + redistribution;

            double diff = prnew[global_r] - prold[global_r];
            norm_sq_local += diff * diff;

            // Calcolo inline branchless del dangling mass
            if (out_degree[global_r] == 0) {
                dm_local += prnew[global_r];
            }
        }

        // Riduzione di coda globale
        double local_data[2]  = {dm_local, norm_sq_local};
        double global_data[2] = {0.0, 0.0};
        MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        dm_global = global_data[0];
        norm      = sqrt(global_data[1]);
        iteration_count++;

    } while (norm > ERROR);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        compute_end = MPI_Wtime();
        compute_time = compute_end - compute_start;
    }

    // ========================================================================
    // FASE 3: PROFILING E VALIDAZIONE
    // ========================================================================
    if (rank == MASTER) {
        total_time = setup_time + compute_time;

        double sum_pr = pagerank_validate(NODES, prold);

        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  PR SUM CHECK                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Total PR Sum:        %12.10f                              \n", sum_pr);
        printf("║  Expected Sum:        %12.10f                             \n", 1.0);

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

        char label[100];
        sprintf(label, "MPI OVERLAP (%d processes)", NPROC);
        measure_print_summary(label, setup_time, compute_time, total_time, iteration_count);

        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  LOAD BALANCE                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");

        double max_nnz = sendcnts[0];
        double avg_nnz = (double)EDGES / NPROC;
        for (r = 1; r < NPROC; r++) {
            if (sendcnts[r] > max_nnz) max_nnz = sendcnts[r];
        }
        double load_balance = measure_load_balance(max_nnz, avg_nnz);
        printf("║  Load Balance (NNZ):  %12.2f%%                             \n", load_balance * 100.0);
        printf("║  Partition Size (NNZ): avg=%.1f, max=%.0f                  \n", avg_nnz, max_nnz);

        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        double sequential_time = 0.0;
        FILE *seq_file = fopen("sequential/sequential_time.txt", "r");
        if (seq_file != NULL) {
            fscanf(seq_file, "%lf", &sequential_time);
            fclose(seq_file);

            char label1[30];
            sprintf(label1, "Sequenziale");
            char label2[30];
            sprintf(label2, "MPI ASYNC (%d proc)", NPROC);

            measure_print_comparison(label1, sequential_time, label2, compute_time);
        } else {
            printf("Warning: Could not open sequential/sequential_time.txt\n");
        }
        fflush(stdout);
    }

    // Cleanup memorie aggiuntive allocate per l'overlap
    free(metadata_buffer);
    free(prold);
    free(prnew);
    free(partial_prnew);
    free(sendcnts);
    free(displs);
    free(prows);
    free(displs_pr);
    free(loc_rowptr);
    free(loc_colind);
    free(loc_val);
    free(rem_rowptr);
    free(rem_colind);
    free(rem_val);

    MPI_Finalize();
    return 0;
}
