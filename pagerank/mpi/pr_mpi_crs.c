#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "../libraries/pagerank_utils2.h"
#include "../libraries/measure.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 1e-6

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CSRGraph g;
    CSRGraph master_g;

    int N = 0, E = 0;
    int i;

    double setup_start = 0.0, setup_end = 0.0;
    double compute_start = 0.0, compute_end = 0.0;
    double total_start = 0.0, total_end = 0.0;

    // Start total timing on master only
    if (rank == MASTER) {
        total_start = measure_get_time();
        setup_start = measure_get_time();
    }

    /* ----------------------
       LOAD GRAPH ONLY ON MASTER
    ---------------------- */
    if (rank == MASTER) {
        if (csr_build_from_file("dataset/data0.dat", &master_g) != 0) {
            printf("Graph loading failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        csr_normalize(&master_g);

        N = master_g.N;
        E = master_g.E;
    }

    /* ----------------------
       BROADCAST GRAPH SIZE
    ---------------------- */
    MPI_Bcast(&N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* ----------------------
       ALLOCATE ON ALL RANKS
    ---------------------- */
    g.N = N;
    g.E = E;
    g.row_ptr = malloc((N + 1) * sizeof(int));
    g.col_idx = malloc(E * sizeof(int));
    g.val     = malloc(E * sizeof(double));
    g.outdeg  = malloc(N * sizeof(int));

    /* ----------------------
       BROADCAST CSR STRUCTURE
    ---------------------- */
    if (rank == MASTER) {
        memcpy(g.row_ptr, master_g.row_ptr, (N + 1) * sizeof(int));
        memcpy(g.col_idx, master_g.col_idx, E * sizeof(int));
        memcpy(g.val, master_g.val, E * sizeof(double));
        memcpy(g.outdeg, master_g.outdeg, N * sizeof(int));
    }

    MPI_Bcast(g.row_ptr, N + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(g.col_idx, E, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(g.val, E, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(g.outdeg, N, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* ----------------------
       CLEAN UP MASTER GRAPH (only on master)
    ---------------------- */
    if (rank == MASTER) {
        csr_free(&master_g);
        setup_end = measure_get_time();  // End setup timing on master
    }

    /* ----------------------
       INIT PAGE RANK
    ---------------------- */
    double *pr = malloc(N * sizeof(double));
    double *local = calloc(N, sizeof(double));

    for (i = 0; i < N; i++)
        pr[i] = 1.0 / N;

    /* ----------------------
       IMPROVED PARTITIONING WITH BALANCED DISTRIBUTION
    ---------------------- */
    int start = 0;
    int end = 0;

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int rem = N % size;
    int r = 0;

    for (r = 0; r < size; r++) {
        counts[r] = N / size + (r < rem ? 1 : 0);
    }

    displs[0] = 0;
    for (r = 1; r < size; r++) {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    start = displs[rank];
    end = start + counts[rank];

    /* ----------------------
       PREALLOC GLOBAL BUFFER
    ---------------------- */
    double *global = malloc(N * sizeof(double));

    double norm;

    // Start compute timing on all ranks (sync all ranks first)
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MASTER) {
        compute_start = measure_get_time();
    }

    /* ----------------------
       ITERATION
    ---------------------- */
    int iteration_count = 0;
    do {
        for (i = 0; i < N; i++)
            local[i] = 0.0;

        /* CSR COMPUTE */
        for (i = start; i < end; i++) {
            int j;
            for (j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
                local[g.col_idx[j]] += g.val[j] * pr[i];
            }
        }

        /* GLOBAL REDUCTION */
        MPI_Allreduce(local, global, N,
                      MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        /* UPDATE */
        double norm_local = 0.0;

        for (i = 0; i < N; i++) {
            double newv =
                DAMPING * global[i]
                + (1.0 - DAMPING) / N;

            norm_local += (newv - pr[i]) * (newv - pr[i]);
            pr[i] = newv;
        }

        MPI_Allreduce(&norm_local, &norm,
                      1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);

        norm = sqrt(norm);
        iteration_count++;

    } while (norm > ERROR);

    // End compute timing on master
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MASTER) {
        compute_end = measure_get_time();
        total_end = measure_get_time();
    }

    /* ----------------------
       CHECK PR SUM (Total probability mass)
    ---------------------- */
    double local_pr_sum = 0.0;
    double global_pr_sum = 0.0;
    
    // Each rank computes sum of PR values in its partition
    for (i = start; i < end; i++) {
        local_pr_sum += pr[i];
    }
    
    // Reduce to get total sum across all ranks
    MPI_Reduce(&local_pr_sum, &global_pr_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    /* ----------------------
       FINAL OUTPUT (SAFE)
    ---------------------- */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        double setup_time = setup_end - setup_start;
        double compute_time = compute_end - compute_start;
        double total_time = total_end - total_start;

        printf("\n===== FINAL PAGERANK =====\n");
        printf("Number of nodes: %d\n", N);
        printf("Number of edges: %d\n", E);
        printf("Iterations: %d\n", iteration_count);
        printf("MPI Processes: %d\n", size);
        printf("\n");

        /* ----------------------
           PRINT PR SUM CHECK
        ---------------------- */
        printf("╔════════════════════════════════════════════════════════════════╗\n");
        printf("║                    PR SUM CHECK                               ║\n");
        printf("╠════════════════════════════════════════════════════════════════╣\n");
        printf("║  Total PR Sum:        %12.10f                              ║\n", global_pr_sum);
        printf("║  Expected Sum:        %12.10f (should be 1.0)             ║\n", 1.0);
        
        double pr_diff = fabs(global_pr_sum - 1.0);
        printf("║  Difference:          %12.10f                              ║\n", pr_diff);
        
        if (pr_diff < 1e-9) {
            printf("║  Status:              ✓ PASSED (within tolerance)          ║\n");
        } else if (pr_diff < 1e-6) {
            printf("║  Status:              ⚠ WARNING (slightly off)            ║\n");
        } else {
            printf("║  Status:              ✗ FAILED (significant error)        ║\n");
        }
        printf("╚════════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        // Print first few nodes
        int print_limit = (N < 20) ? N : 10;
        printf("First %d PageRank values:\n", print_limit);
        for (i = 0; i < print_limit; i++) {
            printf("Node %d -> %.10f\n", i, pr[i]);
        }
        if (N > 20) {
            printf("... (%d more nodes)\n", N - 10);
        }
        printf("\n");

        /* ----------------------
           READ SEQUENTIAL TIME
        ---------------------- */
        double seq_time = 0.0;
        FILE *seq_file = fopen("sequential/sequential_time.txt", "r");
        if (seq_file != NULL) {
            if (fscanf(seq_file, "%lf", &seq_time) == 1) {
                printf("\n");
                printf("╔════════════════════════════════════════════════════════════════╗\n");
                printf("║                    PERFORMANCE METRICS                        ║\n");
                printf("╠════════════════════════════════════════════════════════════════╣\n");
                printf("║  Setup Time:          %12.6f seconds                        ║\n", setup_time);
                printf("║  Compute Time:        %12.6f seconds                        ║\n", compute_time);
                printf("║  Total Time:          %12.6f seconds                        ║\n", total_time);
                printf("║  Sequential Time:     %12.6f seconds                        ║\n", seq_time);
                printf("╠════════════════════════════════════════════════════════════════╣\n");

                // Calculate speedup
                double speedup = measure_speedup(seq_time, total_time);
                printf("║  Speedup:             %12.4f x                             ║\n", speedup);
                
                // Calculate efficiency
                double efficiency = measure_efficiency(speedup, size);
                printf("║  Efficiency:          %12.2f%%                             ║\n", efficiency * 100.0);
                
                // Performance improvement
                double improvement = ((seq_time - total_time) / seq_time) * 100.0;
                if (improvement > 0) {
                    printf("║  Improvement:         %+12.2f%%                             ║\n", improvement);
                } else {
                    printf("║  Improvement:         %12.2f%% (slower)                  ║\n", improvement);
                }

                // Communication overhead estimation
                double comm_overhead = measure_communication_overhead(total_time, compute_time);
                printf("║  Communication Overhead: %10.2f%%                           ║\n", comm_overhead);
                
                // Load balance
                double max_count = counts[0];
                double avg_count = (double)N / size;
                int r;
                for (r = 1; r < size; r++) {
                    if (counts[r] > max_count) max_count = counts[r];
                }
                double load_balance = measure_load_balance(max_count, avg_count);
                printf("║  Load Balance:        %12.2f%%                             ║\n", load_balance * 100.0);
                printf("║  Partition Size:      avg=%.1f, max=%.0f                 ║\n", avg_count, max_count);
                
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

    /* ----------------------
       CLEANUP
    ---------------------- */
    free(pr);
    free(local);
    free(global);
    free(counts);
    free(displs);

    csr_free(&g);

    MPI_Finalize();
    return 0;
}