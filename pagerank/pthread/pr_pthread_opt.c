#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define CORES 6
#define MASTER 0

double sqrt(double x);
void *mat_vec(void *);

// CSR sparse matrix data
double *val, *prold, *prnew;
int *rowind, *colptr, *sum;

// Privatized memory: one contiguous block of CORES * NODES elements
double *all_prnew;

// Partial norms array for lock-free global convergence check
double part_norm[CORES];

// Only barriers remain; all mutexes eliminated
pthread_barrier_t our_barrier;
pthread_barrier_t our_barrier2;
double norm, err = 0.00001;

int NODES;
int EDGES;
const char* FILEPATH;

int main(int argc, char *argv[])
{
    printf("Program start (Privatized Version)\n");
    pthread_t p_threads[CORES];

    pthread_barrier_init(&our_barrier, NULL, CORES);
    pthread_barrier_init(&our_barrier2, NULL, CORES);

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    NODES = graph->nodes;
    EDGES = graph->edges;
    FILEPATH = graph->filepath;

    long t;
    int rc = 0;

    val    = (double*)calloc(EDGES, sizeof(double));
    rowind = (int*)calloc(EDGES, sizeof(int));
    colptr = (int*)calloc(NODES + 1, sizeof(int));
    sum    = (int*)calloc(NODES, sizeof(int));

    prold = (double*)malloc(NODES * sizeof(double));
    prnew = (double*)calloc(NODES, sizeof(double));

    // Allocate privatized memory block: CORES * NODES contiguous doubles
    all_prnew = (double*)calloc(CORES * NODES, sizeof(double));
    if (!all_prnew) {
        fprintf(stderr, "Errore: Memoria insufficiente per all_prnew\n");
        return 1;
    }

    // Build CSC matrix from file
    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    if (csc_build_from_file(filename, NODES, EDGES, val, rowind, colptr, sum) != 0) {
        free(all_prnew);
        return 1;
    }

    // Normalize columns
    csc_normalize_columns(NODES, EDGES, val, colptr, sum);

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);

    printf("initialization complete\n");

    norm = 1.0;

    printf("\n=== INIZIO COMPUTAZIONE PARALLELA (ZERO MUTEX) ===\n");

    double start_time = get_time();

    for (t = 0; t < CORES; t++) {
        rc = pthread_create(&p_threads[t], NULL, mat_vec, (void*)t);
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (t = 0; t < CORES; t++) {
        pthread_join(p_threads[t], NULL);
    }

    double end_time = get_time();
    double tempo_parallelo = end_time - start_time;

    printf("\n=============================================\n");
    printf("TEMPO DELLA POWER ITERATION (%d THREAD): %.6f secondi\n", CORES, tempo_parallelo);
    printf("=============================================\n");

    // Validate PageRank sum equals 1.0
    double sum_pr = pagerank_validate(NODES, prold);
    printf("Somma finale di controllo PR = %.10f\n", sum_pr);

    /*
    for(i = 0; i < NODES; i++) {
        printf("Node %d: PageRank = %.10f\n", i, prold[i]);
    }
    */

    free(all_prnew);
    free(val); free(rowind); free(colptr);
    free(prold); free(prnew); free(sum);

    pthread_barrier_destroy(&our_barrier);
    pthread_barrier_destroy(&our_barrier2);
    return 0;
}

void *mat_vec(void *rank)
{
    long tid = (long)rank;

    int i, col;
    double current_norm;

    // Column partitioning for SpMV phase using utility function
    int pcols[CORES], displs[CORES];
    compute_column_distribution(NODES, CORES, pcols, displs);

    int col_start = displs[tid];
    int col_end = col_start + pcols[tid];

    // Row partitioning for reduction phase (same distribution)
    int node_start = displs[tid];
    int node_end = node_start + pcols[tid];

    // Private memory pointer for this thread
    double *my_private_prnew = &all_prnew[tid * NODES];

    do {
        // Clear private vector
        memset(my_private_prnew, 0, NODES * sizeof(double));

        // Private SpMV: no mutex needed
        csc_spmv_range(val, rowind, colptr, prold, my_private_prnew,
                       col_start, col_end, 0);

        // Barrier 1: all threads must finish SpMV before reduction
        pthread_barrier_wait(&our_barrier);

        // Parallel reduction: sum contributions from all private arrays
        for (i = node_start; i < node_end; i++) {
            double raw_pagerank = 0.0;
            int t;
            for (t = 0; t < CORES; t++) {
                raw_pagerank += all_prnew[t * NODES + i];
            }
            prnew[i] = raw_pagerank;
        }

        // Apply damping, compute norm, and update prold for assigned rows
        double local_norm_sq = 0.0;
        pagerank_update_and_norm_range(prnew, prold, node_start, node_end,
                                       0.0, DAMPING, NODES, &local_norm_sq);

        // Store partial norm for global reduction
        part_norm[tid] = local_norm_sq;

        // Barrier 2: all threads must write partial norms before reading
        pthread_barrier_wait(&our_barrier2);

        // Lock-free global norm computation
        double total_norm_sq = 0.0;
        int t;
        for (t = 0; t < CORES; t++) {
            total_norm_sq += part_norm[t];
        }
        current_norm = sqrt(total_norm_sq);

    } while (current_norm > err);

    pthread_exit(NULL);
}