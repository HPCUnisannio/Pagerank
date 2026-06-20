#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define CORES 2
#define MASTER 0
#define ERR 0.00001
#define DAMPING 0.85

double sqrt(double x);
void *mat_vec(void *);

// CSR sparse matrix data
double *val, *prold, *prnew;
int *rowind, *colptr, *sum;

// Synchronization primitives
pthread_mutex_t add_mutex;
pthread_mutex_t wait_mutex;
pthread_mutex_t norm_mutex;
pthread_barrier_t our_barrier;
pthread_barrier_t our_barrier2;
pthread_cond_t proceed_cv;

double norm;
int var_wait;

// Dangling nodes mass with dedicated mutex
double global_dangling_mass;
pthread_mutex_t dangling_mutex;

int NODES;
int EDGES;
const char* FILEPATH;

int main(int argc, char *argv[])
{
    printf("Program start\n");
    pthread_t p_threads[CORES];

    pthread_mutex_init(&add_mutex, NULL);
    pthread_mutex_init(&wait_mutex, NULL);
    pthread_mutex_init(&norm_mutex, NULL);
    pthread_mutex_init(&dangling_mutex, NULL);
    pthread_barrier_init(&our_barrier, NULL, CORES);
    pthread_barrier_init(&our_barrier2, NULL, CORES);
    pthread_cond_init(&proceed_cv, NULL);

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

    // Build CSC matrix from file
    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    if (csc_build_from_file(filename, NODES, EDGES, val, rowind, colptr, sum) != 0) {
        return 1;
    }

    // Normalize columns
    csc_normalize_columns(NODES, EDGES, val, colptr, sum);

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);

    printf("initialization complete\n");

    norm = 1.0;
    var_wait = 0;

    printf("\n=== STRUTTURA CSC COSTRUITA ===\n");

    // Launch parallel power iteration
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

    // Verify PageRank sum equals 1.0
    double sum_pr = pagerank_validate(NODES, prold);
    printf("VERIFICA MATEMATICA: Somma PR = %.10f\n", sum_pr);
    printf("=============================================\n");

    /*
    for(i = 0; i < NODES; i++) {
        printf("Node %d: PageRank = %.10f\n", i, prold[i]);
    }
    */

    pthread_mutex_destroy(&add_mutex);
    pthread_mutex_destroy(&wait_mutex);
    pthread_mutex_destroy(&norm_mutex);
    pthread_mutex_destroy(&dangling_mutex);
    pthread_barrier_destroy(&our_barrier);
    pthread_barrier_destroy(&our_barrier2);
    pthread_cond_destroy(&proceed_cv);

    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(sum);

    pthread_exit(NULL);
}

void *mat_vec(void *rank)
{
    long tid = (long)rank;

    int i, col, local_col, my_first_col, my_last_col;
    double *localpr = (double*)calloc(NODES, sizeof(double));
    double current_norm;

    // 1D column partitioning using utility function
    int pcols[CORES], displs[CORES];
    compute_column_distribution(NODES, CORES, pcols, displs);

    local_col = pcols[tid];
    my_first_col = displs[tid];
    my_last_col = my_first_col + local_col - 1;

    do {
        // Reset dangling mass for new iteration
        if (tid == MASTER) {
            global_dangling_mass = 0.0;
        }

        pthread_barrier_wait(&our_barrier2);

        // Compute local dangling mass contribution
        double local_dangling_mass = pagerank_compute_dangling_mass_range(
            prold, sum, my_first_col, my_last_col + 1);

        if (local_dangling_mass > 0.0) {
            pthread_mutex_lock(&dangling_mutex);
            global_dangling_mass += local_dangling_mass;
            pthread_mutex_unlock(&dangling_mutex);
        }

        // Sparse matrix-vector multiplication
        csc_spmv_range(val, rowind, colptr, prold, localpr,
                       my_first_col, my_last_col + 1, 0);

        // Thread-safe accumulation into global vector
        for (i = 0; i < NODES; i++) {
            if (localpr[i] != 0.0) {
                pthread_mutex_lock(&add_mutex);
                prnew[i] += localpr[i];
                pthread_mutex_unlock(&add_mutex);
            }
        }

        pthread_barrier_wait(&our_barrier);

        // Master applies damping and checks convergence
        if (tid == MASTER) {
            pthread_mutex_lock(&norm_mutex);

            double redistribution = (global_dangling_mass * DAMPING) / NODES;
            var_wait = 0;

            double norm_sq = 0.0;
            pagerank_update_and_norm_range(prnew, prold, 0, NODES,
                                           redistribution, DAMPING, NODES, &norm_sq);
            norm = sqrt(norm_sq);

            pthread_mutex_unlock(&norm_mutex);

            memset(prnew, 0, NODES * sizeof(double));

            var_wait = 1;
            pthread_cond_broadcast(&proceed_cv);
        } else {
            pthread_mutex_lock(&wait_mutex);
            while (!var_wait) {
                pthread_cond_wait(&proceed_cv, &wait_mutex);
            }
            pthread_mutex_unlock(&wait_mutex);
        }

        memset(localpr, 0, NODES * sizeof(double));

        pthread_barrier_wait(&our_barrier);

        // Read convergence criterion under mutex protection
        pthread_mutex_lock(&norm_mutex);
        current_norm = norm;
        pthread_mutex_unlock(&norm_mutex);

    } while (current_norm > ERR);

    free(localpr);
    pthread_exit(NULL);
}