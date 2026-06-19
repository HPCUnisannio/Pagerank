#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"

#define CORES 6
#define MASTER 0

double sqrt(double x);
void *mat_vec(void *);

// CSR sparse matrix data
double *val, *prold, *prnew, *damp1, *damp2, *diff;
int *rowind, *colptr, *sum;

// Privatized memory: one contiguous block of CORES * NODES elements
double *all_prnew;

// Partial norms array for lock-free global convergence check
double part_norm[CORES];

// Only barriers remain; all mutexes eliminated
pthread_barrier_t our_barrier;
pthread_barrier_t our_barrier2;
double norm, norm_sq, err = 0.00001;

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

    FILE *fp;
    int colindex, link, i, j=0, c, colmatch=0, localsum=0;
    long t;
    int rc=0;

    val = (double*)calloc(EDGES, sizeof(double));
    rowind = (int*)calloc(EDGES, sizeof(int));
    colptr = (int*)calloc(NODES+1, sizeof(int));
    int co, index;

    prold = (double*)malloc(NODES*sizeof(double));
    prnew = (double*)calloc(NODES, sizeof(double));
    damp1 = (double*)malloc(NODES*sizeof(double));
    damp2 = (double*)malloc(NODES*sizeof(double));
    diff = (double*)calloc(NODES, sizeof(double));
    sum = (int*)calloc(NODES, sizeof(int));

    // Allocate privatized memory block: CORES * NODES contiguous doubles
    all_prnew = (double*)calloc(CORES * NODES, sizeof(double));
    if (!all_prnew) {
        fprintf(stderr, "Errore: Memoria insufficiente per all_prnew\n");
        return 1;
    }

    // Initialize PageRank vectors
    for(i=0; i<NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = 0.85;
        damp2[i] = 0.15/NODES;
    }
    printf("initialization complete\n");

    norm = 1.0;

    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        free(all_prnew);
        return 1;
    }

    // CSC matrix construction
    for(i = 0; i<EDGES; i++) {
        int ret = fscanf(fp, "%d %d", &colindex, &link);
        if (ret != 2) {
            printf("ERROR: fscanf failed at line %d, ret=%d\n", i, ret);
            break;
        }
        colindex = colindex - 1;
        link = link - 1;
        rowind[i] = link;
        if(colmatch==colindex) {
            localsum += 1;
        }
        else {
            sum[j] = localsum;
            colptr[j+1] = colptr[j] + localsum;
            localsum = 1;
            j += 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }
    sum[j] = localsum;
    colptr[j+1]= EDGES;
    fclose(fp);

    // Column normalization
    index = 0;
    for(i = 0; i<NODES; i++) {
        co = sum[i];
        for(j = index; j < index+co; j++) {
            val[j] = val[j]/co;
        }
        index += co;
    }

    printf("\n=== INIZIO COMPUTAZIONE PARALLELA (ZERO MUTEX) ===\n");

    double start_time = get_time();

    for(t=0; t<CORES; t++) {
        rc = pthread_create(&p_threads[t], NULL, mat_vec, (void*)t);
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for(t= 0; t<CORES; t++) {
        pthread_join(p_threads[t], NULL);
    }

    double end_time = get_time();
    double tempo_parallelo = end_time - start_time;

    printf("\n=============================================\n");
    printf("TEMPO DELLA POWER ITERATION (%d THREAD): %.6f secondi\n", CORES, tempo_parallelo);
    printf("=============================================\n");

    // Validate PageRank sum equals 1.0
    double sum_pr = 0.0;
    for(i=0; i<NODES; i++) {
        sum_pr += prold[i];
    }
    printf("Somma finale di controllo PR = %.10f\n", sum_pr);

    free(all_prnew);
    free(val); free(rowind); free(colptr);
    free(prold); free(prnew); free(damp1); free(damp2); free(diff); free(sum);

    pthread_barrier_destroy(&our_barrier);
    pthread_barrier_destroy(&our_barrier2);
    return 0;
}

void *mat_vec(void *rank) {
    long tid = (long)rank;

    int i, j, col;
    double current_norm;

    // Column partitioning for SpMV phase
    int col_chunks = NODES / CORES;
    int col_start = tid * col_chunks;
    int col_end = (tid == CORES - 1) ? NODES : (tid + 1) * col_chunks;

    // Row partitioning for reduction phase
    int node_chunks = NODES / CORES;
    int node_start = tid * node_chunks;
    int node_end = (tid == CORES - 1) ? NODES : (tid + 1) * node_chunks;

    // Private memory pointer for this thread
    double *my_private_prnew = &all_prnew[tid * NODES];

    do {
        // Clear private vector
        memset(my_private_prnew, 0, NODES * sizeof(double));

        // Private SpMV: no mutex needed
        for(col = col_start; col < col_end; col++) {
            for(j = colptr[col]; j < colptr[col+1]; j++) {
                my_private_prnew[rowind[j]] += val[j] * prold[col];
            }
        }

        // Barrier 1: all threads must finish SpMV before reduction
        pthread_barrier_wait(&our_barrier);

        double local_norm_sq = 0.0;

        // Parallel reduction and local norm computation
        for(i = node_start; i < node_end; i++) {
            double raw_pagerank = 0.0;

            // Strided access: sum contributions from all private arrays
            int t;
            for(t = 0; t < CORES; t++) {
                raw_pagerank += all_prnew[t * NODES + i];
            }

            // Apply damping formula
            prnew[i] = raw_pagerank * damp1[i] + damp2[i];

            // Compute local squared difference
            double d = prnew[i] - prold[i];
            local_norm_sq += d * d;

            // Update old vector
            prold[i] = prnew[i];
        }

        // Store partial norm for global reduction
        part_norm[tid] = local_norm_sq;

        // Barrier 2: all threads must write partial norms before reading
        pthread_barrier_wait(&our_barrier2);

        // Lock-free global norm computation
        double total_norm_sq = 0.0;
        int t;
        for(t = 0; t < CORES; t++) {
            total_norm_sq += part_norm[t];
        }
        current_norm = sqrt(total_norm_sq);

    } while(current_norm > err);

    pthread_exit(NULL);
}