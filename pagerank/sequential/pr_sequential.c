#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define DAMPING 0.85
#define ERR     0.00001

double sqrt(double x);

int main(int argc, char *argv[])
{
    printf("Program start\n");

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    // Allocate CSC matrix structures
    double *val    = (double *) calloc(EDGES, sizeof(double));
    int    *rowind = (int *)    calloc(EDGES, sizeof(int));
    int    *colptr = (int *)    calloc(NODES + 1, sizeof(int));
    int    *sum    = (int *)    calloc(NODES, sizeof(int));

    // Allocate PageRank vectors
    double *prold = (double *) malloc(NODES * sizeof(double));
    double *prnew = (double *) calloc(NODES, sizeof(double));

    double norm, norm_sq;

    if (!val || !rowind || !colptr || !prold || !prnew || !sum) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        return 1;
    }

    // Build CSC matrix from file
    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    
    double t_setup_start = get_time();
    if (csc_build_from_file(filename, NODES, EDGES, val, rowind, colptr, sum) != 0) {
        return 1;
    }

    // Normalize columns
    csc_normalize_columns(NODES, EDGES, val, colptr, sum);

    // Initialize PageRank vector
    pagerank_init_vector(NODES, prold);
    double t_setup_end = get_time();
    double setup_time = t_setup_end - t_setup_start;

    printf("Setup up complete\n");

    // Power iteration
    double t_compute_start = get_time();
    
    do {
        memset(prnew, 0, NODES * sizeof(double));

        // Compute dangling mass and redistribution
        double dangling_mass = pagerank_compute_dangling_mass(NODES, prold, sum);
        double redistribution = (dangling_mass * DAMPING) / NODES;

        // Sparse matrix-vector multiplication
        csc_spmv_range(val, rowind, colptr, prold, prnew, 0, NODES, 0);

        // Apply damping, teleportation, redistribution, compute norm, and update prold
        norm_sq = 0.0;
        pagerank_update_and_norm_range(prnew, prold, 0, NODES, redistribution,
                                       DAMPING, NODES, &norm_sq);

        norm = sqrt(norm_sq);

    } while (norm > ERR);

    double t_compute_end = get_time();
    double compute_time = t_compute_end - t_compute_start;
    double total_time = t_compute_end - t_setup_start;

    // Print execution summary using measure library
    measure_print_summary("SEQUENTIAL BASELINE", setup_time, compute_time, total_time);

    // Validate PageRank sum equals 1.0
    double sum_pr = pagerank_validate(NODES, prnew);
    printf("VERIFICA MATEMATICA: Somma finale PR = %.10f\n", sum_pr);
    printf("=============================================\n");

    /*
    for(i = 0; i < NODES; i++) {
        printf("Node %d: PageRank = %.10f\n", i, prnew[i]);
    }
    */

    // Save sequential time to file for later comparison with parallel versions
    FILE *time_file = fopen("sequential/sequential_time.txt", "w");
    if (time_file) {
        fprintf(time_file, "%.10f\n", compute_time);
        fprintf(time_file, "%.10f\n", setup_time);
        fclose(time_file);
        printf("Sequential time saved to 'sequential_time.txt'\n");
    }

    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(sum);

    return 0;
}