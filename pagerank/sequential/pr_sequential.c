#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

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

    // Damping arrays
    double *damp1 = (double *) malloc(NODES * sizeof(double));
    double *damp2 = (double *) malloc(NODES * sizeof(double));

    double *diff = (double *) malloc(NODES * sizeof(double));

    double norm, norm_sq;

    int col, i, j;

    int iteration_count = 0;

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
    for(i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = 0.85;
        damp2[i] = 0.15 / NODES;
    }

    double t_setup_end = get_time();
    double setup_time = t_setup_end - t_setup_start;

    printf("Setup up complete\n");

    // Power iteration
    double t_compute_start = get_time();
    
    do {
        iteration_count++;
        memset(prnew, 0, NODES * sizeof(double));

        // Compute dangling mass and redistribution
        double dangling_mass = pagerank_compute_dangling_mass(NODES, prold, sum);

        norm = 0.0;
		for(col = 0; col<NODES; col++) {
			for(j=colptr[col]; j<colptr[col+1]; j++) {
				prnew[rowind[j]] += val[j]*prold[col];
			}
		}

		for(i = 0; i<NODES; i++) {
			prnew[i] = prnew[i]*damp1[i]+damp2[i] + ((dangling_mass * damp1[i]) / NODES);
		}

        //norm calculation and vector copying from new to old
		norm_sq = 0.0;
		for(i=0; i<NODES;i++) {
			diff[i] = prnew[i] - prold[i];
			norm_sq += diff[i]*diff[i]; //l2 norm || prnew-prold ||
			prold[i] = prnew[i];
		}

	    norm = sqrt(norm_sq);

    } while (norm > ERR);

    double t_compute_end = get_time();
    double compute_time = t_compute_end - t_compute_start;
    double total_time = t_compute_end - t_setup_start;

    // Print execution summary using measure library
    measure_print_summary("SEQUENTIAL BASELINE", setup_time, compute_time, total_time, iteration_count);

    // Validate PageRank sum equals 1.0
    double sum_pr = 0.0;
    for (i = 0; i < NODES; i++) {
        sum_pr += prnew[i];
    }

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
    free(damp1);
    free(damp2);
    free(diff);

    return 0;
}