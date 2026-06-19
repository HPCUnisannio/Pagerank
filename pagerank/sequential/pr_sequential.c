#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"

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

    FILE *fp;
    int colindex, link, i, j = 0, col, c, colmatch = 0, localsum = 0;
    int co, index;

    double *val    = (double *) calloc(EDGES,       sizeof(double));
    int    *rowind = (int *)    calloc(EDGES,        sizeof(int));
    int    *colptr = (int *)    calloc(NODES + 1,    sizeof(int));

    double *prold = (double *) malloc(NODES * sizeof(double));
    double *prnew = (double *) calloc(NODES,  sizeof(double));
    double *damp1 = (double *) malloc(NODES * sizeof(double));
    double *damp2 = (double *) malloc(NODES * sizeof(double));
    double *diff  = (double *) calloc(NODES,  sizeof(double));

    double norm, norm_sq;

    int *sum = (int *) calloc(NODES, sizeof(int));

    if (!val || !rowind || !colptr || !prold || !prnew ||
        !damp1 || !damp2 || !diff || !sum) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        return 1;
    }

    // Initialize vectors
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = DAMPING;
        damp2[i] = (1.0 - DAMPING) / NODES;
    }
    printf("initialization complete\n");

    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        return 1;
    }

    // CSC matrix construction with dangling nodes support
    localsum = 0;
    colmatch = -1;

    for (i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Errore lettura file alla riga %d\n", i + 1);
            fclose(fp);
            return 1;
        }

        colindex = colindex - 1;
        link     = link - 1;
        rowind[i] = link;

        if (i == 0) {
            colmatch = colindex;
            localsum = 1;
        } else if (colmatch == colindex) {
            localsum += 1;
        } else {
            sum[colmatch] = localsum;
            for (c = colmatch + 1; c <= colindex; c++) {
                colptr[c] = colptr[colmatch] + localsum;
            }
            localsum = 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }

    if (EDGES > 0) {
        sum[colmatch] = localsum;
        for (c = colmatch + 1; c <= NODES; c++) {
            colptr[c] = EDGES;
        }
    }
    fclose(fp);

    // Column normalization
    index = 0;
    for (i = 0; i < NODES; i++) {
        co = sum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }

    // Power iteration
    double start_time = get_time();
    do {
        memset(prnew, 0, NODES * sizeof(double));
        norm = 0.0;

        // Compute dangling mass from nodes with no outgoing links
        double dangling_mass = 0.0;
        for (col = 0; col < NODES; col++) {
            if (sum[col] == 0) {
                dangling_mass += prold[col];
            }
        }

        double redistribution = (dangling_mass * DAMPING) / NODES;

        // Sparse matrix-vector multiplication
        for (col = 0; col < NODES; col++) {
            for (j = colptr[col]; j < colptr[col + 1]; j++) {
                prnew[rowind[j]] += val[j] * prold[col];
            }
        }

        // Apply damping, teleportation and dangling redistribution
        for (i = 0; i < NODES; i++) {
            prnew[i] = (prnew[i] * damp1[i]) + damp2[i] + redistribution;
        }

        // Compute L2 norm and update old vector
        norm_sq = 0.0;
        for (i = 0; i < NODES; i++) {
            diff[i]  = prnew[i] - prold[i];
            norm_sq += diff[i] * diff[i];
            prold[i] = prnew[i];
        }

        norm = sqrt(norm_sq);

    } while (norm > ERR);

    double end_time = get_time();
    double tempo_sequenziale = end_time - start_time;

    printf("\n=============================================\n");
    printf("TEMPO DELLA POWER ITERATION SEQUENZIALE: %.6f secondi\n", tempo_sequenziale);
    printf("=============================================\n");

    // Validate PageRank sum equals 1.0
    double sum_pr = 0;
    for (i = 0; i < NODES; i++) sum_pr += prnew[i];

    printf("VERIFICA MATEMATICA: Somma finale PR = %.10f\n", sum_pr);
    printf("=============================================\n");

    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(damp1);
    free(damp2);
    free(diff);
    free(sum);

    return 0;
}