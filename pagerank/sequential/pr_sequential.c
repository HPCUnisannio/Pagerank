// Command to compile: 
// gcc sequential/pr_sequential.c libraries/data.c libraries/measure.c -o sequential/pr_sequential -Ilibraries -lm
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

    // === NUOVO PARSER SEQUENZIALE ROBUSTO PER GESTIONE DEI POZZI ===
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
            sum[colmatch] = localsum; // 'sum' nel sequenziale tiene traccia degli out-degree
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
    // ==========================================================


    index = 0;
    for (i = 0; i < NODES; i++) {
        co = sum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }
/*
   printf("\n======================= CSC construction complete ==========================\n");
    printf("\n--- VERIFICA COSTRUZIONE CSC ---\n");
    printf("COLPTR: ");
    for (int c = 0; c <= NODES; c++) printf("%d ", colptr[c]);
    printf("\nROWIND: ");
    for (int c = 0; c < EDGES; c++) printf("%d ", rowind[c]);
    printf("\nVAL:    ");
    for (int c = 0; c < EDGES; c++) printf("%.2f ", val[c]);
    printf("\n============================================================================\n");
*/

//===================================================================================================================
   // POWER ITERATION (Calcolo del PageRank)
   //===================================================================================================================
   double start_time = get_time();
   do {
        /* 1. Reset del vettore per la nuova iterazione */
        memset(prnew, 0, NODES * sizeof(double));
        norm = 0.0;

        /* * ---------------------------------------------------------------------
         * 2. GESTIONE DEI DANGLING NODES (Nodi Pozzo)
         * ---------------------------------------------------------------------
         * Un "dangling node" è un nodo senza link uscenti. Nella moltiplicazione
         * matrice-vettore standard, il PageRank di questo nodo non viene trasferito
         * a nessuno, causando una "perdita di massa" globale (la somma totale
         * del vettore scende sotto 1.0).
         * Per risolvere, raccogliamo tutta questa probabilità "persa" per poi
         * redistribuirla equamente a tutta la rete.
         */
        double dangling_mass = 0.0;
        for (col = 0; col < NODES; col++) {
            // L'array 'sum' contiene l'out-degree (numero di link uscenti) calcolato prima.
            // Se è 0, siamo di fronte a un nodo pozzo.
            if (sum[col] == 0) {
                // Accumuliamo il suo PageRank attuale nella massa totale da redistribuire
                dangling_mass += prold[col];
            }
        }

        /* * Calcoliamo la quota esatta che ogni nodo riceverà dalla massa persa.
         * Moltiplichiamo per DAMPING perché questa massa segue la regola dei link
         * (viene cliccata con probabilità d) e la dividiamo per il numero totale di nodi (N).
         */
        double redistribution = (dangling_mass * DAMPING) / NODES;


        /* * ---------------------------------------------------------------------
         * 3. MOLTIPLICAZIONE MATRICE-VETTORE SPARSA
         * ---------------------------------------------------------------------
         * Calcoliamo il trasferimento di PageRank attraverso i link reali.
         * Si usa il formato CSC per scorrere solo gli archi effettivamente esistenti.
         */
        for (col = 0; col < NODES; col++) {
            for (j = colptr[col]; j < colptr[col + 1]; j++) {
                prnew[rowind[j]] += val[j] * prold[col];
            }
        }


        /* * ---------------------------------------------------------------------
         * 4. AGGIORNAMENTO FINALE: Damping, Teleportation e Redistribuzione
         * ---------------------------------------------------------------------
         * Ora assembliamo l'equazione completa per ogni nodo 'i':
         * prnew[i] = (Voti dai link * Damping) + Probabilità di salto casuale + Quota nodi pozzo
         *
         * - prnew[i] * damp1[i]: Applica il damping (0.85) ai voti ricevuti.
         * - damp2[i]: Aggiunge il teleporting (0.15 / N).
         * - redistribution: Aggiunge la quota recuperata dai dangling nodes.
         */
        for (i = 0; i < NODES; i++) {
            prnew[i] = (prnew[i] * damp1[i]) + damp2[i] + redistribution;
        }


        /* * ---------------------------------------------------------------------
         * 5. CALCOLO DELL'ERRORE E AGGIORNAMENTO VETTORE
         * ---------------------------------------------------------------------
         * Calcoliamo la Norma L2 (distanza euclidea) tra il nuovo vettore e il vecchio.
         * Contemporaneamente, copiamo i nuovi valori in 'prold' per prepararci
         * al prossimo giro.
         */
        norm_sq = 0.0;
        for (i = 0; i < NODES; i++) {
            diff[i]  = prnew[i] - prold[i];
            norm_sq += diff[i] * diff[i];
            prold[i] = prnew[i]; // Aggiorna il vettore vecchio con il nuovo
        }

        // Radice quadrata della somma dei quadrati delle differenze
        norm = sqrt(norm_sq);

    } while (norm > ERR); // Continua finché la differenza è maggiore di 0.00001

    double end_time = get_time();
    double tempo_sequenziale = end_time - start_time;

   printf("\n=============================================\n");
   printf("TEMPO DELLA POWER ITERATION SEQUENZIALE: %.6f secondi\n", tempo_sequenziale);
   printf("=============================================\n");

   double sum_pr = 0;
   for(i = 0; i < NODES; i++) sum_pr += prnew[i];

   printf("VERIFICA MATEMATICA: Somma finale PR = %.10f\n", sum_pr);
   printf("=============================================\n");

    /*
    printf("\n--- VETTORE PAGERANK FINALE ---\n");
    for (i = 0; i < NODES; i++) {
        printf("Nodo %d: %.6f\n", i + 1, prnew[i]);
    }
    printf("=============================================\n");
*/

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