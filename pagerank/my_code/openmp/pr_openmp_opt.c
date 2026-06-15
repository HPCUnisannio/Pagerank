/*
================================================================================
# 🏆 APPROCCIO C: PRIVATIZZAZIONE MEMORIA (ZERO-ATOMIC)
================================================================================
Questa è la versione più spinta in assoluto. Elimina completamente la contesa
di memoria ('atomic') assegnando a ogni thread un proprio buffer privato per i
risultati parziali, per poi fonderli in una fase di riduzione esplicita.

| FASE DEL CICLO             | MECCANISMO                                         | VANTAGGIO HPC                           |
| :------------------------- | :------------------------------------------------- | :-------------------------------------- |
| 0. Allocazione Privata     | 'local_prnew[NUM_THREADS][NODES]' allocato in RAM  | Ogni thread ha il suo spazio esclusivo. |
| 1. Moltiplicazione         | Scrittura su 'local_prnew[tid]' (NIENTE ATOMIC)    | Massima velocità hardware, zero stalli. |
| 2. Fusione e Damping       | I thread si dividono i NODI e sommano le colonne   | Località della cache massimizzata.      |

💡 NOTE TECNICHE:
Abbiamo convertito una "dipendenza da scrittura" (conflitto sui nodi destinazione)
in un uso extra di memoria (Memory-Compute Tradeoff). Scambiamo RAM per velocità.
================================================================================




Notiamo che con un numero minore di thread che settiamo noi programmaticamente omp_set_num_threads(4);
otteniamo maggiore velocità di esecuzione.
/*
================================================================================
# 📊 ANALISI DELLE PRESTAZIONI: IL FENOMENO DEL "MEMORY WALL"
================================================================================
OSSERVAZIONE EMPIRICA:
In questa versione con privatizzazione della memoria (Approccio C), si nota
un comportamento controintuitivo: RIDUCENDO il numero di thread, le prestazioni
MIGLIORANO, mentre usando il massimo numero di core disponibili il tempo di
esecuzione subisce un degrado o non scala linearmente.

MOTIVAZIONE TECNICA (Da Compute-Bound a Memory-Bound):
Rimuovendo le istruzioni '#pragma omp atomic', abbiamo eliminato le contese
sulla CPU, ma abbiamo spostato il collo di bottiglia sul bus della RAM.
L'algoritmo non è più limitato dalla potenza di calcolo, ma dalla velocità
con cui la memoria riesce a fornire i dati (Memory Bandwidth).

| CAUSA PRINCIPALE         | DESCRIZIONE DEL COLLO DI BOTTIGLIA                   |
| :----------------------- | :--------------------------------------------------- |
| 1. Overhead di Riduzione | La fusione finale richiede 'T' letture per ogni nodo |
|                          | (dove T = num_threads). Con T=16 e 685k nodi, la CPU |
|                          | deve fare ~11 milioni di accessi RAM per iterazione! |
| 2. Saturazione del Bus   | Troppi thread richiedono vettori enormi nello stesso |
|                          | istante, creando un "ingorgo" sul bus della RAM.     |
| 3. Cache Thrashing       | La memoria locale di T thread eccede la capacità     |
|                          | della Cache L3. La CPU è costretta a scartare e      |
|                          | ricaricare i dati dalla RAM continuamente.           |

💡 CONCLUSIONE E SWEET SPOT:
In algoritmi legati alla banda di memoria, l'aggiunta di thread segue una
curva a "U". Il tempo ottimale si ottiene identificando lo "Sweet Spot":
un numero di thread (solitamente inferiore ai core fisici) che bilancia
la parallelizzazione del calcolo senza saturare la larghezza di banda RAM.
================================================================================

*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "data.h"

#ifdef _WIN32
    #include <windows.h>
    double get_time() {
       LARGE_INTEGER t, f;
       QueryPerformanceCounter(&t);
       QueryPerformanceFrequency(&f);
       return (double)t.QuadPart / f.QuadPart;
    }
#else
#include <time.h>
double get_time() {
       struct timespec t;
       clock_gettime(CLOCK_MONOTONIC, &t);
       return t.tv_sec + t.tv_nsec / 1e9;
    }
#endif
#define DAMPING 0.85
#define err     0.00001

double sqrt(double x);

int main(int argc, char *argv[])
{

    printf("Program start\n");

    omp_set_num_threads(4);

    GraphType graph_type = GRAPH_BIGGEST;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    FILE *fp;
    int colindex, link, i, j = 0, col, colmatch = 0, localsum = 0;
    int co, index;

    double *val    = (double *) calloc(EDGES,       sizeof(double));
    int    *rowind = (int *)    calloc(EDGES,        sizeof(int));
    int    *colptr = (int *)    calloc(NODES + 1,    sizeof(int));

    double *prold = (double *) malloc(NODES * sizeof(double));
    double *prnew = (double *) calloc(NODES,  sizeof(double));
    double *damp1 = (double *) malloc(NODES * sizeof(double));
    double *damp2 = (double *) malloc(NODES * sizeof(double));
    double *diff  = (double *) calloc(NODES,  sizeof(double));

    /* ── variabili per la norma ────────────────────────────────
       norm_sq → somma dei quadrati delle differenze: Σ(prnew[i]-prold[i])²
       norm    → norma L2 = √norm_sq, usata come criterio di convergenza
    */
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

    for (i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Errore lettura file alla riga %d\n", i + 1);
            fclose(fp);
            return 1;
        }

        colindex = colindex - 1;
        link     = link - 1;
        rowind[i] = link;

        if (colmatch == colindex) {
            localsum += 1;
        } else {
            sum[j]       = localsum;
            colptr[j + 1] = colptr[j] + localsum;
            localsum     = 1;
            j           += 1;
            colmatch     = colindex;
        }

        val[i] = 1.0;
    }

    sum[j]      = localsum;
    colptr[j + 1] = EDGES;
    fclose(fp);

    index = 0;
    for (i = 0; i < NODES; i++) {
        co = sum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }

   for (col = 0; col < NODES; col++) {
      double col_sum = 0.0;
      for (int k = colptr[col]; k < colptr[col + 1]; k++) {
         col_sum += val[k];
      }
   }
   printf("\n");

   printf("CSC construction complete\n");



    //===========================================================================
    // PREPARAZIONE: ALLOCAZIONE DELLA MEMORIA PRIVATA PER THREAD
    //===========================================================================
    int num_threads = omp_get_max_threads();
    double **local_prnew = (double **)malloc(num_threads * sizeof(double *));
    for (int t = 0; t < num_threads; t++) {
        local_prnew[t] = (double *)calloc(NODES, sizeof(double));
    }

    //===========================================================================
    // MISURO IL TEMPO DI ESECUZIONE (OpenMP ha il suo timer nativo!)
    //===========================================================================
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        // Otteniamo l'ID del thread e dichiariamo gli indici come puramente locali
        int tid = omp_get_thread_num();
        int i_local, j_local, col_local, t_local;

        do {
            /* 1. AZZERAMENTO PRIVATO (Senza barriere)
               Ogni thread azzera solo e unicamente il suo array.
            */
            for (i_local = 0; i_local < NODES; i_local++) {
                local_prnew[tid][i_local] = 0.0;
            }

            /* 2. MOLTIPLICAZIONE ZERO-ATOMIC
               Torniamo allo scheduling statico nativo di OpenMP.
               Nessun conflitto: ogni thread scrive nel proprio local_prnew.
            */
            #pragma omp for
            for (col_local = 0; col_local < NODES; col_local++) {
                for (j_local = colptr[col_local]; j_local < colptr[col_local + 1]; j_local++) {
                    local_prnew[tid][rowind[j_local]] += val[j_local] * prold[col_local];
                }
            }

            /* Un solo thread azzera norm_sq, gli altri non lo aspettano (nowait) */
            #pragma omp single nowait
            {
                norm_sq = 0.0;
            }

            /* 3 & 4. FUSIONE, DAMPING E NORMA (Parallelizzazione sui nodi)
               I thread leggono in colonna i risultati parziali degli altri thread.
            */
            #pragma omp for reduction(+:norm_sq)
            for (i_local = 0; i_local < NODES; i_local++) {
                
                double sum_local = 0.0;
                for (t_local = 0; t_local < num_threads; t_local++) {
                    sum_local += local_prnew[t_local][i_local];
                }
                
                prnew[i_local] = sum_local * damp1[i_local] + damp2[i_local];
                
                diff[i_local]  = prnew[i_local] - prold[i_local];
                norm_sq += diff[i_local] * diff[i_local];
                prold[i_local] = prnew[i_local];
            }

            /* 5. AGGIORNAMENTO NORMA FINALE */
            #pragma omp single
            {
                norm = sqrt(norm_sq);
            }

        } while (norm > err);
    } // Fine regione parallela

    double end_time = omp_get_wtime();

    double tempo_parallelo = end_time - start_time;

    printf("\n=============================================\n");
    printf("TEMPO POWER ITERATION OpenMP (PRIVATIZZATA): %.6f secondi\n", tempo_parallelo);
    printf("=============================================\n");


//    printf("\n=== PAGERANK FINALE ===\n");
//    for (j = 0; j < NODES; j++) {
//        printf("Nodo %d: %f\n", j + 1, prnew[j]);
//    }


   double sum_pr=0;
   for(int i=0;i<NODES;i++) sum_pr+=prnew[i];
   printf("Somma PR = %.10f\n", sum_pr);


    // Liberazione memoria di base
    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(damp1);
    free(damp2);
    free(diff);
    free(sum);

    // Liberazione memoria privata
    for (int t = 0; t < num_threads; t++) {
        free(local_prnew[t]);
    }
    free(local_prnew);

    return 0;
}