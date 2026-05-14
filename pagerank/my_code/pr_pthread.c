#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define NODES 6
#define EDGES 19
#define CORES 3
#define MASTER 0
#define FILEPATH "../dataset/data0.dat"
double sqrt(double x);
void *mat_vec(void *);
double *val, *prold, *prnew, *damp1, *damp2, *diff;
int *rowind, *colptr, *sum;
pthread_mutex_t add_mutex;
pthread_mutex_t wait_mutex;
pthread_mutex_t norm_mutex;          // NUOVO: protegge accesso a norm
pthread_barrier_t our_barrier;
pthread_barrier_t our_barrier2;      // NUOVO: seconda barriera post-calcolo
pthread_cond_t proceed_cv;
double norm, norm_sq, err = 0.000001;
int var_wait;

int main(int argc, char *argv[])
{
    printf("Program start\n");
    pthread_t p_threads[CORES];
    pthread_mutex_init(&add_mutex, NULL);
    pthread_mutex_init(&wait_mutex, NULL);
    pthread_mutex_init(&norm_mutex, NULL);           // NUOVO
    pthread_barrier_init(&our_barrier, NULL, CORES);
    pthread_barrier_init(&our_barrier2, NULL, CORES); // NUOVO
    pthread_cond_init(&proceed_cv, NULL);

    FILE *fp;
    int colindex, link, i, j=0, col, colmatch=0, localsum=0;
    long t;
    int k, rc=0;

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

    for(i=0; i<NODES; i++) {
        //prold[i] = 0.25;
        prold[i] = 1.0 / NODES;
        damp1[i] = 0.85;
        damp2[i] = 0.15/NODES;
    }
    printf("initialization complete\n");

    norm = 1.0;
    var_wait = 0;

    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        return 1;
    }

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

    index = 0;
    for(i = 0; i<NODES; i++) {
        co = sum[i];
        for(j = index; j < index+co; j++) {
            val[j] = val[j]/co;
        }
        index += co;
    }

    printf("\n=== STRUTTURA CSC COSTRUITA (DEBUG) ===\n");
    printf("colptr: [");
    for (i = 0; i <= NODES; i++) {
        printf("%d", colptr[i]);
        if (i < NODES) printf(", ");
    }
    printf("]\n\n");

    printf("rowind: [");
    for (i = 0; i < EDGES; i++) {
        printf("%d", rowind[i]);
        if (i < EDGES - 1) printf(", ");
    }
    printf("]\n\n");

    printf("val (normalizzati): [");
    for (i = 0; i < EDGES; i++) {
        printf("%.6f", val[i]);
        if (i < EDGES - 1) printf(", ");
    }
    printf("]\n\n");

    printf("Somma per colonna (verifica normalizzazione):\n");
    for (col = 0; col < NODES; col++) {
        double col_sum = 0.0;
        for (int k = colptr[col]; k < colptr[col + 1]; k++) {
            col_sum += val[k];
        }
        printf("  Colonna %d (nodo %d): somma = %.10f %s\n",
               col, col, col_sum,
               (fabs(col_sum - 1.0) < 0.0001) ? "✓" : "✗ ERRORE");
    }
    printf("\n");

    printf("\n");

    // Thread creation
    for(t=0; t<CORES; t++) {
        printf("\nIn main: creating thread %ld\n", t);
        pthread_create(&p_threads[t], NULL, mat_vec, (void*)t);
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for(t= 0; t<CORES; t++) {
        pthread_join(p_threads[t], NULL);
    }

    /* AGGIUNTA DI DEBUG */
    // Print formattato come versione serial
    printf("\n=== PAGERANK FINALE (PTHREAD - %d THREAD) ===\n", CORES);
    for(j = 0; j < NODES; j++) {
        printf("Nodo %d: %f\n", j + 1, prold[j]);
    }

    // Somma e validazione
    double sum_pr = 0.0;
    for(int i=0; i<NODES; i++) {
        sum_pr += prold[i];
    }
    printf("\nSomma PR = %.10f\n", sum_pr);


    printf("Printing pagerank vector (CONVERGED)\n");
    for(j=0; j<NODES; j++) {
        printf("%f\t", prold[j]);  // ← CORRETTO: Stampa prold (valori finali), non prnew
    }
    printf("\n");

    /* Clean up and exit */
    pthread_mutex_destroy(&add_mutex);
    pthread_mutex_destroy(&wait_mutex);
    pthread_mutex_destroy(&norm_mutex);      // NUOVO
    pthread_barrier_destroy(&our_barrier);
    pthread_barrier_destroy(&our_barrier2);  // NUOVO
    pthread_cond_destroy(&proceed_cv);
    pthread_exit(NULL);
}

// Power iteration - Calcolo PageRank con sincronizzazione corretta
void *mat_vec(void *rank) {
    long tid = (long)rank;
    printf("\n%ld Reached thread function\n", tid);

    int i, j, col, local_col, my_first_col, my_last_col;
    double *localpr = (double*)calloc(NODES, sizeof(double));
    double current_norm;  // DICHIARATA QUI - Prima del loop
    printf("my thread id is %ld\n", tid);

    if (tid==0) {
        local_col = NODES/CORES + NODES%CORES;
        my_first_col = 0;
        my_last_col = my_first_col + local_col - 1;
    }
    else {
        local_col = NODES/CORES;
        my_first_col = tid*local_col + NODES%CORES;
        my_last_col = my_first_col + local_col - 1;
        printf("this is %ld\t %d\n", tid, my_last_col);
    }

    printf("reached do\n");
    printf("%d\t%d\n", my_first_col, my_last_col);

    do {
        // Passo 1: Ogni thread calcola il suo contributo a localpr
        for(col=my_first_col; col<=my_last_col; col++) {
            for(j = colptr[col]; j<colptr[col+1]; j++) {
                localpr[rowind[j]] += val[j]*prold[col];
            }
        }

        // Passo 2: Somma thread-safe dei contributi in prnew
        for(i=0; i<NODES; i++) {
            if (localpr[i]!=0.0) {
                pthread_mutex_lock(&add_mutex);
                prnew[i] += localpr[i];
                pthread_mutex_unlock(&add_mutex);
            }
        }

        // Passo 3: BARRIERA 1 - Sincronizza tutti i thread
        printf("Thread %ld reached barrier 1\n", tid);
        pthread_barrier_wait(&our_barrier);

        // Passo 4: MASTER thread calcola norm e aggiorna, altri aspettano
        if (rank==MASTER) {
            // Calcola norm sotto protezione di mutex
            pthread_mutex_lock(&norm_mutex);

            var_wait = 0;
            norm_sq = 0.0;
            for(i=0; i<NODES; i++) {
                // Applica damping factor
                prnew[i] = prnew[i] * damp1[i] + damp2[i];

                // Calcola differenza per convergenza
                diff[i] = prnew[i] - prold[i];
                norm_sq += diff[i]*diff[i];

                // Salva nuovo valore come vecchio per la prossima iterazione
                prold[i] = prnew[i];
            }

            // Calcola norma per test di convergenza
            norm = sqrt(norm_sq);
            printf("MASTER: norm = %f, err = %f\n", norm, err);

            pthread_mutex_unlock(&norm_mutex);

            // Ricomincia prnew per prossima iterazione
            memset(prnew, 0, NODES*sizeof(double));

            // Segnala ai thread worker di procedere
            var_wait = 1;
            pthread_cond_broadcast(&proceed_cv);
        }
        else {
            // WORKER threads aspettano il calcolo del MASTER
            pthread_mutex_lock(&wait_mutex);
            while(!var_wait) {
                pthread_cond_wait(&proceed_cv, &wait_mutex);
            }
            pthread_mutex_unlock(&wait_mutex);
        }

        // Passo 5: Resetta localpr per ogni thread (ora fatto localmente)
        memset(localpr, 0, NODES*sizeof(double));

        // Passo 6: BARRIERA 2 - CRITICA! Sincronizza prima di leggere norm
        // =========================================================
        // QUESTA È LA CHIAVE DELLA SOLUZIONE!
        // Assicura che TUTTI i thread abbiano finito e leggano norm coerentemente
        printf("Thread %ld reached barrier 2\n", tid);
        pthread_barrier_wait(&our_barrier2);

        // Passo 7: Leggi norm in modo thread-safe
        pthread_mutex_lock(&norm_mutex);
        current_norm = norm;
        pthread_mutex_unlock(&norm_mutex);

        printf("Thread %ld checking: current_norm = %f, err = %f\n", tid, current_norm, err);

        // Passo 8: Verifica condizione di convergenza (THREAD-SAFE)
        // =========================================================
        // ORA tutti i thread leggono lo STESSO norm !
        // Non c'è più race condition!

    } while(current_norm > err);  // CORRETTO: Lettura sincronizzata di norm

    printf("Thread %ld: Exiting (norm=%f <= err=%f)\n", tid, norm, err);
    free(localpr);
    pthread_exit(NULL);
}

