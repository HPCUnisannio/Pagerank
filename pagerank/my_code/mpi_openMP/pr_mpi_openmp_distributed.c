
/*
 * CONFIGURAZIONE CON REGIONE PARALLALE A GRANA GROSSA
 * POOL DI THREAD CREATO UNA SOLA VOLTA FUORI DA DO-WHILE
 * RIMOZIONE DI atomic E PRIVATIZZAZIONE DEL VETTORE sum
 * MIGLIORAMENTO MPI CON POST PROCESSING DISTRIBUITO TRA TUTTI I PROCESSI
 * ASINCRONIA PER OVERLAP TRA COMUNICAZIONE E CALCOLO DELLA DAMPLING MASS
 * RIMOZIONE DELLA DISTRIBUZIONE DELLE COLONNE TRA PROCESSI(RIDONDANTE)
 *
*/

/*
# PageRank Ibrido (MPI + OpenMP) - Versione Ottimizzata a Grana Grossa

Questa versione implementa il calcolo del PageRank distribuito su grafi diretti, combinando **MPI** per il calcolo distribuito tra i nodi e **OpenMP** per il parallelismo a memoria condivisa all'interno di ogni nodo.

L'architettura è fortemente orientata alle prestazioni, minimizzando l'overhead di sincronizzazione, sovrapponendo comunicazione e calcolo, ed eliminando i colli di bottiglia seriali.

---

## 🚀 Ottimizzazioni Chiave Implementate

1. **Regione Parallela a Grana Grossa (Coarse-Grained)**
   * Il costrutto `#pragma omp parallel` viene aperto **una sola volta** prima del ciclo `do-while`.
   * Il pool di thread sopravvive per tutta l'esecuzione delle iterazioni, azzerando completamente l'overhead di creazione e distruzione dei thread ad ogni ciclo. Le sincronizzazioni avvengono tramite barriere implicite o esplicite (`#pragma omp barrier`).

2. **Privatizzazione e Rimozione di `atomic`**
   * Durante la fase di SpMV (Sparse Matrix-Vector Multiplication), l'uso di `#pragma omp atomic` per evitare *race conditions* sui nodi di destinazione distruggerebbe le prestazioni.
   * È stato introdotto un array 2D `thread_sums` pre-allocato. Ogni thread accumula i risultati parziali sul proprio vettore privato. Al termine della SpMV, i vettori privati vengono ridotti in parallelo nel vettore globale `sum`.

3. **Asincronia e Overlap (Comunicazione / Calcolo)**
   * La somma globale dei vettori tra i processi MPI è affidata a una **`MPI_Iallreduce`** (non bloccante).
   * Mentre le schede di rete elaborano la riduzione e scambiano i dati, le CPU calcolano in parallelo la propria quota di *dangling mass* (la massa persa nei nodi pozzo). La `MPI_Wait` viene chiamata solo prima di applicare la redistribuzione, nascondendo efficacemente i tempi morti di latenza di rete.

4. **Post-Processing Distribuito Simmetrico al 100%**
   * Il calcolo del *damping factor*, la redistribuzione della massa, l'aggiornamento del vettore storico (`prold`) e il calcolo dell'errore (norma locale) non vengono più calcolati per intero da ogni processo.
   * Ogni rank MPI esegue il post-processing **esclusivamente sulla propria fetta di colonne** (`rec_col`). Questo garantisce una scalabilità perfetta della fase di update al crescere del numero di processi.

5. **Lettura e Distribuzione Dati I/O-Bound (Nessuna Scatter)**
   * È stata rimossa la fase di distribuzione dei dati via `MPI_Scatterv`. In questa specifica configurazione, ogni processo legge il dataset in modo indipendente e punta direttamente alla propria porzione in memoria tramite i vettori `displs` e `sendcnts`. Questo scambia un maggiore consumo di memoria/IO con un setup di rete immediato.

---

## ⚙️ Flusso di Esecuzione dell'Iterazione (Ciclo `do-while`)

All'interno della regione parallela fissa, il lavoro è diviso in 7 blocchi sequenziali orchestrati tra i thread OpenMP e il thread Master per le chiamate MPI:

* **Blocco 1 (Reset):** Azzeramento locale degli array privati dei thread e degli accumulatori.
* **Blocco 2 (SpMV):** Calcolo parallelo del prodotto matrice-vettore. Ogni thread legge le colonne assegnate al proprio processo e scrive sul proprio vettore di output privato.
* **Blocco 3 (Riduzione Thread):** I thread sommano i propri vettori parziali dentro il vettore `sum` del processo locale sfruttando la vettorizzazione (`#pragma omp for simd`).
* **Blocco 4 (Invio Rete):** Il thread Master avvia la `MPI_Iallreduce` asincrona per sommare i vettori `sum` di tutti i processi nel vettore `prnew`.
* **Blocco 5 (Overlap e Dangling Mass):** Mentre la rete lavora, i thread calcolano la *dangling mass* locale sui nodi di propria competenza. Segue una rapida `MPI_Allreduce` per sommare i valori e una barriera d'attesa (`MPI_Wait`) per garantire l'arrivo completo di `prnew`.
* **Blocco 6 (Update Distribuito):** Ogni processo aggiorna $PR_{new}$ applicando damping e redistribuzione **solo sui propri nodi**, per poi aggiornare in-place $PR_{old}$ e calcolare il delta di scarto al quadrato (Norma locale).
* **Blocco 7 (Norma Globale):** Somma delle norme al quadrato locali tramite `MPI_Allreduce`, calcolo della radice quadrata e verifica della condizione di arresto ($Norma \leq Errore$).

---

## 💻 Istruzioni di Esecuzione

Il programma legge automaticamente il numero di thread come primo argomento della riga di comando. Se non specificato, utilizzerà il numero massimo di thread disponibili sulla macchina.

**Compilazione (Esempio con CMake/Make):**
```bash
mpicc -O3 -fopenmp -o pagerank_hybrid pr_mpi_openmp_distributed.c -lm

*/


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>


#define NODES 685230
#define EDGES 7600595
#define FILEPATH "pagerank/dataset/data2.dat"

/*
 *PER RUNNARE PRIMA SETTARE NUMERO THREAD COME ARGOMENTO DA PASSARE AL MAIN
 *ESEMPIO 3 Processi MPI x 2 Thread
mpiexec -n 3 ".\cmake-build-debug\pr_mpi_openmp_distributed.exe" 2
*/

//#define FILEPATH "../pagerank/dataset/data2.dat"

/*
    * Per eseguire da riga di comando e settare più processi, ad esempio 4
    * mpiexec -n 4 ".\cmake-build-debug\pr_mpi_OpenMP4.exe"
*/
// path da usare quando runni da riga di comando



// PER TEST CORRETTEZZA
/*
#define NODES 6
#define EDGES 19
#define FILEPATH "pagerank/dataset/data0.dat"
*/

/*
#define NODES 4039
#define EDGES 176468
#define FILEPATH "pagerank/dataset/data1.dat"
*/

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char **argv)
{
    int NPROC, rank, num_threads;
    // Se l'utente passa un argomento extra, lo usiamo come numero di thread
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    } else
    {
        num_threads = omp_get_max_threads();
    }


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    // 2. Stampa delle informazioni di setup (Solo MASTER)
    if (rank == 0) { // Usa 'MASTER' se hai definito una macro per lo 0
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK IBRIDO [ MPI(post processing distribuito simmetrico) + OpenMP(grana grossa + privatizzazione + master thread) ]\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf("=============================================\n\n");
        // QUESTA RIGA FORZA LA STAMPA IMMEDIATA A SCHERMO
        fflush(stdout);
    }



    FILE *fp;
    int colindex, link, i, j = 0, col, colmatch = -1, localsum = 0;
    int co, index;

    // Allocazione strutture principali
    double * val = (double*)calloc(EDGES, sizeof(double));
    int * rowind =(int*)calloc(EDGES, sizeof(int));
    int * colptr = (int*)calloc(NODES + 1, sizeof(int));
    int * readsum = (int*)calloc(NODES, sizeof(int));

    double * prold = (double*)malloc(NODES * sizeof(double));
    double * prnew = (double*)calloc(NODES, sizeof(double));

    // double * damp1 = (double*)malloc(NODES * sizeof(double));
    //double * damp2 = (double*)malloc(NODES * sizeof(double));
    //double * diff = (double*)calloc(NODES, sizeof(double));
    // Al posto delle 2 malloc e del loop di inizializzazione
    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    double * sum = (double*)calloc(NODES, sizeof(double));

    int *sendcnts = malloc(sizeof(int) * NPROC);
    int *displs = malloc(sizeof(int) * NPROC);
    int *pcols = (int*)malloc(NPROC * sizeof(int));
    int *displs_pr = (int*)malloc(NPROC * sizeof(int));

    int rec_col;

    // Inizializzazione vettori
    for(i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
        //damp1[i] = DAMPING;
        //damp2[i] = (1.0 - DAMPING) / NODES;
    }

    if (rank == MASTER) {
        printf("Initialization complete\n");
    }

    // Lettura file
    fp = fopen(FILEPATH, "r");
    if (fp == NULL) {
        fprintf(stderr, "Rank %d - Errore: impossibile aprire il file '%s'\n", rank, FILEPATH);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- FIX 1: Lettura robusta a prova di "Dangling Nodes" ---
    localsum = 0;
    for(i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Rank %d - Errore lettura file alla riga %d\n", rank, i + 1);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        colindex = colindex - 1;
        link = link - 1;
        rowind[i] = link;

        if (i == 0) {
            colmatch = colindex;
            localsum = 1;
        } else if (colmatch == colindex) {
            localsum += 1;
        } else {
            readsum[colmatch] = localsum;
            for(int c = colmatch + 1; c <= colindex; c++) {
                colptr[c] = colptr[colmatch] + localsum;
            }
            localsum = 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }
    if (EDGES > 0) {
        readsum[colmatch] = localsum;
        for (int c = colmatch + 1; c <= NODES; c++) {
            colptr[c] = EDGES;
        }
    }
    fclose(fp);

    // Normalizzazione della matrice CSC
    index = 0;
    for(i = 0; i < NODES; i++) {
        co = readsum[i];
        for(j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }

    if (rank == MASTER) {
        printf("val, rowind and colptr have been populated\n");
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
    }

    // Calcolo della distribuzione delle colonne tra i processi
    for(i = 0; i < NPROC; i++) {
        if(i == 0) {
            pcols[i] = NODES / NPROC + NODES % NPROC;
            displs_pr[i] = 0;
        } else {
            pcols[i] = NODES / NPROC;
            displs_pr[i] = pcols[i-1] + displs_pr[i-1];
        }
    }

    // Calcolo dei conteggi di elementi non-zero (sendcnts) e relativi spiazzamenti (displs)
    j = 0;
    for(i = 0; i < NPROC; i++) {
        j = j + pcols[i];
        int k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0) {
            displs[i] = 0;
        } else
            displs[i] = sendcnts[i-1] + displs[i-1];
    }

    // Allocazione dei buffer di ricezione locali per lo Scatterv
    // double * rec_val = (double*)malloc(sendcnts[rank] * sizeof(double));
    // int * rec_row = (int*)malloc(sendcnts[rank] * sizeof(int));
    // double * rec_pr = (double*)malloc(pcols[rank] * sizeof(double));

    /* Le ScatterV sono ridondanti in quanto già ogni processo possiede la porzione di dati su cui deve operare
    // Distribuzione dei dati
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE,
        rec_val, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pcols, 1, MPI_INT,
        &rec_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT,
        rec_row, sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    */

    // Configurazione senza distribuzione dei dati
    // Copia locale: ogni processo prende la sua fetta da val e rowind (già in memoria)
    // Inutile fare la distribuzione
    /*
    memcpy(rec_val, val + displs[rank], sendcnts[rank] * sizeof(double));
    rec_col = pcols[rank];                     // già calcolato da tutti
    memcpy(rec_row, rowind + displs[rank], sendcnts[rank] * sizeof(int));
    */
    double *rec_val = val + displs[rank];
    int    *rec_row = rowind + displs[rank];
    rec_col = pcols[rank];

    double begin = MPI_Wtime();


    // Variabili Shared tra i thread della successiva regione parallela
    double norm = 0.0; // norma globale
    // dangling mass locale (calcolata in parallelo da tutti i thread del processo, poi sommata con reduction)
    double dm_local = 0.0;
    // dangling mass globale per accumulare iterazione per iterazione i valori locali
    double dm_global = 0.0;
    // norma quadrata locale(calcolata in parallelo da tutti i thread del processo, poi sommata con reduction)
    double norm_sq_local = 0.0;


    // VARIABILI PER MISURAZIONE TEMPO COMUNICAZIONE VS CALCOLO
    double t_spmv = 0.0, t_thread_red = 0.0, t_allreduce_dm = 0.0;
    double t_allreduce_prnew = 0.0, t_update = 0.0, t_norm = 0.0;

    // --- CONFIGURAZIONE STRUTTURE PER PRIVATIZZAZIONE OPENMP (No MPI) ---
    // Creiamo un array di puntatori condiviso: conterrà il vettore sum di ogni thread
    //int actual_threads = num_threads;
    double **thread_sums = (double**)calloc(num_threads,  sizeof(double*));
    // ============================================================================================================
    // REGIONE PARALLELA A GRANA GROSSA --> CREIAMO QUI UNA SOLA VOLTA IL POOL DI THREAD
    // ============================================================================================================
#pragma omp parallel private(i, j)
    {

        // Ogni thread alloca una volta sola il proprio vettore sum privato sulla Heap
        int tid = omp_get_thread_num();
        thread_sums[tid] = (double*)calloc(NODES, sizeof(double));
        double *local_sum = thread_sums[tid];

        // t_phase è privata per thread, ma solo thread 0 la usa in blocchi master
        double t_phase = 0.0;

        // ============================================================================================================
        // REGIONE PARALLELA A GRANA GROSSA (Versione Corretta e Standard-Compliant)
        // ============================================================================================================
        do
        {
            // ----- BLOCCO 1: reset locale ------------------------------
            memset(local_sum, 0, NODES * sizeof(double));

            // reset delle variabili locali per il calcolo della dangling mass e della norma viene fatto
            // da un solo thread, il master, gli altri vedranno quindi i valori azzerati essendo queste
            // variabili condivise
#pragma omp master
            {
                dm_local = 0.0;
                norm_sq_local = 0.0;
            }


            // --- BLOCCO 2: SpMV CALCOLO PARALLELO MATRICE x VETTORE ---
            // Ogni processo usa prold[global_col] direttamente invece di rec_pr che riceveva —> no Scatterv
            int global_col_start = displs_pr[rank];

            // master per test temporale
#pragma omp master
            { t_phase = MPI_Wtime(); }

#pragma omp for schedule(dynamic, 512)
            for (int local_col = 0; local_col < rec_col; local_col++) {
                int global_col = global_col_start + local_col;
                int start_idx = colptr[global_col] - displs[rank];
                int end_idx   = colptr[global_col + 1] - displs[rank];

                for (j = start_idx; j < end_idx; j++) {
                    //local_sum[rec_row[j]] += rec_val[j] * rec_pr[local_col];
                    local_sum[rec_row[j]] += rec_val[j] * prold[global_col]; // usiamo direttamente prold
                }
            } // [BARRIERA 1 Implicita]
            // I thread devono aspettare che tutti abbiano finito di scrivere nei propri local_sum
            // prima di procedere alla riduzione

            // master per test temporale
#pragma omp master
            { t_spmv += MPI_Wtime() - t_phase; }


            // --- BLOCCO 3: RIDUZIONE PARALLELA DEI THREAD SUM -> SUM GLOBALE ---

            //master per test temporale
#pragma omp master
{ t_phase = MPI_Wtime(); }

#pragma omp for simd
            for(i = 0; i < NODES; i++)
            {
                double total_row_sum = 0.0;
                for(int t = 0; t < num_threads; t++) {
                    if(thread_sums[t] != NULL)
                        total_row_sum += thread_sums[t][i];
                }
                sum[i] = total_row_sum;
            } // [BARRIERA 2 Implicita]

            // master per test temporale
#pragma omp master
            { t_thread_red += MPI_Wtime() - t_phase; }

            // --- BLOCCO 4: RIDUZIONE MPI COLLETTIVA ASINCRONA) ---
            MPI_Request request;
#pragma omp master
            {
                t_phase = MPI_Wtime();  // inizia a contare da qui
                // Facciamo una riduzione globale in background in modo tale che tutti ricevano prnew "grezzo"
                MPI_Iallreduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
            }
            // Non ho bisogno di barriera in questo caso, i processi iniziano a computare la massa di pozzi

            // FASE DI POST PROCESSING DISTRIBUITA TRA TUTTI I PROCESSI
            // --- BLOCCO 5: OVERLAP CALCOLO DANGLING MASS PER GESTIRE I NODI POZZI
            // Calcoliamo la quota locale di dangling mass basandoci solo sulle NOSTRE colonne
#pragma omp for simd reduction(+:dm_local)
            for(i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;
                if (readsum[global_col] == 0) {
                    dm_local += prold[global_col];
                }
            } // [BARRIERA 3 Implicita]

            // Sommiamo le dangling mass locali di tutti i processi (Sincronizzazione scalare velocissima)
#pragma omp master
            {
                MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                //  Attendere che finisca la comunicazione
                // Ci assicuriamo che prnew sia stato completamente ricevuto in background
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                t_allreduce_prnew += MPI_Wtime() - t_phase;
                // Nota: t_allreduce_prnew misura il tempo TOTALE del blocco 4+5
                // cioè: max(Iallreduce, dm_locale+allreduce_dm) — il vero overlap
            }
#pragma omp barrier // [BARRIERA 4 Esplicita]
            // Dobbiamo aspettare che tutti i processi abbiamo prnew completo e la dangling mass globale,
            // poi possiamo procedere con l'aggiornamento e il calcolo della norma

            // redistribuzione privata per ogni thread di ogni processo
            double redistribution = dm_global * DAMPING / NODES;

            // --- BLOCCO 6: AGGIORNAMENTO VETTORE PR
            // Distribuzione totale: ogni processo cicla SOLO sulle sue colonne (rec_col)
            // Invece di fare NODES iterazioni, gestisce solo le colonne che ha trattato nel calcolo

            // master per test temporale
#pragma omp master
            { t_phase = MPI_Wtime(); }

            // damping + aggiornamento prold + norma
#pragma omp for simd reduction(+:norm_sq_local)
            for(i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;

                // Aggiorniamo prnew e prold solo per la nostra porzione
                prnew[global_col] = prnew[global_col] * DAMPING + (1.0 - DAMPING) / NODES + redistribution;

                double diff = prnew[global_col] - prold[global_col];
                norm_sq_local += diff * diff;

                prold[global_col] = prnew[global_col]; // Aggiornato in-place per la prossima SpMV
            } // [BARRIERA 5 Implicita]

            // master per test temporale
#pragma omp master
            { t_update += MPI_Wtime() - t_phase; }

            // --- BLOCCO 7: CALCOLO DELLA NORMA GLOBALE
#pragma omp master
{
    t_phase = MPI_Wtime();
    // Sommiamo le norme locali di tutti i processi (Sincronizzazione scalare velocissima)
    double norm_sq_global = 0.0;
    MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm = sqrt(norm_sq_global);
    t_norm += MPI_Wtime() - t_phase;
}
#pragma omp barrier //[BARRIERA 6 Esplicita] Attendere la norma per poter proseguire
            // Tutti i processi hanno ora la norma aggiornata, possono decidere se continuare o terminare l'iterazione



        } while(norm > ERROR);


        free(local_sum);
    }
    free(thread_sums);
    // ==============================================================================================================
    // FINE REGIONE PARALLELA A GRANA GROSSA
    // ==============================================================================================================

    // Dopo il timing, fuori dalla regione parallela, solo il processo MASTER stampa
    if(rank == MASTER) {
        double t_total = t_spmv + t_thread_red + t_allreduce_prnew + t_update + t_norm;
        printf("\n--- PROFILO TEMPO (totale su tutte le iterazioni) ---\n");
        printf("  SpMV locale      : %7.3f s  (%5.1f%%)\n", t_spmv,
               100.0 * t_spmv / t_total);
        printf("  Riduzione thread : %7.3f s  (%5.1f%%)\n", t_thread_red,
               100.0 * t_thread_red / t_total);
        printf("  Allreduce+attesa : %7.3f s  (%5.1f%%)\n", t_allreduce_prnew,
               100.0 * t_allreduce_prnew / t_total);
        printf("  Aggiornamento    : %7.3f s  (%5.1f%%)\n", t_update,
               100.0 * t_update / t_total);
        printf("  Norma MPI        : %7.3f s  (%5.1f%%)\n", t_norm,
               100.0 * t_norm / t_total);
        printf("  Totale misurato  : %7.3f s\n", t_total);
    }




    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;


    // ===================== SEZIONE DI TEST (verifica + stampa) ==================================================
    // (puoi commentare l'intero blocco quando passi in produzione)

    double *full_pr = (double*)malloc(NODES * sizeof(double));

    MPI_Allgatherv(prnew + displs_pr[rank], pcols[rank], MPI_DOUBLE,
                   full_pr, pcols, displs_pr, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    if (rank == MASTER) {
        double sum_pr = 0.0;
        for (i = 0; i < NODES; i++) {
            sum_pr += full_pr[i];
        }
        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("=============================================\n");
        printf("Tempo totale power iteration: %f secondi\n", time_spent);

        /*
        printf("\n--- VETTORE PAGERANK FINALE ---\n");
        for (i = 0; i < NODES; i++) {
            printf("Nodo %d: %.6f\n", i + 1, full_pr[i]);
        }
        printf("=============================================\n");
        */
    }

    free(full_pr);

    // ===================== FINE SEZIONE DI TEST ================================================================





        // Liberazione memoria
        free(val); free(rowind); free(colptr); free(readsum);
        free(prold); free(prnew); free(sum);

        free(sendcnts); free(displs); free(pcols); free(displs_pr);
        //free(rec_val); free(rec_row);

        MPI_Finalize();
        return 0;
}
