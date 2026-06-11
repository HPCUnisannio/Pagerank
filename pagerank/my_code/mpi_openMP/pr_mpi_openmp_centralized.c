/*
PageRank Ibrido (MPI + OpenMP) - Versione ad Architettura Distribuita con I/O Centralizzato
Questa variante del PageRank ibrido introduce un modello a memoria distribuita pura per la gestione della matrice del grafo. A differenza della versione precedente (in cui ogni processo leggeva ridondantemente il file intero), qui la gestione dell'I/O e la costruzione della struttura dati iniziale sono completamente centralizzate sul processo MASTER, ottimizzando drasticamente l'occupazione di RAM sui nodi di calcolo (Worker).

🎯 Innovazioni e Differenze Architetturali Rispetto alla Versione Precedente
1. I/O Centralizzato e Isolamento dei Worker
Master-Only I/O: Solo il processo MASTER (rank == 0) apre il file, esegue il parsing del testo tramite fscanf e costruisce la struttura in formato CSC (Compressed Sparse Column).

Zero Overhead di File System per i Worker: I nodi di calcolo non accedono al disco, eliminando i colli di bottiglia legati all'I/O concorrente su cluster.

2. Massima Efficienza della Memoria (RAM)
Nella versione precedente, ogni processo manteneva in memoria gli interi array globali val e rowind di dimensione pari a EDGES (milioni di archi).

In questa versione, gli array globali vengono allocati esclusivamente dal Master. I Worker allocano solo i buffer locali rec_val e rec_row di dimensione my_cnt (esattamente la quota di non-zeri di loro competenza). Il Master, subito dopo la distribuzione, libera (free) la matrice globale.

3. Distribuzione Intelligente in Due Fasi (Metadati + Dati)
Fase 1 (Metadati): Viene eseguito un MPI_Bcast degli array di controllo colptr e readsum. Grazie a queste informazioni, ogni processo (Master e Worker) è in grado di calcolare autonomamente le dimensioni esatte dei blocchi di colonne (pcols) e il numero esatto di non-zeri associati (sendcnts e displs).

Fase 2 (Dati Pesanti): Viene eseguita una MPI_Scatterv per distribuire i coefficienti della matrice (val) e gli indici di riga (rowind). Ogni processo riceve solo ciò che deve effettivamente computare.

⚙️ Flusso di Esecuzione delle Iterazioni (OpenMP Coarse-Grained)
Il cuore del calcolo mantiene l'eccellente ottimizzazione a grana grossa (regione parallela aperta una sola volta fuori dal do-while), ma lavora sui buffer locali:

Fase A (SpMV Locale Distribuita): I thread OpenMP eseguono il prodotto matrice-vettore ciclando sulle colonne locali (rec_col). Gli indici per accedere a rec_val e rec_row vengono mappati scalando l'offset globale (displs[rank]). L'accumulo avviene sui vettori privati local_sum per eliminare i conflitti di scrittura.

Fase B (Riduzione Thread): I vettori privati dei thread vengono ridotti nel vettore sum locale al processo tramite istruzioni vettoriali SIMD.

Fase C (Overlap Rete/CPU con Iallreduce): Viene lanciata la MPI_Iallreduce asincrona sul vettore sum. Mentre la rete scambia i dati globali, la CPU calcola in parallelo la dangling mass locale (dm_local).

Fase D & E (Post-Processing e Norma): Ricevuto il PageRank globale tramite MPI_Wait, i processi applicano il damping e calcolano lo scarto quadratico medio solo per le proprie colonne, garantendo un bilanciamento del carico simmetrico e scalabile.

📊 Vantaggi e Svantaggi di questo Approccio
Vantaggi:
Footprint di Memoria Ottimale: Ideale per cluster con nodi aventi RAM limitata, poiché la matrice intera risiede solo sul Master per pochi istanti.

Flessibilità d'Uso: Funziona su qualsiasi architettura di cluster (anche senza file system condiviso), dato che i nodi ricevono i dati esclusivamente via rete MPI.

Svantaggi / Limitazioni:
Collo di Bottiglia sul Master in Fase di Setup: La lettura sequenziale con fscanf e la successiva MPI_Scatterv gravano interamente sul Master. Per grafi nell'ordine di centinaia di milioni di archi, la fase di inizializzazione potrebbe richiedere molto tempo (sebbene il ciclo di calcolo rimanga velocissimo).
*/



#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

/*
#define NODES 685230
#define EDGES 7600595
#define FILEPATH "pagerank/dataset/data2.dat"
*/
//mpiexec -n 2 ".\cmake-build-debug\pr_mpi_OpenMP6.exe" 6
// PER TEST CORRETTEZZA
#define NODES 6
#define EDGES 19
#define FILEPATH "pagerank/dataset/data0.dat"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char **argv)
{
    int NPROC, rank, num_threads;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    if (rank == MASTER) {
        printf("\n=============================================\n");
        printf(" AVVIO PAGERANK IBRIDO (lettura centralizzata)\n");
        printf("=============================================\n");
        printf(" -> Processi MPI totali    : %d\n", NPROC);
        printf(" -> Thread OpenMP/processo : %d\n", num_threads);
        printf(" -> Core logici totali     : %d\n", NPROC * num_threads);
        printf("=============================================\n\n");
        fflush(stdout);
    }

    int i, j;
    int colindex, link, colmatch = -1, localsum = 0;
    int co, index;

    // Strutture dati allocate solo dal Master per la lettura
    double *val = NULL;
    int *rowind = NULL;

    // Strutture condivise per i metadati (esistono su tutti i processi)
    int *colptr  = (int*)calloc(NODES + 1, sizeof(int));
    int *readsum = (int*)calloc(NODES, sizeof(int));

    // Vettori centrali dell'algoritmo
    double *prold = (double*)malloc(NODES * sizeof(double));
    double *prnew = (double*)calloc(NODES, sizeof(double));
    double *sum   = (double*)calloc(NODES, sizeof(double));

    const double DAMP1 = DAMPING;
    const double DAMP2 = (1.0 - DAMPING) / NODES;

    // Vettori di utility per la ripartizione dei carichi MPI
    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *pcols     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_col;

    // INIZIALIZZAZIONE: Tutti i processi impostano il vettore iniziale.
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
    }

    // ========================================================================
    // 1. LETTURA E COSTRUZIONE CSC (SOLO MASTER)
    // Il master legge sequenzialmente e prepara la matrice in formato CSC.
    // ========================================================================
    if (rank == MASTER) {
        FILE *fp = fopen(FILEPATH, "r");
        if (fp == NULL) {
            fprintf(stderr, "Rank %d - Errore apertura file '%s'\n", rank, FILEPATH);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        val    = (double*)calloc(EDGES, sizeof(double));
        rowind = (int*)calloc(EDGES, sizeof(int));

        localsum = 0;
        for (i = 0; i < EDGES; i++) {
            if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
                fprintf(stderr, "Rank %d - Errore lettura file\n", rank);
                fclose(fp);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            colindex--; link--;
            rowind[i] = link;

            if (i == 0) {
                colmatch = colindex;
                localsum = 1;
            } else if (colmatch == colindex) {
                localsum++;
            } else {
                readsum[colmatch] = localsum;
                for (int c = colmatch + 1; c <= colindex; c++) {
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

        // Normalizzazione (pesi degli archi)
        index = 0;
        for (i = 0; i < NODES; i++) {
            co = readsum[i];
            for (j = index; j < index + co; j++) val[j] /= co;
            index += co;
        }
        if (rank == MASTER) printf("CSC costruita e normalizzata --> Distribuzione dati\n");
    }

    // ========================================================================
    // 2. DISTRIBUZIONE DATI GLOBALI
    // I metadati (colptr e readsum) servono a tutti per calcolare gli offset.
    // ========================================================================
    MPI_Bcast(colptr,  NODES + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(readsum, NODES,     MPI_INT, MASTER, MPI_COMM_WORLD);

    // CALCOLO RIPARTIZIONE: Si assegnano blocchi continui di colonne ai processi
    for (i = 0; i < NPROC; i++) {
        if (i == 0) {
            pcols[i]     = NODES / NPROC + NODES % NPROC;
            displs_pr[i] = 0;
        } else {
            pcols[i]     = NODES / NPROC;
            displs_pr[i] = pcols[i-1] + displs_pr[i-1];
        }
    }

    // Sfruttando 'colptr', ogni processo calcola quanti valori non-zeri riceverà (sendcnts)
    // e da quale offset (displs) partirà la sua porzione.
    j = 0;
    for (i = 0; i < NPROC; i++) {
        j += pcols[i];
        int k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0) displs[i] = 0;
        else        displs[i] = sendcnts[i-1] + displs[i-1];
    }

    int my_cnt = sendcnts[rank];
    rec_col = pcols[rank];

    double *rec_val = (double*)malloc(my_cnt * sizeof(double));
    int    *rec_row = (int*)malloc(my_cnt * sizeof(int));

    // SCATTER: Il Master invia a ciascun processo SOLO i non-zeri delle proprie colonne.
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(rowind, sendcnts, displs, MPI_INT, rec_row, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        free(val);
        free(rowind);
    }

    // ========================================================================
    // 3. INIZIO ITERAZIONI (IL CUORE DEL CALCOLO)
    // ========================================================================
    double begin = MPI_Wtime();

    double norm = 0.0;
    double dm_local = 0.0, dm_global = 0.0, norm_sq_local = 0.0;
    double t_spmv = 0.0, t_thread_red = 0.0, t_allreduce_prnew = 0.0;
    double t_update = 0.0, t_norm = 0.0;

    // Buffer locale per ogni thread per evitare Race Conditions durante la SpMV
    double **thread_sums = (double**)calloc(num_threads, sizeof(double*));

    #pragma omp parallel private(i, j)
    {
        int tid = omp_get_thread_num();
        thread_sums[tid] = (double*)calloc(NODES, sizeof(double));
        double *local_sum = thread_sums[tid];
        double t_phase = 0.0;

        do {
            // Reset dei cronometri e accumulatori all'inizio di ogni iterazione
            memset(local_sum, 0, NODES * sizeof(double));

            #pragma omp master
            { dm_local = 0.0; norm_sq_local = 0.0; }

            int global_col_start = displs_pr[rank];

            // ----------------------------------------------------------------
            // FASE A: Prodotto Matrice-Vettore (SpMV) Parallelo
            // ----------------------------------------------------------------
            #pragma omp master
            t_phase = MPI_Wtime();

            // Grazie al formato CSC, ogni processo legge solo le PROPRIE colonne
            // dal vettore 'prold'. Non c'è bisogno di raccogliere l'intero vettore!
            #pragma omp for schedule(dynamic, 512)
            for (int local_col = 0; local_col < rec_col; local_col++) {
                int global_col = global_col_start + local_col;
                int start_idx = colptr[global_col] - displs[rank];
                int end_idx   = colptr[global_col + 1] - displs[rank];
                for (j = start_idx; j < end_idx; j++) {
                    local_sum[rec_row[j]] += rec_val[j] * prold[global_col];
                }
            }

            #pragma omp master
            t_spmv += MPI_Wtime() - t_phase;

            // ----------------------------------------------------------------
            // FASE B: Riduzione Thread Locali -> Vettore Globale 'sum'
            // ----------------------------------------------------------------
            #pragma omp master
            t_phase = MPI_Wtime();

            #pragma omp for simd
            for (i = 0; i < NODES; i++) {
                double total_row_sum = 0.0;
                for (int t = 0; t < num_threads; t++)
                    if (thread_sums[t] != NULL)
                        total_row_sum += thread_sums[t][i];
                sum[i] = total_row_sum;
            }

            #pragma omp master
            t_thread_red += MPI_Wtime() - t_phase;

            // ----------------------------------------------------------------
            // FASE C: Asincronia (Overlap MPI / CPU)
            // ----------------------------------------------------------------
            MPI_Request request;
            #pragma omp master
            {
                t_phase = MPI_Wtime();
                // LANCIO RETE: La rete inizia a sommare e scambiare il vettore 'sum'
                MPI_Iallreduce(sum, prnew, NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
            }

            // LAVORO CPU: Mentre i dati viaggiano, i thread calcolano la dangling mass locale
            #pragma omp for simd reduction(+:dm_local)
            for (i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;
                if (readsum[global_col] == 0)
                    dm_local += prold[global_col];
            }

            #pragma omp master
            {
                // Unisco la massa dispersa locale in una globale (scalare, istantaneo)
                MPI_Allreduce(&dm_local, &dm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                // SINCRONIZZAZIONE: Assicuro che prnew sia arrivato prima di proseguire
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                t_allreduce_prnew += MPI_Wtime() - t_phase;
            }
            #pragma omp barrier // Impedisce ai thread di avviare il damping senza i dati pronti

            double redistribution = dm_global * DAMPING / NODES;

            // ----------------------------------------------------------------
            // FASE D: Aggiornamento Distribuito e Norma
            // ----------------------------------------------------------------
            #pragma omp master
            t_phase = MPI_Wtime();

            // MAGIA DEL DISTRIBUITO: Invece di iterare su NODES, iteriamo solo su rec_col!
            #pragma omp for simd reduction(+:norm_sq_local)
            for (i = 0; i < rec_col; i++) {
                int global_col = global_col_start + i;

                // Si calcola il PageRank finale per questo nodo
                prnew[global_col] = prnew[global_col] * DAMP1 + DAMP2 + redistribution;

                // Calcolo locale della variazione per la condizione di stop
                double diff = prnew[global_col] - prold[global_col];
                norm_sq_local += diff * diff;

                // Update In-Place: prold è già pronto per la SpMV del ciclo successivo
                prold[global_col] = prnew[global_col];
            }

            #pragma omp master
            t_update += MPI_Wtime() - t_phase;

            // ----------------------------------------------------------------
            // FASE E: Convergenza Globale
            // ----------------------------------------------------------------
            #pragma omp master
            {
                t_phase = MPI_Wtime();
                double norm_sq_global = 0.0;
                // Si sommano i delta locali (scalari) per ottenere l'errore globale
                MPI_Allreduce(&norm_sq_local, &norm_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                norm = sqrt(norm_sq_global);
                t_norm += MPI_Wtime() - t_phase;
            }
            #pragma omp barrier

        } while (norm > ERROR);

        free(local_sum);
    }
    free(thread_sums);

    // ========================================================================
    // PROFILAZIONE E VALIDAZIONE
    // ========================================================================
    if (rank == MASTER) {
        double t_total = t_spmv + t_thread_red + t_allreduce_prnew + t_update + t_norm;
        printf("\n--- PROFILO TEMPO (totale su tutte le iterazioni) ---\n");
        printf("  SpMV locale      : %7.3f s  (%5.1f%%)\n", t_spmv, 100.0 * t_spmv / t_total);
        printf("  Riduzione thread : %7.3f s  (%5.1f%%)\n", t_thread_red, 100.0 * t_thread_red / t_total);
        printf("  Allreduce+attesa : %7.3f s  (%5.1f%%)\n", t_allreduce_prnew, 100.0 * t_allreduce_prnew / t_total);
        printf("  Aggiornamento    : %7.3f s  (%5.1f%%)\n", t_update, 100.0 * t_update / t_total);
        printf("  Norma MPI        : %7.3f s  (%5.1f%%)\n", t_norm, 100.0 * t_norm / t_total);
        printf("  Totale misurato  : %7.3f s\n", t_total);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double time_spent = end - begin;

    // ========================================================================
    // VERIFICA MATEMATICA (Da disattivare in produzione)
    // Raccoglie i pezzi "puliti" calcolati nella Fase D da tutti i processi
    // ========================================================================
    double *full_pr = (double*)malloc(NODES * sizeof(double));

    MPI_Allgatherv(prnew + displs_pr[rank], pcols[rank], MPI_DOUBLE,
                   full_pr, pcols, displs_pr, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    if (rank == MASTER) {
        double sum_pr = 0.0;
        for (i = 0; i < NODES; i++) sum_pr += full_pr[i];
        printf("\n=============================================\n");
        printf("VERIFICA MATEMATICA VETTORE PAGERANK:\n");
        printf("Somma totale di tutti gli elementi: %.10f\n", sum_pr);
        printf("Tempo iterazioni: %f secondi\n", time_spent);
        printf("\n--- VETTORE FINALE (primi e ultimi) ---\n");
        for (i = 0; i < (NODES < 10 ? NODES : 10); i++)
        {
            printf("Nodo %d: %.6f\n", i + 1, full_pr[i]);
        }
        printf("=============================================\n");
    }

    free(full_pr);

    // Pulizia finale della memoria
    free(colptr); free(readsum);
    free(prold);  free(prnew);   free(sum);
    free(sendcnts); free(displs);
    free(pcols);    free(displs_pr);
    free(rec_val);  free(rec_row);

    MPI_Finalize();
    return 0;
}