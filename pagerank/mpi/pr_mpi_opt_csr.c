#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "../libraries/data.h"
#include "../libraries/measure.h"
#include "../libraries/pagerank_utils.h"

#define MASTER 0
#define DAMPING 0.85
#define ERROR 0.00001

int main(int argc, char *argv[])
{
    int NPROC, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPROC);

    // Caricamento metadati del grafo
    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    const int NODES = graph->nodes;
    const int EDGES = graph->edges;
    const char* FILEPATH = graph->filepath;

    if (rank == MASTER) {
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  AVVIO PAGERANK MPI Ottimizzato\n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Nodi: %d - Archi: %d\n", NODES, EDGES);
        printf("║  Processi MPI totali : %d\n", NPROC);
        printf("║  Struttura Dati      : Compressed Sparse Row (CSR)\n");
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }

    // Strutture dati globali: allocate interamente solo dal MASTER
    double *val = NULL;
    int *colind = NULL;

    // ========================================================================
    // OTTIMIZZAZIONE: Allocazione contigua dei metadati strutturali
    // ========================================================================
    int *metadata_buffer = (int*)calloc((NODES + 1) + NODES, sizeof(int));
    int *rowptr     = metadata_buffer;                  // Punta all'inizio del blocco
    int *out_degree = metadata_buffer + (NODES + 1);    // Sfasato subito dopo rowptr

    // Vettori PageRank (Dimensione globale su tutti i nodi)
    double *prold = (double*)malloc(NODES * sizeof(double));
    double *prnew = (double*)calloc(NODES, sizeof(double));

    // Costanti scalari per il Damping
    const double DAMP1 = DAMPING;

    // Array per il partizionamento e la distribuzione del carico
    int *sendcnts  = malloc(sizeof(int) * NPROC);
    int *displs    = malloc(sizeof(int) * NPROC);
    int *prows     = malloc(NPROC * sizeof(int));
    int *displs_pr = malloc(NPROC * sizeof(int));
    int rec_row;

    double dm_global = 0.0;

    // Inizializzazione uniforme del vettore PageRank
    pagerank_init_vector(NODES, prold);

    // ========================================================================
    // FASE 1: SETUP E DISTRIBUZIONE DELLA MATRICE CSR
    // ========================================================================
    // Variabili per la misurazione dei tempi
    double setup_start = 0.0, setup_end = 0.0;
    double compute_start = 0.0, compute_end = 0.0;
    double setup_time = 0.0, compute_time = 0.0, total_time = 0.0;

    if (rank == MASTER) {
        setup_start = MPI_Wtime();
    }

    // 1a. Lettura file e costruzione CSR (Solo il MASTER)
    if (rank == MASTER) {
        val    = (double*)calloc(EDGES, sizeof(double));
        colind = (int*)calloc(EDGES, sizeof(int));

        if (csr_build_from_file(FILEPATH, NODES, EDGES, val, colind, rowptr, out_degree) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Innesco (Bootstrap) della Dangling Mass per l'iterazione zero (k=0)
        int total_dangling_nodes = 0;
        int i;
        for (i = 0; i < NODES; i++) {
            if (out_degree[i] == 0) {
                total_dangling_nodes++;
            }
        }
        dm_global = (double) total_dangling_nodes / NODES;
    }

    // 1b. Broadcast ACCORPATO dei metadati strutturali (Unica chiamata di rete)
    MPI_Bcast(metadata_buffer, (2 * NODES + 1), MPI_INT, MASTER, MPI_COMM_WORLD);

    // Invia il valore di dm_global iniziale a tutti i processi
    MPI_Bcast(&dm_global, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // Calcolo della distribuzione bilanciata delle RIGHE del grafo tra i processi
    compute_column_distribution(NODES, NPROC, prows, displs_pr);

    // Calcolo dei conteggi di elementi non-zero (nnz) e relativi offset per ogni processo
    compute_nnz_distribution(NPROC, prows, rowptr, sendcnts, displs);

    int my_cnt = sendcnts[rank];
    rec_row = prows[rank];

    // Allocazione delle strutture locali per la porzione di matrice CSR assegnata
    double *rec_val    = (double*)malloc(my_cnt * sizeof(double));
    int    *rec_colind = (int*)malloc(my_cnt * sizeof(int));

    // Il Master distribuisce le porzioni di vettori CSR tramite Scatterv
    MPI_Scatterv(val, sendcnts, displs, MPI_DOUBLE, rec_val, my_cnt, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(colind, sendcnts, displs, MPI_INT, rec_colind, my_cnt, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Il Master rilascia subito la memoria dei vettori globali non più necessari
    if (rank == MASTER) {
        free(val);
        free(colind);

        setup_end = MPI_Wtime();                     // timestamp di fine setup
        setup_time = setup_end - setup_start;        // durata effettiva del setup
        printf("Setup completato in %.6f secondi.\n", setup_time);
        printf("Inizio computazione parallela iterativa...\n");
        fflush(stdout);
    }

    // ========================================================================
    // INZIO REGIONE COMPUTAZIONALE (POWER ITERATION CORE)
    // ========================================================================
    // Barriera di sincronizzazione per garantire la fine del setup su tutti i nodi
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        compute_start = MPI_Wtime();   // inizio della computazione (dopo la barriera)
    }

    // ========================================================================
    // FASE 2: COMPUTAZIONE ITERATIVA PAGERANK (VERSIONE CSR V2 - COALESCED)
    // ========================================================================

    double norm = 0.0;
    double dm_local = 0.0, norm_sq_local = 0.0;
    int iteration_count = 0;

    int global_row_start = displs_pr[rank];
    int global_row_end   = global_row_start + rec_row;

    do {
        norm_sq_local = 0.0;
        dm_local      = 0.0;

        // 1. Calcolo del fattore di redistribuzione basato sulla dm_global dell'iterazione PRECEDENTE
        double redistribution = dm_global * DAMP1 / NODES;

        // 2. Moltiplicazione Matrice-Vettore (SpMV) locale su formato CSR
        csr_spmv_range(rec_val, rec_colind, rowptr, prold, prnew,
                       global_row_start, global_row_end, displs[rank]);

        // 3. Applicazione locale del Damping e calcolo locale del quadrato dell'errore
        pagerank_update_and_norm_range(prnew, prold,
                                       global_row_start, global_row_end,
                                       redistribution, DAMPING, NODES, &norm_sq_local);

        // 4. Pre-calcolo della Dangling Mass locale guardando avanti al vettore appena calcolato (prnew)
        dm_local = pagerank_compute_dangling_mass_range(prnew, out_degree,
                                                        global_row_start, global_row_end);

        // ====================================================================
        // FUSIONE DELLE COMUNICAZIONI (Message Coalescing)
        // ====================================================================
        double local_data[2]  = {dm_local, norm_sq_local};
        double global_data[2] = {0.0, 0.0};

        MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        dm_global = global_data[0];
        norm      = sqrt(global_data[1]);
        // ====================================================================

        // 5. Condivisione globale e sincrona dei blocchi di vettori aggiornati via Allgatherv
        MPI_Allgatherv(prnew + global_row_start, rec_row, MPI_DOUBLE,
                       prold, prows, displs_pr, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        iteration_count++;

    } while (norm > ERROR);

    // Sincronizzazione finale per garantire che tutti i processi abbiano terminato il ciclo
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MASTER) {
        compute_end = MPI_Wtime();
        compute_time = compute_end - compute_start;   // durata della computazione
    }

    // ========================================================================
    // FASE 3: PROFILING E VALIDAZIONE CON CRUSCOTTO AVANZATO
    // ========================================================================
    if (rank == MASTER) {
        total_time = setup_time + compute_time;

        double sum_pr = pagerank_validate(NODES, prold);

        /* ----------------------
            PRINT PR SUM CHECK
        ---------------------- */
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  PR SUM CHECK                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║  Total PR Sum:        %12.10f                              \n", sum_pr);
        printf("║  Expected Sum:        %12.10f                             \n", 1.0);

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

        /* ----------------------
            TIME PERFORMANCE & METRICS DISPLAY
        ---------------------- */
        // Print execution summary
        char label[100];
        sprintf(label, "MPI BASE (%d processes)", NPROC);
        measure_print_summary(label, setup_time, compute_time, total_time, iteration_count);

        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  LOAD BALANCE                             \n");
        printf("╠════════════════════════════════════════════════════════════╣\n");

        // Bilanciamento del carico reale calcolato sugli archi (NNZ) assegnati ai processi
        double max_nnz = sendcnts[0];
        double avg_nnz = (double)EDGES / NPROC;
        int r;
        for (r = 1; r < NPROC; r++) {
            if (sendcnts[r] > max_nnz) max_nnz = sendcnts[r];
        }
        double load_balance = measure_load_balance(max_nnz, avg_nnz);
        printf("║  Load Balance (NNZ):  %12.2f%%                             \n", load_balance * 100.0);
        printf("║  Partition Size (NNZ): avg=%.1f, max=%.0f                  \n", avg_nnz, max_nnz);

        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        /* ----------------------
            READ SEQUENTIAL TIME & METRICS DISPLAY
        ---------------------- */
        double sequential_time = 0.0;
        FILE *seq_file = fopen("sequential/sequential_time.txt", "r");
        if (seq_file != NULL) {
            fscanf(seq_file, "%lf", &sequential_time);
            fclose(seq_file);

            char label1[30];
            sprintf(label1, "Sequenziale");
            char label2[30];
            sprintf(label2, "MPI (%d processes)", NPROC);
            
            measure_print_comparison(label1, sequential_time, label2, compute_time);
        } else {
            printf("Warning: Could not open sequential/sequential_time.txt\n");
        }
        fflush(stdout);
    }

    // Deallocazione della memoria locale ed eliminazione delle strutture dati
    free(metadata_buffer); // Rilascia in un colpo solo sia rowptr che out_degree
    free(prold); 
    free(prnew);
    free(sendcnts); 
    free(displs); 
    free(prows); 
    free(displs_pr);
    free(rec_val); 
    free(rec_colind);

    MPI_Finalize();
    return 0;
}