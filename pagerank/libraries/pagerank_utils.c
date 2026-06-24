#include "pagerank_utils.h"

int csc_build_from_file(const char *filepath, int NODES, int EDGES,
                        double *val, int *rowind, int *colptr, int *readsum)
{
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file '%s'\n", filepath);
        return -1;
    }

    int i, c;
    int colindex, link, colmatch = -1, localsum = 0;

    for (i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Errore lettura file alla riga %d\n", i + 1);
            fclose(fp);
            return -1;
        }

        colindex--;
        link--;
        rowind[i] = link;

        if (i == 0) {
            colmatch = colindex;
            localsum = 1;
        } else if (colmatch == colindex) {
            localsum += 1;
        } else {
            readsum[colmatch] = localsum;
            for (c = colmatch + 1; c <= colindex; c++) {
                colptr[c] = colptr[colmatch] + localsum;
            }
            localsum = 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }

    if (EDGES > 0) {
        readsum[colmatch] = localsum;
        for (c = colmatch + 1; c <= NODES; c++) {
            colptr[c] = EDGES;
        }
    }

    fclose(fp);
    return 0;
}

void csc_normalize_columns(int NODES, int EDGES, double *val,
                           int *colptr, int *readsum)
{
    int i, j, index = 0;
    for (i = 0; i < NODES; i++) {
        int co = readsum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }
}

void pagerank_init_vector(int NODES, double *prold)
{
    int i;
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
    }
}

double pagerank_compute_dangling_mass(int NODES, double *prold, int *readsum)
{
    double dangling_mass = 0.0;
    int i;
    for (i = 0; i < NODES; i++) {
        if (readsum[i] == 0) {
            dangling_mass += prold[i];
        }
    }
    return dangling_mass;
}

double pagerank_compute_dangling_mass_range(double *prold, int *readsum,
                                            int col_start, int col_end)
{
    double dm_local = 0.0;
    int i;
    for (i = col_start; i < col_end; i++) {
        if (readsum[i] == 0) {
            dm_local += prold[i];
        }
    }
    return dm_local;
}

void csc_spmv_range(double *val, int *rowind, int *colptr, double *prold,
                    double *prnew, int col_start, int col_end, int displ)
{
    int col, j;
    for (col = col_start; col < col_end; col++) {
        int start_idx = colptr[col] - displ;
        int end_idx   = colptr[col + 1] - displ;
        for (j = start_idx; j < end_idx; j++) {
            prnew[rowind[j]] += val[j] * prold[col];
        }
    }
}

void pagerank_apply_damping(int NODES, double *prnew, double *prold,
                            double redistribution, double damping)
{
    double damp1 = damping;
    double damp2 = (1.0 - damping) / NODES;
    int i;
    for (i = 0; i < NODES; i++) {
        prnew[i] = prnew[i] * damp1 + damp2 + redistribution;
        prold[i] = prnew[i];
    }
}

void pagerank_update_and_norm_range(double *prnew, double *prold,
                                    int node_start, int node_end,
                                    double redistribution, double damping,
                                    int total_nodes, double *norm_sq_local)
{
    double damp1 = damping;
    double damp2 = (1.0 - damping) / total_nodes;
    int i;
    for (i = node_start; i < node_end; i++) {
        prnew[i] = prnew[i] * damp1 + damp2 + redistribution;
        
        double diff = prnew[i] - prold[i];
        *norm_sq_local += diff * diff;
        
        prold[i] = prnew[i];
    }
}

double pagerank_compute_norm_range(double *prnew, double *prold,
                                   int node_start, int node_end)
{
    double norm_sq = 0.0;
    int i;
    for (i = node_start; i < node_end; i++) {
        double diff = prnew[i] - prold[i];
        norm_sq += diff * diff;
    }
    return norm_sq;
}

void compute_column_distribution(int NODES, int NPROC, int *pcols, int *displs)
{
    int i;
    for (i = 0; i < NPROC; i++) {
        if (i == 0) {
            pcols[i] = NODES / NPROC + NODES % NPROC;
            displs[i] = 0;
        } else {
            pcols[i] = NODES / NPROC;
            displs[i] = pcols[i - 1] + displs[i - 1];
        }
    }
}

void compute_nnz_distribution(int NPROC, int *pcols, int *colptr,
                              int *sendcnts, int *displs)
{
    int j = 0;
    int i;
    for (i = 0; i < NPROC; i++) {
        j += pcols[i];
        int k = j - pcols[i];
        sendcnts[i] = colptr[j] - colptr[k];
        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = sendcnts[i - 1] + displs[i - 1];
        }
    }
}

double pagerank_validate(int NODES, double *pr)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < NODES; i++) {
        sum += pr[i];
    }
    return sum;
}

void reduce_thread_sums(int NODES, int num_threads, double **thread_sums, double *sum)
{
    int i, t;
    for (i = 0; i < NODES; i++) {
        double total = 0.0;
        for (t = 0; t < num_threads; t++) {
            if (thread_sums[t] != NULL) {
                total += thread_sums[t][i];
            }
        }
        sum[i] = total;
    }
}







//======================================================================================================================
// AGGIUNTE PER VERSIONE CSR
//======================================================================================================================
// Aggiungi in pagerank_utils.c

int csr_build_from_file(const char *filepath, int NODES, int EDGES,
                        double *val, int *colind, int *rowptr, int *out_degree)
{
    // Tutte le dichiarazioni rigorosamente in cima al blocco
    int *in_degree;
    int *src_edges;
    int *dst_edges;
    FILE *fp;
    int i;
    int u, v, idx;

    in_degree = (int*)calloc(NODES, sizeof(int));
    src_edges = (int*)malloc(EDGES * sizeof(int));
    dst_edges = (int*)malloc(EDGES * sizeof(int));

    fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Errore apertura file\n");
        free(in_degree);
        free(src_edges);
        free(dst_edges);
        return -1;
    }

    for (i = 0; i < EDGES; i++) {
        fscanf(fp, "%d %d", &u, &v);
        u--; v--; // Conversione 0-based
        src_edges[i] = u;
        dst_edges[i] = v;

        out_degree[u]++; // Gradi uscenti (serve per normalizzare)
        in_degree[v]++;  // Gradi entranti (serve per allocare la CSR)
    }
    fclose(fp);

    // Costruzione del vettore rowptr (puntatori di riga)
    rowptr[0] = 0;
    for (i = 0; i < NODES; i++) {
        rowptr[i+1] = rowptr[i] + in_degree[i];
        in_degree[i] = 0; // Lo azzeriamo per riutilizzarlo come offset nel ciclo successivo
    }

    // Popolamento di colind e val
    for (i = 0; i < EDGES; i++) {
        u = src_edges[i];
        v = dst_edges[i];
        idx = rowptr[v] + in_degree[v]; // Troviamo la posizione corretta per la destinazione

        colind[idx] = u;
        // Normalizzazione immediata: il peso del link dipende dal grado uscente del nodo sorgente
        val[idx] = 1.0 / out_degree[u];

        in_degree[v]++;
    }

    free(src_edges);
    free(dst_edges);
    free(in_degree);
    return 0;
}

// Funzione SpMV ottimizzata per CSR (Compatibile C89)
void csr_spmv_range(double *val, int *colind, int *rowptr, double *prold,
                    double *prnew, int row_start, int row_end, int displ)
{
    // Dichiarazioni in cima al blocco
    int i, j;
    int start_idx, end_idx;
    double sum;

    for (i = row_start; i < row_end; i++) {
        sum = 0.0;
        start_idx = rowptr[i] - displ;
        end_idx   = rowptr[i + 1] - displ;

        for (j = start_idx; j < end_idx; j++) {
            sum += val[j] * prold[colind[j]];
        }
        // Assegnamento DIRETTO. Nessuna Race Condition e nessuna sovrascrittura sparsa!
        prnew[i] = sum;
    }
}