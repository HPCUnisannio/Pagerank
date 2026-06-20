#ifndef PAGERANK_UTILS_H
#define PAGERANK_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DAMPING 0.85
#define ERROR 0.00001

/**
 * Builds the CSC (Compressed Sparse Column) matrix from an edge list file.
 * Handles dangling nodes and non-sequential node indices.
 * 
 * @param filepath      Path to the edge list file
 * @param NODES         Total number of nodes
 * @param EDGES         Total number of edges
 * @param val           Output: edge weights array (must be pre-allocated EDGES)
 * @param rowind        Output: row indices array (must be pre-allocated EDGES)
 * @param colptr        Output: column pointers array (must be pre-allocated NODES+1)
 * @param readsum       Output: out-degree per node (must be pre-allocated NODES)
 * @return              0 on success, -1 on error
 */
int csc_build_from_file(const char *filepath, int NODES, int EDGES,
                        double *val, int *rowind, int *colptr, int *readsum);

/**
 * Normalizes column weights so each column sums to 1.0 (stochastic matrix).
 * 
 * @param NODES         Total number of nodes
 * @param EDGES         Total number of edges
 * @param val           Edge weights array (modified in place)
 * @param colptr        Column pointers array
 * @param readsum       Out-degree per node
 */
void csc_normalize_columns(int NODES, int EDGES, double *val, 
                           int *colptr, int *readsum);

/**
 * Initializes the PageRank vector with uniform probability.
 * 
 * @param NODES         Total number of nodes
 * @param prold         PageRank vector to initialize (must be pre-allocated NODES)
 */
void pagerank_init_vector(int NODES, double *prold);

/**
 * Computes the dangling mass from nodes with no outgoing links.
 * 
 * @param NODES         Total number of nodes
 * @param prold         Current PageRank vector
 * @param readsum       Out-degree per node
 * @return              Total dangling mass
 */
double pagerank_compute_dangling_mass(int NODES, double *prold, int *readsum);

/**
 * Computes a portion of the dangling mass for a range of columns.
 * Used by parallel versions where each process/thread handles a subset.
 * 
 * @param prold         Current PageRank vector
 * @param readsum       Out-degree per node
 * @param col_start     First column to check
 * @param col_end       Last column to check (exclusive)
 * @return              Dangling mass for the specified column range
 */
double pagerank_compute_dangling_mass_range(double *prold, int *readsum,
                                            int col_start, int col_end);

/**
 * Performs sparse matrix-vector multiplication for a range of columns.
 * 
 * @param val           Edge weights array
 * @param rowind        Row indices array
 * @param colptr        Column pointers array
 * @param prold         Input PageRank vector
 * @param prnew         Output vector (accumulated, not zeroed)
 * @param col_start     First column to process
 * @param col_end       Last column to process (exclusive)
 * @param displ         Displacement for index translation (MPI versions)
 */
void csc_spmv_range(double *val, int *rowind, int *colptr, double *prold,
                    double *prnew, int col_start, int col_end, int displ);

/**
 * Applies the PageRank formula: damping, teleportation, and redistribution.
 * 
 * @param NODES         Total number of nodes
 * @param prnew         Raw PageRank vector (modified in place)
 * @param prold         Old PageRank vector (updated with new values)
 * @param redistribution Redistribution value from dangling nodes
 * @param damping       Damping factor (typically 0.85)
 */
void pagerank_apply_damping(int NODES, double *prnew, double *prold,
                            double redistribution, double damping);

/**
 * Applies the PageRank formula for a range of nodes and computes partial norm.
 * 
 * @param prnew         Raw PageRank vector (modified in place)
 * @param prold         Old PageRank vector (updated with new values)
 * @param node_start    First node to process (inclusive)
 * @param node_end      Last node to process (exclusive)
 * @param redistribution Redistribution value from dangling nodes
 * @param damping       Damping factor (typically 0.85)
 * @param total_nodes   Total number of nodes (needed for teleporting)
 * @param norm_sq_local Output: partial squared norm (accumulated)
 */
void pagerank_update_and_norm_range(double *prnew, double *prold,
                                    int node_start, int node_end,
                                    double redistribution, double damping,
                                    int total_nodes, double *norm_sq_local);

/**
 * Computes the L2 norm squared between two vectors for a range of nodes.
 * 
 * @param prnew         New PageRank vector
 * @param prold         Old PageRank vector
 * @param node_start    First node to process
 * @param node_end      Last node to process (exclusive)
 * @return              Squared L2 norm for the range
 */
double pagerank_compute_norm_range(double *prnew, double *prold,
                                   int node_start, int node_end);

/**
 * Computes the column distribution for N processes/threads.
 * 
 * @param NODES         Total number of nodes
 * @param NPROC         Number of processes/threads
 * @param pcols         Output: columns per process
 * @param displs        Output: displacement for each process
 */
void compute_column_distribution(int NODES, int NPROC, int *pcols, int *displs);

/**
 * Computes non-zero element distribution based on CSC column pointers.
 * 
 * @param NPROC         Number of processes/threads
 * @param pcols         Columns per process
 * @param colptr        Column pointers array
 * @param sendcnts      Output: number of non-zeros per process
 * @param displs        Output: displacement for each process
 */
void compute_nnz_distribution(int NPROC, int *pcols, int *colptr,
                              int *sendcnts, int *displs);

/**
 * Validates the PageRank result by checking the sum equals 1.0.
 * 
 * @param NODES         Total number of nodes
 * @param pr            PageRank vector to validate
 * @return              Sum of all PageRank values
 */
double pagerank_validate(int NODES, double *pr);

/**
 * Reduces thread-local sums into a process-level sum array.
 * Used by OpenMP privatized versions.
 * 
 * @param NODES         Total number of nodes
 * @param num_threads   Number of threads
 * @param thread_sums   Array of thread-local sum vectors
 * @param sum           Output: process-level sum vector
 */
void reduce_thread_sums(int NODES, int num_threads, double **thread_sums, double *sum);

#endif /* PAGERANK_UTILS_H */