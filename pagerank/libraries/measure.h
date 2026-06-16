#ifndef PAGERANK_MEASURE_H
#define PAGERANK_MEASURE_H

#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
#endif

/**
 * Performance measurement library for PageRank computation
 * Provides utilities for measuring time, speedup, efficiency, and other metrics
 */

/* ===== TIME MEASUREMENT ===== */

static inline double get_time(void) {
#ifdef _WIN32
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#endif
}

/**
 * Get current time in seconds
 * Works on both Windows and Unix-like systems
 * @return Current time as double in seconds
 */
double measure_get_time(void);

/* ===== SPEEDUP & EFFICIENCY ===== */

/**
 * Calculate speedup between sequential and parallel execution
 * @param seq_time Sequential execution time in seconds
 * @param par_time Parallel execution time in seconds
 * @return Speedup value (should be ideally close to num_threads)
 */
double measure_speedup(double seq_time, double par_time);

/**
 * Calculate parallel efficiency
 * Efficiency = Speedup / Number_of_Processors
 * @param speedup The calculated speedup
 * @param num_threads Number of threads/processors used
 * @return Efficiency value between 0 and 1 (ideally close to 1)
 */
double measure_efficiency(double speedup, int num_threads);

/**
 * Calculate weak scaling efficiency
 * Measures how efficiency scales with problem size and number of processors
 * @param seq_time_small Sequential time for small problem
 * @param par_time_large Parallel time for proportionally larger problem
 * @param num_threads Number of threads/processors used
 * @return Weak scaling efficiency
 */
double measure_weak_scaling(double seq_time_small, double par_time_large, int num_threads);

/* ===== PERFORMANCE METRICS ===== */

/**
 * Calculate communication overhead (for parallel implementations)
 * @param total_time Total parallel execution time
 * @param computation_time Time spent in actual computation
 * @return Communication overhead as percentage of total time
 */
double measure_communication_overhead(double total_time, double computation_time);

/**
 * Calculate load balance factor
 * @param max_local_time Maximum time spent by any thread
 * @param avg_local_time Average time across all threads
 * @return Load balance factor (0-1, where 1 is perfect balance)
 */
double measure_load_balance(double max_local_time, double avg_local_time);

/* ===== UTILITY FUNCTIONS ===== */

/**
 * Print a formatted performance summary
 * @param label Name of the execution (e.g., "Sequential", "Parallel with 4 threads")
 * @param execution_time Time taken for execution
 * @param nodes Number of nodes in the graph
 * @param iterations Number of PageRank iterations
 */
void measure_print_summary(const char *label, double execution_time, 
                           int nodes, int iterations);

/**
 * Compare two execution times and print comparison
 * @param label1 Name of first execution
 * @param time1 Time of first execution
 * @param label2 Name of second execution
 * @param time2 Time of second execution
 */
void measure_print_comparison(const char *label1, double time1,
                              const char *label2, double time2);

#endif // PAGERANK_MEASURE_H
