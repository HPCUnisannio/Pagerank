#include "measure.h"
#include "data.h"
#include <stdio.h>
#include <math.h>

/* ===== TIME MEASUREMENT ===== */

double measure_get_time(void) {
    return get_time();  // Uses inline get_time() from measure.h
}

/* ===== SPEEDUP & EFFICIENCY ===== */

double measure_speedup(double seq_time, double par_time) {
    if (par_time <= 0.0) {
        fprintf(stderr, "Warning: Invalid parallel time (%.6f)\n", par_time);
        return 0.0;
    }
    return seq_time / par_time;
}

double measure_efficiency(double speedup, int num_threads) {
    if (num_threads <= 0) {
        fprintf(stderr, "Warning: Invalid number of threads (%d)\n", num_threads);
        return 0.0;
    }
    return speedup / num_threads;
}

double measure_weak_scaling(double seq_time_small, double par_time_large, int num_threads) {
    if (num_threads <= 0 || par_time_large <= 0.0) {
        fprintf(stderr, "Warning: Invalid parameters for weak scaling\n");
        return 0.0;
    }
    // Weak scaling: how well speedup scales with increased problem size
    // Expected speedup for weak scaling = number of processors (if communication is minimal)
    double expected_time = seq_time_small;  // Time should remain constant if perfectly scaled
    return expected_time / par_time_large;
}

/* ===== PERFORMANCE METRICS ===== */

double measure_communication_overhead(double total_time, double computation_time) {
    if (total_time <= 0.0) {
        fprintf(stderr, "Warning: Invalid total time (%.6f)\n", total_time);
        return 0.0;
    }
    if (computation_time > total_time) {
        fprintf(stderr, "Warning: Computation time exceeds total time\n");
        return 0.0;
    }
    return ((total_time - computation_time) / total_time) * 100.0;
}

double measure_load_balance(double max_local_time, double avg_local_time) {
    if (avg_local_time <= 0.0) {
        fprintf(stderr, "Warning: Invalid average time (%.6f)\n", avg_local_time);
        return 0.0;
    }
    if (max_local_time < avg_local_time) {
        fprintf(stderr, "Warning: Max time is less than average time\n");
        return 0.0;
    }
    return avg_local_time / max_local_time;
}

/* ===== UTILITY FUNCTIONS ===== */

void measure_print_summary(const char *label, double setup_time, double compute_time, double total_time) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  EXECUTION SUMMARY: %s\n", label);
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  Setup Time:       %.6f seconds\n", setup_time);
    printf("║  Compute Time:     %.6f seconds\n", compute_time);
    printf("║  Total Time:       %.6f seconds\n", total_time);
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void measure_print_comparison(const char *label1, double time1,
                              const char *label2, double time2) {
    double speedup = measure_speedup(time1, time2);
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  PERFORMANCE COMPARISON\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  %-30s  Time: %.6f sec\n", label1, time1);
    printf("║  %-30s  Time: %.6f sec\n", label2, time2);
    printf("╠════════════════════════════════════════════════════════════╣\n");
    
    if (speedup > 0) {
        printf("║  Speedup (vs %s):           %.4f x\n", label1, speedup);
        
        double improvement = ((time1 - time2) / time1) * 100.0;
        if (improvement > 0) {
            printf("║  Performance Improvement:  %.2f%%\n", improvement);
        } else {
            printf("║  Performance Degradation:  %.2f%%\n", fabs(improvement));
        }
    }
    
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}
