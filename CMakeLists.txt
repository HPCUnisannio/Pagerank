cmake_minimum_required(VERSION 3.10)
project(Pagerank C)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)

# Configurazione esplicita di MPI per Windows (MSMPI)
set(MPI_C_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_C_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")



# Eseguibile per OpenMP (versione 3 - privatizzata)
add_executable(pr_openmp_opt pagerank/my_code/openmp/pr_openmp_opt.c)
target_compile_options(pr_openmp_opt PRIVATE -fopenmp)
target_link_options(pr_openmp_opt PRIVATE -fopenmp)
target_link_libraries(pr_openmp_opt PRIVATE OpenMP::OpenMP_C m)

# Eseguibile per MPI (versione base sistemata)
add_executable(pr_mpi pagerank/my_code/mpi/pr_mpi.c)
target_include_directories(pr_mpi PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(pr_mpi PRIVATE ${MPI_C_LIBRARIES} m)

# ============================================================================
# Eseguibile per MPI puro ottimizzato
# Solo MPI con centralizzazione della lettura del datasest e distribuzione del post processing(simmetrico)
# Prevista asincronia dove possibile
# ============================================================================
add_executable(pr_mpi2 pagerank/my_code/mpi/pr_mpi2.c)
target_include_directories(pr_mpi2 PRIVATE ${MPI_C_INCLUDE_PATH})
target_compile_options(pr_mpi2 PRIVATE -fopenmp)
target_link_options(pr_mpi2 PRIVATE -fopenmp)
target_link_libraries(pr_mpi2 PRIVATE ${MPI_C_LIBRARIES} OpenMP::OpenMP_C m)






# ============================================================================
# Eseguibile per MPI + OpenMP (versione base + OpenMP grana grossa + privatizzazione + master)
# ============================================================================
add_executable(pr_mpi_openmp pagerank/my_code/mpi_openMP/pr_mpi_openmp.c)
target_include_directories(pr_mpi_openmp PRIVATE ${MPI_C_INCLUDE_PATH})
target_compile_options(pr_mpi_openmp PRIVATE -fopenmp)
target_link_options(pr_mpi_openmp PRIVATE -fopenmp)
target_link_libraries(pr_mpi_openmp PRIVATE ${MPI_C_LIBRARIES} OpenMP::OpenMP_C m)

# ============================================================================
# Eseguibile per MPI + OpenMP 
# MPI ottimizzato con rimozione distribuzione ridondante ed inutile dei dati iniziale,
# inserimento distribuzione del post processing,
# OpenMP grana grossa + privatizzazione + uso di master thread
# ============================================================================
add_executable(pr_mpi_openmp_distributed pagerank/my_code/mpi_openMP/pr_mpi_openmp_distributed\.c)
target_include_directories(pr_mpi_openmp_distributed PRIVATE ${MPI_C_INCLUDE_PATH})
target_compile_options(pr_mpi_openmp_distributed PRIVATE -fopenmp)
target_link_options(pr_mpi_openmp_distributed PRIVATE -fopenmp)
target_link_libraries(pr_mpi_openmp_distributed PRIVATE ${MPI_C_LIBRARIES} OpenMP::OpenMP_C m)


# ============================================================================
# Eseguibile per MPI + OpenMP
# MPI con centralizzazione della lettura del datasest e distribuzione
# ottimizzato con distribuzione del post processing
# OpenMP grana grossa + privatizzazione + uso di master thread
# ============================================================================
add_executable(pr_mpi_openmp_centralized pagerank/my_code/mpi_openMP/pr_mpi_openmp_centralized.c)
target_include_directories(pr_mpi_openmp_centralized PRIVATE ${MPI_C_INCLUDE_PATH})
target_compile_options(pr_mpi_openmp_centralized PRIVATE -fopenmp)
target_link_options(pr_mpi_openmp_centralized PRIVATE -fopenmp)
target_link_libraries(pr_mpi_openmp_centralized PRIVATE ${MPI_C_LIBRARIES} OpenMP::OpenMP_C m)





# Eseguibile per Pthreads (opzionale)
add_executable(pr_pthread_opt pagerank/my_code/pthread/pr_pthread_opt.c)
target_link_libraries(pr_pthread_opt PRIVATE pthread m)