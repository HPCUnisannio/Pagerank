# Comandi di Compilazione ed Esecuzione per Tutte le Versioni PageRank

## Panoramica

Questo documento raccoglie tutti i comandi di compilazione ed esecuzione per le diverse implementazioni dell'algoritmo PageRank analizzate. Le implementazioni coprono quattro paradigmi di parallelismo: Pthreads, OpenMP, MPI puro e MPI+OpenMP ibrido.

---

## 0. SEQUENZIALE

### Versione Sequenziale di Riferimento (Baseline)

Compilazione:

gcc sequential/pr_sequential.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o sequential/pr_sequential -Ilibraries -lm

Esecuzione:

./sequential/pr_sequential

---

## 1. PTHREADS

### Versione Base con Mutex

Compilazione:

gcc pthread/pr_pthread.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o pthread/pr_pthread -Ilibraries -lpthread -lm

Esecuzione:

./pthread/pr_pthread

### Versione Ottimizzata Privatizzata (Zero Mutex)

Compilazione:

gcc pthread/pr_pthread_opt.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o pthread/pr_pthread_opt -Ilibraries -lpthread -lm

Esecuzione:

./pthread/pr_pthread_opt

---

## 2. OPENMP

### Versione Privatizzata (Zero Atomic)

Compilazione:

gcc -fopenmp -O2 -o openmp/pr_openmp_opt openmp/pr_openmp_opt.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -Ilibraries -lm

Esecuzione (con 4 thread):

./openmp/pr_openmp_opt 4

---

## 3. MPI PURO

### Versione Base

Compilazione:

mpicc mpi/pr_mpi.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o mpi/pr_mpi -Ilibraries -lm

Esecuzione (con 4 processi):

mpirun -np 4 -machinefile mpi/machinefile.txt mpi/pr_mpi

### Versione Ottimizzata

Compilazione:

mpicc mpi/pr_mpi_opt.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o mpi/pr_mpi_opt -Ilibraries -lm

Esecuzione (con 4 processi):

mpirun -np 4 -machinefile mpi/machinefile.txt mpi/pr_mpi_opt

---

## 4. MPI + OPENMP IBRIDO

### Versione Base (Post-Processing Centralizzato sul Master)

Compilazione:

mpicc mpi_openMP/pr_mpi_openmp.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp -Ilibraries -fopenmp -lm

Esecuzione (con 4 processi MPI e 2 thread OpenMP per processo):

mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp 2

### Versione Centralizzata (I/O Centralizzato, Post-Processing Distribuito)

Compilazione:

mpicc mpi_openMP/pr_mpi_openmp_centralized.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp_centralized -Ilibraries -fopenmp -lm

Esecuzione (con 4 processi MPI e 2 thread OpenMP per processo):

mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp_centralized 2

### Versione Replicata (Dati Replicati, Zero Comunicazioni di Setup)

Compilazione:

mpicc mpi_openMP/pr_mpi_openmp_replicated.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c -o mpi_openMP/pr_mpi_openmp_replicated -Ilibraries -fopenmp -lm

Esecuzione (con 4 processi MPI e 2 thread OpenMP per processo):

mpirun -np 4 -machinefile mpi_openMP/machinefile.txt mpi_openMP/pr_mpi_openmp_replicated 2

---

## 5. NOTE

### Librerie Richieste

Tutte le versioni richiedono i file:
- libraries/data.h e libraries/data.c per la gestione dei dataset
- libraries/measure.h e libraries/measure.c per la misurazione dei tempi

### Flag di Compilazione Comuni

- -Ilibraries: include la directory libraries per gli header
- -lm: link alla libreria matematica per sqrt()
- -lpthread: link alla libreria pthread (solo versioni Pthread)
- -fopenmp: abilita OpenMP (versioni OpenMP e ibride)
- -O2: ottimizzazione livello 2 (consigliato per OpenMP)

### Machinefile

Per le versioni MPI, il file machinefile.txt deve contenere l'elenco degli host disponibili, ad esempio:
compute-0-0
compute-0-1
compute-0-2
compute-0-3

### Numero di Processi e Thread

- Versione Sequenziale: nessun parallelismo, esecuzione single-core
- Versioni Pthread: il numero di thread e' definito dalla macro CORES nel codice
- Versioni OpenMP: il numero di thread e' passato come argomento da riga di comando
- Versioni MPI: il numero di processi e' specificato con -np in mpirun
- Versioni ibride MPI+OpenMP: numero di processi MPI con -np, numero di thread OpenMP come argomento