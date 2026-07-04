# Comandi di Compilazione ed Esecuzione per Tutte le Versioni PageRank

## Indice

- [0. Sequenziale](#0-sequenziale)
- [1. Pthreads](#1-pthreads)
- [2. OpenMP](#2-openmp)
- [3. MPI](#3-mpi)
- [4. MPI + OpenMP](#4-mpi--openmp-ibrido)
- [5. Note](#5-note)

---

## 0. SEQUENZIALE

### pr_sequential

#### Compilazione

```bash
gcc sequential/pr_sequential.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c \
-o sequential/pr_sequential -Ilibraries -lm
```

#### Esecuzione

```bash
./sequential/pr_sequential
```

#### Benchmark (6 esecuzioni)

```bash
for i in {1..6}; do
    ./sequential/pr_sequential
done
```

---

## 1. PTHREADS

### pr_pthread

#### Compilazione

```bash
gcc pthread/pr_pthread.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c \
-o pthread/pr_pthread -Ilibraries -lpthread -lm
```

#### Esecuzione

```bash
./pthread/pr_pthread
```

---

### pr_pthread_opt (Zero Mutex)

#### Compilazione

```bash
gcc pthread/pr_pthread_opt.c libraries/pagerank_utils.c libraries/data.c libraries/measure.c \
-o pthread/pr_pthread_opt -Ilibraries -lpthread -lm
```

#### Esecuzione

```bash
./pthread/pr_pthread_opt
```

---

## 2. OPENMP

### pr_openmp_opt (Zero Atomic)

#### Compilazione

```bash
gcc <FLAGS> -o openmp/pr_openmp_opt openmp/pr_openmp_opt.c \
libraries/pagerank_utils.c libraries/data.c libraries/measure.c \
-Ilibraries -lm
```

#### Esecuzione

```bash
./openmp/pr_openmp_opt <NTHREADS>
```

---

## 3. MPI

### pr_mpi

#### Compilazione

```bash
mpicc <FLAGS> mpi/pr_mpi.c libraries/pagerank_utils.c \
libraries/data.c libraries/measure.c \
-o mpi/pr_mpi -Ilibraries -lm
```

#### Compilazione con MPE

```bash
mpecc <FLAGS> -mpilog -lpthread \
-o mpi/pr_mpi -Ilibraries -lm \
mpi/pr_mpi.c libraries/pagerank_utils.c \
libraries/data.c libraries/measure.c
```

#### Esecuzione

```bash
mpirun -np <NPROC> <OPT> \
-machinefile mpi/machinefile.txt \
mpi/pr_mpi
```

#### Benchmark (6 esecuzioni)

```bash
for i in {1..6}; do
    mpirun -np <NPROC> <OPT> \
    -machinefile mpi/machinefile.txt \
    mpi/pr_mpi
done
```

---

### pr_mpi_opt

#### Compilazione

```bash
mpicc <FLAGS> mpi/pr_mpi_opt.c libraries/pagerank_utils.c \
libraries/data.c libraries/measure.c \
-o mpi/pr_mpi_opt -Ilibraries -lm
```

#### Esecuzione

```bash
mpirun -np <NPROC> <OPT> \
-machinefile mpi/machinefile.txt \
mpi/pr_mpi_opt
```

#### Benchmark (6 esecuzioni)

```bash
for i in {1..6}; do
    mpirun -np <NPROC> <OPT> \
    -machinefile mpi/machinefile.txt \
    mpi/pr_mpi_opt
done
```

---

## 4. MPI + OPENMP IBRIDO

### pr_hybrid

#### Compilazione

```bash
mpicc <FLAGS> mpi_openMP/pr_hybrid.c \
libraries/pagerank_utils.c libraries/data.c libraries/measure.c \
-o mpi_openMP/pr_hybrid -Ilibraries -fopenmp -lm
```

#### Esecuzione

```bash
mpirun -np <NPROC> <OPT> \
-machinefile mpi_openMP/machinefile.txt \
mpi_openMP/pr_hybrid <NTHREADS>
```

#### Benchmark (6 esecuzioni)

```bash
for i in {1..6}; do
    mpirun -np <NPROC> <OPT> \
    -machinefile mpi_openMP/machinefile.txt \
    mpi_openMP/pr_hybrid <NTHREADS>
done
```

---

## 5. NOTE

### Placeholder

| Placeholder | Significato |
|--------------|-------------|
| `<FLAGS>` | Flag di compilazione (es. `-O3; -fopenmp`) |
| `<OPT>` | Opzioni di esecuzione MPI (es. `--mca btl self,openib; --map-by node`) |
| `<NPROC>` | Numero di processi MPI |
| `<NTHREADS>` | Numero di thread OpenMP |

### Librerie Richieste

Tutte le versioni richiedono:

- `libraries/data.h` e `libraries/data.c`
- `libraries/measure.h` e `libraries/measure.c`
- `libraries/pagerank_utils.h` e `libraries/pagerank_utils.c`

### Flag di Compilazione Comuni

- `-Ilibraries` → Include la directory degli header.
- `-lm` → Link alla libreria matematica.
- `-lpthread` → Link alla libreria Pthreads.
- `-fopenmp` → Abilita OpenMP.
- `-O3` → Ottimizzazione di compilazione.
- `-mpilog` → Abilita il logging MPE.

### Opzioni MPI Comuni

- `--mca btl self,openib` → Utilizza Infiniband.
- `--map-by node` → Distribuisce i processi in modalità round-robin sui nodi.

### Machinefile

Esempio di `machinefile.txt`:

```text
compute-0-0
compute-0-1
compute-0-2
compute-0-3
```

### Parallelismo

| Versione | Parallelismo |
|----------|--------------|
| Sequenziale | Nessun parallelismo |
| Pthreads | Numero di thread definito dalla macro `CORES` |
| OpenMP | Numero di thread passato da riga di comando |
| MPI | Numero di processi specificato con `-np` |
| MPI + OpenMP | Processi MPI (`-np`) + thread OpenMP (`<NTHREADS>`) |