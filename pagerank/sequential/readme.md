# PageRank Sequenziale: Scelte Progettuali

## Panoramica

Questa implementazione rappresenta la versione sequenziale di riferimento dell'algoritmo PageRank Power Iteration. Funge da baseline corretta e validata per il confronto con tutte le versioni parallele (Pthreads, OpenMP, MPI, MPI+OpenMP).

## Struttura Dati: Formato CSC (Compressed Sparse Column)

La matrice del grafo e' memorizzata in formato CSC per massimizzare l'efficienza della moltiplicazione matrice-vettore sparsa:

- val[EDGES]: valori degli archi (pesi normalizzati)
- rowind[EDGES]: indici di riga (nodi destinazione) per ogni arco
- colptr[NODES+1]: puntatori alle colonne, colptr[col] indica l'inizio degli archi della colonna col
- sum[NODES]: out-degree di ogni nodo (numero di link uscenti)

## Algoritmo: Power Iteration

L'algoritmo itera fino alla convergenza (norma L2 < 0.00001) eseguendo a ogni passo:

### Fase 1: Calcolo Massa Dangling Nodes

I nodi senza link uscenti (out-degree = 0) causano perdita di massa di probabilita'. La massa viene raccolta:

double dangling_mass = 0.0;
for (col = 0; col < NODES; col++) {
    if (sum[col] == 0) {
        dangling_mass += prold[col];
    }
}
double redistribution = (dangling_mass * DAMPING) / NODES;

### Fase 2: Moltiplicazione Matrice-Vettore Sparsa

Prodotto della matrice CSC per il vettore PageRank corrente:

for (col = 0; col < NODES; col++) {
    for (j = colptr[col]; j < colptr[col + 1]; j++) {
        prnew[rowind[j]] += val[j] * prold[col];
    }
}

### Fase 3: Applicazione Damping e Redistribuzione

Formula completa del PageRank:

prnew[i] = prnew[i] * DAMPING + (1 - DAMPING) / NODES + redistribution

Dove:
- prnew[i] * DAMPING: contributo dai link entranti smorzato del 15%
- (1 - DAMPING) / NODES: teleporting (probabilita' di salto casuale)
- redistribution: quota recuperata dai dangling nodes

### Fase 4: Calcolo Convergenza

Norma L2 (distanza euclidea) tra nuovo e vecchio vettore:

norm_sq = 0.0;
for (i = 0; i < NODES; i++) {
    diff[i]  = prnew[i] - prold[i];
    norm_sq += diff[i] * diff[i];
    prold[i] = prnew[i];
}
norm = sqrt(norm_sq);

## Correttezza Matematica

### Inizializzazione Stocastica

Il vettore iniziale assegna probabilita' uniforme: prold[i] = 1.0 / NODES, garantendo somma = 1.0.

### Conservazione della Massa

La gestione dei dangling nodes con redistribuzione garantisce che la somma del vettore PageRank rimanga esattamente 1.0 a ogni iterazione.

### Applicazione Centralizzata del Damping

Il damping viene applicato in un ciclo separato dopo la moltiplicazione, garantendo che la formula sia eseguita una sola volta per nodo.

## Robustezza del Parser

Il parser CSC gestisce correttamente:
- Nodi non sequenziali (buchi negli indici)
- Dangling nodes (nodi senza archi uscenti)
- File con formato non perfettamente ordinato

## Vantaggi e Svantaggi

### Vantaggi

- Implementazione di riferimento corretta e validata
- Codice semplice e manutenibile
- Gestione completa dei dangling nodes
- Verifica matematica integrata (somma finale = 1.0)
- Parser robusto per grafi irregolari

### Svantaggi

- Esecuzione puramente sequenziale
- Tempo proporzionale alla dimensione del grafo
- Nessuno sfruttamento di architetture multi-core

## Comando di Compilazione ed Esecuzione

Compilazione:

gcc sequential/pr_sequential.c libraries/data.c libraries/measure.c -o sequential/pr_sequential -Ilibraries -lm

Esecuzione:

./sequential/pr_sequential

## Ruolo come Baseline

Questa versione sequenziale serve come:

1. Riferimento di correttezza per tutte le versioni parallele
2. Baseline per il calcolo dello speedup
3. Validazione dell'algoritmo matematico
4. Punto di partenza per il debugging delle versioni parallele

## Strutture Dati e Occupazione Memoria

| Struttura | Dimensione | Descrizione |
|-----------|------------|-------------|
| val | EDGES * 8 byte | Pesi degli archi normalizzati |
| rowind | EDGES * 4 byte | Indici dei nodi destinazione |
| colptr | (NODES+1) * 4 byte | Puntatori di inizio colonna |
| sum | NODES * 4 byte | Out-degree per nodo |
| prold | NODES * 8 byte | Vettore PageRank precedente |
| prnew | NODES * 8 byte | Vettore PageRank nuovo |
| damp1 | NODES * 8 byte | Costante di damping (0.85) |
| damp2 | NODES * 8 byte | Costante di teleporting |
| diff | NODES * 8 byte | Differenza tra iterazioni |