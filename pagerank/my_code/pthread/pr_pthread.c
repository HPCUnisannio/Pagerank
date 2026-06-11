
/*
================================================================================
# ANALISI DEI BACHI ORIGINALI E DEI FIX APPLICATI
================================================================================

1. IL DEADLOCK: signal vs broadcast
   - Originale: Il Master usava 'pthread_cond_signal(&proceed_cv);'. Questa
     istruzione sveglia esattamente un solo thread in attesa. Se avevi 6 core
     (1 Master, 5 Worker), il Master finiva, lanciava il segnale, 1 Worker si
     svegliava e gli altri 4 rimanevano dormienti all'infinito. Il programma
     si bloccava per sempre alla prima iterazione.
   - Tua Versione: Usi 'pthread_cond_broadcast(&proceed_cv);'. Questo "urla"
     a tutti i thread in attesa sulla condition variable di svegliarsi e
     procedere. Niente più deadlock.

2. LA CORSA AI DATI (Race Condition) E LA SECONDA BARRIERA
   - Originale: C'era una sola barriera (our_barrier). Una volta che i worker
     si svegliavano, andavano diretti a controllare il while(norm > err). Ma
     cosa succedeva se un worker era velocissimo? Poteva rientrare nel ciclo do,
     azzerare il suo array, e rimettersi a sommare roba mentre il Master magari
     non aveva ancora finito di sistemare le variabili per il giro successivo.
     Peggio ancora, i worker leggevano la variabile globale norm mentre il Master
     poteva aver già iniziato a modificarla per il giro dopo.
   - Tua Versione: L'introduzione di 'our_barrier2' è la mossa vincente (insieme
     a current_norm). Obbliga tutti ad aspettare che il Master abbia finito di
     aggiornare il vettore globale e l'errore. Poi, ognuno legge norm in una
     variabile locale current_norm in modo sicuro (protetto da mutex). Ora
     tutti valutano la condizione di fine ciclo all'unisono.

3. LA MATEMATICA ROTTA (Damping sui calcoli parziali)
   - Originale: Guarda cosa faceva l'autore dentro il doppio ciclo for di
     moltiplicazione:
         localpr[rowind[j]] += val[j]*prold[col];
         localpr[rowind[j]] = localpr[rowind[j]]*damp1[rowind[j]] + damp2[rowind[j]];
     È un errore matematico madornale. Il Damping e il Teleporting venivano
     applicati sulle somme parziali. Se 3 thread diversi trovavano archi che
     puntavano al Nodo 5, il Nodo 5 riceveva il bonus del teleporting (0.15/N)
     per ben 3 volte, sballando completamente le probabilità.
   - Tua Versione: Hai tolto questa logica dai thread worker. I worker fanno
     solo la moltiplicazione pura. Il Master, dopo aver unito le somme parziali
     in un unico vettore globale (prnew), applica la formula del damping e
     dei dangling nodes una sola volta per nodo.

4. I DANGLING NODES (completamente assenti)
   - Originale: Oltre a far partire tutti i nodi da 0.25 (creando massa dal
     nulla, 0.25 * 4 = 1.0, ma se i nodi sono 6 diventa 1.5), l'originale non
     considerava minimamente la perdita di probabilità nei nodi pozzo.
   - Tua Versione: Hai integrato il fix matematico distribuendolo sui thread:
     ognuno calcola la local_dangling_mass per le proprie colonne, usa un mutex
     dedicato (dangling_mutex) per accumularla in modo sicuro, e il Master
     infine la usa per redistribuire la probabilità.
================================================================================
*/

/*
================================================================================
# 🧠 ARCHITETTURA DEL PARALLELISMO PTHREADS E SINCRONIZZAZIONE
================================================================================
Questa funzione esegue la Power Iteration del PageRank sfruttando un pool
di thread basati su memoria condivisa. Risolve numerosi bug critici e problemi
matematici presenti nella versione base dell'algoritmo.

## 1. RIPARTIZIONE DEL LAVORO (1D Column Partitioning)
Essendo la memoria condivisa, non c'è bisogno di duplicare o inviare il grafo.
Tutti i thread vedono le strutture dati globali (colptr, rowind, val, prold).
Dividiamo semplicemente gli indici delle colonne in base all'ID del thread ('tid').
Se abbiamo 6000 colonne e 6 thread, ogni thread calcolerà la propagazione
del PageRank per un blocco esclusivo di 1000 nodi.

## 2. GESTIONE DELLA MEMORIA E DATI LOCALI
Ogni thread ha un vettore 'localpr' privato azzerato ad ogni ciclo.
- I thread moltiplicano la propria porzione di matrice per il vettore 'prold'.
- I risultati si accumulano in 'localpr'. Questo evita di bloccare continuamente
  il vettore globale 'prnew' con un mutex per ogni singola moltiplicazione,
  il che distruggerebbe le performance (Lock Contention).

## 3. IL FLUSSO DI SINCRONIZZAZIONE A FASI (Il core del fix)
La sincronizzazione originale causava Deadlock e Race Conditions.
Questa versione utilizza un pattern a stati strettamente controllato:

* FASE A: Accumulo Massa Dangling
  Ogni thread ispeziona le proprie colonne per scovare i Dangling Nodes (sum == 0).
  Aggiornano la variabile condivisa 'global_dangling_mass' usando un lock
  dedicato ('dangling_mutex').

* FASE B: Fusione in 'prnew'
  Finiti i calcoli matematici, ogni thread riversa il proprio 'localpr'
  nel vettore globale 'prnew'. Per evitare sovrascritture, si usa 'add_mutex'
  solo per questa rapida operazione di somma.

* FASE C: BARRIERA 1 ('our_barrier')
  I worker si fermano. Nessuno può procedere finché tutti non hanno
  consegnato i propri risultati parziali al MASTER.

* FASE D: Computazione Singola (MASTER)
  Il thread MASTER (tid == 0) agisce da solo. Prende il vettore grezzo,
  applica la formula finale (Damping + Dangling Redistribution + Teleporting),
  calcola l'errore ('norm') rispetto alla iterazione precedente e copia i
  nuovi valori nel vettore vecchio.
  FIX: L'applicazione del damping qui, a valle, garantisce la correttezza
  matematica che l'originale violava applicandola sui vettori parziali.

* FASE E: Risveglio (Broadcast)
  Il MASTER usa 'pthread_cond_broadcast' (non 'signal' che causava deadlock
  nell'originale risvegliando un solo thread) per dire ai worker dormienti
  che il calcolo globale è finito.

* FASE F: BARRIERA CRITICA 2 ('our_barrier2')
  Il fix fondamentale contro le Race Conditions. Garantisce che TUTTI i worker
  siano svegli e abbiano resettato i loro dati prima che chiunque proceda
  a leggere la variabile condivisa 'norm'. Questo impedisce a un thread
  estremamente veloce di ricominciare il ciclo e inquinare le variabili globali
  mentre altri thread stanno ancora finendo l'iterazione corrente.

## 4. CONDIZIONE DI USCITA THREAD-SAFE
La variabile 'norm' viene letta in 'current_norm' sotto la protezione
di 'norm_mutex'. Ora tutti i thread controllano la condizione del
while(current_norm > ERR) su un valore coerente, e terminano all'unisono.
================================================================================
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifdef _WIN32
    #include <windows.h>
    double get_time() {
        LARGE_INTEGER t, f;
        QueryPerformanceCounter(&t);
        QueryPerformanceFrequency(&f);
        return (double)t.QuadPart / f.QuadPart;
    }
#else
#include <time.h>
double get_time() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + t.tv_nsec / 1e9;
    }
#endif

#define NODES 6
#define EDGES 19



//#define NODES 685230
//#define EDGES 7600595

#define CORES 2
#define MASTER 0
#define FILEPATH "C:\\Users\\UTENTE\\Desktop\\Calcolo Parallelo\\Ipotesi Progetto\\Pagerank\\pagerank\\dataset\\data0.dat"
#define ERR 0.00001
#define DAMPING 0.85

double sqrt(double x);
void *mat_vec(void *);

double *val, *prold, *prnew, *damp1, *damp2, *diff;
int *rowind, *colptr, *sum;

pthread_mutex_t add_mutex;
pthread_mutex_t wait_mutex;
pthread_mutex_t norm_mutex;
pthread_barrier_t our_barrier;
pthread_barrier_t our_barrier2;
pthread_cond_t proceed_cv;

double norm, norm_sq;
int var_wait;

/* =========================================================================
   NUOVA VARIABILE GLOBALE PER I DANGLING NODES
   I worker thread calcoleranno parzialmente la massa dei dangling nodes
   delle proprie porzioni di grafo. Il MASTER farà la riduzione.
   Serve un mutex per proteggerla durante l'accumulo.
   ========================================================================= */
double global_dangling_mass;
pthread_mutex_t dangling_mutex;

int main(int argc, char *argv[])
{
    printf("Program start\n");
    pthread_t p_threads[CORES];

    // Inizializzazione sincronizzazione
    pthread_mutex_init(&add_mutex, NULL);
    pthread_mutex_init(&wait_mutex, NULL);
    pthread_mutex_init(&norm_mutex, NULL);
    pthread_mutex_init(&dangling_mutex, NULL); // NUOVO MUTEX
    pthread_barrier_init(&our_barrier, NULL, CORES);
    pthread_barrier_init(&our_barrier2, NULL, CORES);
    pthread_cond_init(&proceed_cv, NULL);

    FILE *fp;
    int colindex, link, i, j=0, col, colmatch=0, localsum=0;
    long t;
    int k, rc=0;

    val = (double*)calloc(EDGES, sizeof(double));
    rowind = (int*)calloc(EDGES, sizeof(int));
    colptr = (int*)calloc(NODES+1, sizeof(int));

    int co, index;

    prold = (double*)malloc(NODES*sizeof(double));
    prnew = (double*)calloc(NODES, sizeof(double));
    damp1 = (double*)malloc(NODES*sizeof(double));
    damp2 = (double*)malloc(NODES*sizeof(double));
    diff = (double*)calloc(NODES, sizeof(double));
    sum = (int*)calloc(NODES, sizeof(int));

    for(i=0; i<NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = DAMPING;
        damp2[i] = (1.0 - DAMPING)/NODES;
    }
    printf("initialization complete\n");

    norm = 1.0;
    var_wait = 0;

    // Lettura File
    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        return 1;
    }
/*
    for(i = 0; i<EDGES; i++) {
        int ret = fscanf(fp, "%d %d", &colindex, &link);
        if (ret != 2) {
            printf("ERROR: fscanf failed at line %d, ret=%d\n", i, ret);
            break;
        }
        colindex = colindex - 1;
        link = link - 1;
        rowind[i] = link;
        if(colmatch==colindex) {
            localsum += 1;
        }
        else {
            sum[j] = localsum;
            colptr[j+1] = colptr[j] + localsum;
            localsum = 1;
            j += 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }
    sum[j] = localsum;
    colptr[j+1]= EDGES;
    fclose(fp);
    */

    // === NUOVO PARSER SEQUENZIALE ROBUSTO (Allineato a MPI) ===
    localsum = 0;
    colmatch = -1;

    for (i = 0; i < EDGES; i++) {
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Errore lettura file alla riga %d\n", i + 1);
            fclose(fp);
            return 1;
        }

        colindex = colindex - 1;
        link     = link - 1;
        rowind[i] = link;

        if (i == 0) {
            colmatch = colindex;
            localsum = 1;
        } else if (colmatch == colindex) {
            localsum += 1;
        } else {
            sum[colmatch] = localsum; // 'sum' nel sequenziale tiene traccia degli out-degree
            for (int c = colmatch + 1; c <= colindex; c++) {
                colptr[c] = colptr[colmatch] + localsum;
            }
            localsum = 1;
            colmatch = colindex;
        }
        val[i] = 1.0;
    }

    if (EDGES > 0) {
        sum[colmatch] = localsum;
        for (int c = colmatch + 1; c <= NODES; c++) {
            colptr[c] = EDGES;
        }
    }
    fclose(fp);
    // ==========================================================

    // Normalizzazione
    index = 0;
    for(i = 0; i<NODES; i++) {
        co = sum[i];
        for(j = index; j < index+co; j++) {
            val[j] = val[j]/co;
        }
        index += co;
    }

    printf("\n=== STRUTTURA CSC COSTRUITA ===\n");


    // INIZIO COMPUTAZIONE PARALLELA
    double start_time = get_time();

    for(t=0; t<CORES; t++) {
        rc = pthread_create(&p_threads[t], NULL, mat_vec, (void*)t);
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for(t= 0; t<CORES; t++) {
        pthread_join(p_threads[t], NULL);
    }

    double end_time = get_time();
    double tempo_parallelo = end_time - start_time;


    printf("\n=============================================\n");
    printf("TEMPO DELLA POWER ITERATION (%d THREAD): %.6f secondi\n", CORES, tempo_parallelo);
    printf("=============================================\n");

    // Validazione matematica finale
    double sum_pr = 0.0;
    for(int i=0; i<NODES; i++) {
        sum_pr += prold[i];
    }
    printf("VERIFICA MATEMATICA: Somma PR = %.10f\n", sum_pr);
    printf("=============================================\n");


    /* Clean up and exit */
    pthread_mutex_destroy(&add_mutex);
    pthread_mutex_destroy(&wait_mutex);
    pthread_mutex_destroy(&norm_mutex);
    pthread_mutex_destroy(&dangling_mutex); // NUOVO
    pthread_barrier_destroy(&our_barrier);
    pthread_barrier_destroy(&our_barrier2);
    pthread_cond_destroy(&proceed_cv);
    pthread_exit(NULL);
}

// =========================================================================
// POWER ITERATION (Pthreads)
// =========================================================================
void *mat_vec(void *rank) {
    long tid = (long)rank;

    int i, j, col, local_col, my_first_col, my_last_col;
    double *localpr = (double*)calloc(NODES, sizeof(double));
    double current_norm;

    // Divisione del lavoro per partizionamento 1D (per colonne)
    if (tid==0) {
        local_col = NODES/CORES + NODES%CORES;
        my_first_col = 0;
        my_last_col = my_first_col + local_col - 1;
    }
    else {
        local_col = NODES/CORES;
        my_first_col = tid*local_col + NODES%CORES;
        my_last_col = my_first_col + local_col - 1;
    }

    do {
        // --- STEP 0: Reset Iniziale ---
        // Il MASTER resetta la variabile globale dei dangling nodes per il nuovo ciclo
        if (tid == MASTER) {
            global_dangling_mass = 0.0;
        }

        // Sincronizziamo per essere sicuri che global_dangling_mass sia zero prima di procedere
        pthread_barrier_wait(&our_barrier2);

        // --- STEP 1A: Calcolo locale Dangling Nodes ---
        // Ogni thread ispeziona SOLO le sue colonne per trovare nodi pozzo (sum == 0)
        double local_dangling_mass = 0.0;
        for (col = my_first_col; col <= my_last_col; col++) {
            if (sum[col] == 0) {
                local_dangling_mass += prold[col];
            }
        }

        // --- STEP 1B: Riduzione Dangling Mass ---
        // I thread accumulano il proprio totale parziale nella variabile globale protetta
        if (local_dangling_mass > 0.0) {
            pthread_mutex_lock(&dangling_mutex);
            global_dangling_mass += local_dangling_mass;
            pthread_mutex_unlock(&dangling_mutex);
        }

        // --- STEP 1C: Calcolo contributo al PageRank (Moltiplicazione Sparsa) ---
        // Ogni thread itera sui nodi uscenti di sua competenza
        for(col=my_first_col; col<=my_last_col; col++) {
            for(j = colptr[col]; j<colptr[col+1]; j++) {
                localpr[rowind[j]] += val[j]*prold[col];
            }
        }

        // --- STEP 2: Somma thread-safe dei contributi ---
        // Assemblaggio del vettore prnew globale
        for(i=0; i<NODES; i++) {
            if (localpr[i]!=0.0) {
                pthread_mutex_lock(&add_mutex);
                prnew[i] += localpr[i];
                pthread_mutex_unlock(&add_mutex);
            }
        }

        // --- STEP 3: BARRIERA 1 ---
        // Aspettiamo che TUTTI i thread abbiano versato i loro calcoli in prnew
        // e abbiano finito di sommare a global_dangling_mass
        pthread_barrier_wait(&our_barrier);

        // --- STEP 4: Computazione Master (Damping e Convergenza) ---
        if (tid==MASTER) {
            pthread_mutex_lock(&norm_mutex);

            // Calcolo la frazione di redistribuzione globale dei nodi pozzo
            double redistribution = (global_dangling_mass * DAMPING) / NODES;

            var_wait = 0;
            norm_sq = 0.0;

            for(i=0; i<NODES; i++) {
                // Nuova formula PageRank con aggiunta redistribuzione nodi pozzo
                prnew[i] = (prnew[i] * damp1[i]) + damp2[i] + redistribution;

                diff[i] = prnew[i] - prold[i];
                norm_sq += diff[i]*diff[i];

                prold[i] = prnew[i];
            }

            norm = sqrt(norm_sq);

            pthread_mutex_unlock(&norm_mutex);

            memset(prnew, 0, NODES*sizeof(double));

            var_wait = 1;
            pthread_cond_broadcast(&proceed_cv);
        }
        else {
            // I Worker aspettano che il Master finisca il calcolo finale e l'errore
            pthread_mutex_lock(&wait_mutex);
            while(!var_wait) {
                pthread_cond_wait(&proceed_cv, &wait_mutex);
            }
            pthread_mutex_unlock(&wait_mutex);
        }

        // --- STEP 5: Reset strutture locali ---
        memset(localpr, 0, NODES*sizeof(double));

        // --- STEP 6: BARRIERA 2 ---
        // Evitiamo che un thread veloce ricominci il ciclo do-while e modifichi variabili
        // globali (come global_dangling_mass al MASTER) prima che i lenti abbiano letto norm
        pthread_barrier_wait(&our_barrier);

        // --- STEP 7: Check della convergenza ---
        pthread_mutex_lock(&norm_mutex);
        current_norm = norm;
        pthread_mutex_unlock(&norm_mutex);

    } while(current_norm > ERR);

    free(localpr);
    pthread_exit(NULL);
}