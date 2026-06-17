// Command to compile:
// gcc pthread/pr_pthread_opt.c libraries/data.c libraries/measure.c -o pthread/pr_pthread_opt -Ilibraries -lpthread -lm
/*
================================================================================
# TITOLO: PageRank (Power Iteration) con Pthreads
# VERSIONE 5: Privatizzazione della Memoria (Architettura ZERO MUTEX)
================================================================================

## 📌 INTRODUZIONE PER NON ESPERTI (Cosa abbiamo fatto?)
Immagina di avere un ufficio con 6 impiegati (i 6 Core della CPU) che devono scrivere
delle informazioni su un unico grande tabellone comune di 685.230 righe (il vettore prnew).

* **La versione con 1 Mutex Singolo:** È come costringere tutti e 6 gli impiegati a usare un
  unico pennarello. Solo chi ha il pennarello scrive, gli altri 5 stanno fermi in coda ad aspettare.
  Un disastro per il tempo di esecuzione (il parallelismo scompare).

* **La versione con Array di Mutex (685.230 lock):** È come mettere un lucchetto su ogni singola
  riga del tabellone. Se l'impiegato A scrive sulla riga 10 e l'impiegato B sulla riga 50, possono
  lavorare insieme. Ma l'azione di aprire e chiudere continuamente milioni di lucchetti
  (l'overhead dei mutex) e il costo di gestire 27 Megabyte di soli lucchetti rallenta la CPU.

* **La NUOVA versione (Privatizzazione della Memoria):** Abbiamo eliminato TUTTI i lucchetti.
  Ogni impiegato riceve un suo blocco note personale di 685.230 righe (Memoria Privatizzata).
  Ognuno scrive sul proprio blocco alla massima velocità senza MAI guardare o disturbare gli altri.
  Alla fine dell'iterazione, i 6 blocchi note vengono uniti (Riduzione Parallela) in un unico risultato.

* **

---

## ⚡ IN COSA CONSISTE L'OTTIMIZZAZIONE DELLA PRIVATIZZAZIONE?

1. **Scrittura a Contesa Zero (Lock-Free SpMV):** Durante il calcolo dei collegamenti del grafo, la CPU scrive direttamente nell'area dedicata
   al singolo thread. L'assenza di mutex permette ai core di lavorare al 100% delle loro capacità
   senza "Context Switch" (le pause forzate che il Sistema Operativo impone ai thread in attesa).

2. **La Riduzione Parallela "A Falcate" (Strided Access):**
   Invece di far sommare tutto a un unico thread Master (che creerebbe un collo di bottiglia sequenziale),
   il lavoro di unione dei dati viene diviso equamente. Se ci sono 6 core, il Core 0 unisce le righe
   da 0 a 100.000, leggendo il contributo per quelle righe da tutti e 6 i blocchi note privati.

3. **Calcolo della Convergenza Collettivo:**
   Anche il calcolo dell'errore (la norma quadratica per capire quando fermarsi) è stato privatizzato.
   Ogni core calcola l'errore della sua porzione di nodi, lo scrive in un piccolo array condiviso
   (`part_norm`) e poi tutti i core, autonomamente e in parallelo, leggono questo mini-array e
   calcolano la radice quadrata finale. In questo modo si elimina la necessità di fermare i thread
   con variabili di condizione.

---

## 📊 CONFRONTO DIALETTICO TRA LE VERSIONI

| Parametro | Versione con 1 Mutex | Versione con Array di Mutex | Versione 5 (Privatizzazione) |
| :--- | :--- | :--- | :--- |
| **Tempo di Esecuzione** | 🔴 Lentissimo (Soffre di code continue) | 🟡 Buono (Molto meglio del singolo mutex) | 🚀 Massimo (Sfrutta i core al 100%) |
| **Consumo di RAM** | 🟢 Minimo (Nessuna struttura extra) | 🔴 Elevato (~27 MB buttati in soli lucchetti) | 🟡 Ottimizzato (~33 MB spesi in memoria utile) |
| **Complessità Codice** | 🟢 Molto Semplice | 🟡 Medio (Gestione dinamica dei lucchetti) | 🔴 Alta (Richiede la riprogettazione degli indici) |
| **Impatto sulla Cache** | 🟢 Ottimo | 🔴 Pessimo (I lucchetti distruggono la Cache L3) | 🟢 Buono (I dati sono contigui e lineari) |

================================================================================
/*
================================================================================
# 📐 GEOMETRIA DEL PARALLELISMO: SUDDIVISIONE STATICA E ARCHITETTURA PEER-TO-PEER
================================================================================

## 🗺️ Come avviene la suddivisione del lavoro? (Colonne e Righe)
Per evitare che i thread si calpestino i piedi o saltino dei nodi, usiamo una
strategia chiamata "Suddivisione a Blocchi Statici" (Static Block Distribution):

1. `int col_chunks = NODES / CORES;`
   Prendiamo il numero totale di nodi e lo dividiamo per il numero di core (es. 685.230 / 6).
   Questo ci dà la dimensione del "pacchetto base" di colonne (o righe) da assegnare.

2. `int col_start = tid * col_chunks;`
   Ogni thread calcola il proprio punto di partenza moltiplicando il proprio ID (tid)
   per il pacchetto base. Il Thread 0 partirà da 0, il Thread 1 da 114.205, ecc.

3. `int col_end = (tid == CORES - 1) ? NODES : (tid + 1) * col_chunks;`
   Questa è la riga fondamentale per gestire il "resto" della divisione intera.
   Se un grafo non è perfettamente divisibile per 6, avanzerebbero dei nodi.
   La formula dice: "Se sei l'ultimo thread, fatti carico di tutto il resto fino alla fine (NODES).
   Se non sei l'ultimo, fermati dove inizia il thread successivo".

Applichiamo questa identica divisione sia alle COLONNE (nella Fase SpMV di calcolo dei link)
sia alle RIGHE/NODI (nesta fase di Riduzione/unione finale dei dati).

## 🤝 Addio "Dittatura del Master": L'Architettura Peer-to-Peer (P2P)
Nelle versioni precedenti avevi un modello "Master-Worker" (Capo-Operai): i thread operai
lavoravano, poi si spegnevano, e il thread Master da solo calcolava la norma, azzerava
i vettori e dava il via libera. Questo creava un collo di bottiglia sequenziale.

In questa Versione 5 siamo passati a un modello "Peer-to-Peer" (Paritetico).
Il Master (Thread 0) ESISTE ancora fisicamente, ma le sue mansioni sono cambiate:
* Non è più un "capo" che centralizza il lavoro, ma un operaio esattamente come gli altri.
* Lavora sulla sua fetta di colonne e sulla sua fetta di righe.
* Il calcolo della norma globale e il controllo del ciclo `while` vengono fatti da TUTTI
  i thread contemporaneamente in modo autonomo.
Nessun core rimane mai spento ad aspettare gli ordini del Master. Questo è il segreto
delle massime prestazioni.
================================================================================
/*
================================================================================
# 📊 TABELLA RIASSUNTIVA: BILANCIAMENTO GEOMETRICO (FASE B vs FASE C)
================================================================================
Questo schema evidenzia come i thread cambino completamente il loro focus
geometrico durante l'esecuzione dell'algoritmo per garantire la massima efficienza
e l'assenza assoluta di conflitti di scrittura (Race Conditions).

| FASE DEL CODICE             | RIPARTIZIONE BASATA SU... | TIPO DI NODI ASSEGNATI    | RUOLO OPERATIVO DEL THREAD                                         |
| :-------------------------- | :------------------------ | :------------------------ | :----------------------------------------------------------------- |
| FASE B (Calcolo dei link)    | COLONNE (Matrice CSC)     | NODI SORGENTE (Chi vota)  | Analizza le pagine sorgente a lui assegnate e "spinge" (Scatter) i voti nei fogli privati. |
| FASE C (Riduzione Parallela) | RIGHE (Vettore Finale)    | NODI DESTINAZIONE (Riceve)| Fa il guardiano della sua fetta di pagine di arrivo. Somma i voti dai 6 fogli e scrive su prnew. |

## 💡 PERCHÉ QUESTO CAMBIO DI PASSO È FONDAMENTALE?
1. Nella FASE B dividiamo per COLONNE perché il dataset è strutturato in formato CSC
   (Compressed Sparse Column). Questo permette una lettura sequenziale e velocissima della
   matrice, ma genera "voti sparsi" verso destinazioni (righe) totalmente imprevedibili.
2. Nella FASE C passiamo a dividere il lavoro per RIGHE (Nodi Destinazione). In questo modo
   ogni thread è l'unico e l'esclusivo proprietario di una porzione del vettore finale 'prnew'.
   Grazie a questo isolamento geometrico, la scrittura finale avviene a zero contesa,
   eliminando la necessità di utilizzare pesanti Lock Mutex.
================================================================================
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "../libraries/data.h"
#include "../libraries/measure.h"

#define CORES 6          // Numero di thread paralleli che lavoreranno insieme
#define MASTER 0         // Identificativo del thread principale

double sqrt(double x);
void *mat_vec(void *);

// Array globali standard per l'algoritmo PageRank in formato CSC (Compressed Sparse Column)
double *val, *prold, *prnew, *damp1, *damp2, *diff;
int *rowind, *colptr, *sum;

/* SPIEGAZIONE PRIVATIZZAZIONE:
   all_prnew è una matrice gigante "linearizzata" (trattata come un unico vettore lungo).
   La sua dimensione è: (Numero di Core * Numero di Nodi).
   In pratica, contiene 6 vettori prnew consecutivi, uno per ogni thread.
   - Il Thread 0 scriverà da all_prnew[0] a all_prnew[685229]
   - Il Thread 1 scriverà da all_prnew[685230] a all_prnew[1370459]
   ... e così via. Nessun thread toccherà mai l'area di un altro durante la scrittura!
*/
double *all_prnew;

/* Questo mini-array serve per calcolare l'errore globale senza lucchetti.
   Ogni core scriverà il suo errore parziale nella cella corrispondente al proprio ID (tid).
*/
double part_norm[CORES];

// Strumenti di sincronizzazione: Usiamo SOLO BARRIERE. I Mutex (lock) sono stati eliminati.
pthread_barrier_t our_barrier;       // Barriera 1: Aspetta la fine della scrittura privata
pthread_barrier_t our_barrier2;      // Barriera 2: Aspetta la fine del calcolo delle norme locali
double norm, norm_sq, err = 0.00001; // Soglia di precisione per la convergenza

int NODES;
int EDGES;
const char* FILEPATH;

int main(int argc, char *argv[])
{
    printf("Program start (Privatized Version)\n");
    pthread_t p_threads[CORES];

    // Inizializziamo le barriere specificando che dovranno scattare solo quando 6 core (CORES) le avranno raggiunte
    pthread_barrier_init(&our_barrier, NULL, CORES);
    pthread_barrier_init(&our_barrier2, NULL, CORES);

    GraphType graph_type = GRAPH_MEDIUM;
    const Graph* graph = get_graph(graph_type);

    NODES = graph->nodes;
    EDGES = graph->edges;
    FILEPATH = graph->filepath;

    FILE *fp;
    int colindex, link, i, j=0, c, colmatch=0, localsum=0;
    long t;
    int rc=0;

    // Allocazione standard delle strutture dati per il grafo
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

    /* ALLOCAZIONE DELL'ARRAY DI PRIVATIZZAZIONE:
       Chiediamo al sistema operativo uno spazio continuo in memoria RAM.
       6 core * 685230 nodi * 8 byte (double) = circa 32.8 Megabyte.
       Questa allocazione avviene nell'Heap, evitando il rischio di mandare in crash lo Stack.
     */
    all_prnew = (double*)calloc(CORES * NODES, sizeof(double));
    if (!all_prnew) {
        fprintf(stderr, "Errore: Memoria insufficiente per all_prnew\n");
        return 1;
    }

    // Inizializzazione dei valori di partenza per il PageRank (equidistribuzione della probabilità)
    for(i=0; i<NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = 0.85;            // Damping factor classico (85% di probabilità di seguire i link)
        damp2[i] = 0.15/NODES;      // 15% di probabilità di saltare a una pagina casuale
    }
    printf("initialization complete\n");

    norm = 1.0;

    // Apertura e lettura del file di dataset (costruzione della struttura a matrice sparsa CSC)
    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        free(all_prnew);
        return 1;
    }

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

    // Normalizzazione delle colonne (ogni link uscente distribuisce una frazione equa del valore della pagina)
    index = 0;
    for(i = 0; i<NODES; i++) {
        co = sum[i];
        for(j = index; j < index+co; j++) {
            val[j] = val[j]/co;
        }
        index += co;
    }

    printf("\n=== INIZIO COMPUTAZIONE PARALLELA (ZERO MUTEX) ===\n");

    // Fissiamo il tempo di partenza reale prima di avviare i calcoli intensivi
    double start_time = get_time();

    // Creazione effettiva dei 6 Thread lavoratori
    for(t=0; t<CORES; t++) {
        rc = pthread_create(&p_threads[t], NULL, mat_vec, (void*)t);
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // Il programma principale (Main) si ferma e aspetta che tutti i 6 thread abbiano terminato l'algoritmo
    for(t= 0; t<CORES; t++) {
        pthread_join(p_threads[t], NULL);
    }

    // Calcolo del tempo finale e della durata complessiva
    double end_time = get_time();
    double tempo_parallelo = end_time - start_time;

    printf("\n=============================================\n");
    printf("TEMPO DELLA POWER ITERATION (%d THREAD): %.6f secondi\n", CORES, tempo_parallelo);
    printf("=============================================\n");

    // Validazione matematica: la somma dei punteggi di PageRank di tutti i nodi deve essere circa 1.0
    double sum_pr = 0.0;
    for(i=0; i<NODES; i++) {
        sum_pr += prold[i];
    }
    printf("Somma finale di controllo PR = %.10f (Validazione superata se vicino a 1.0)\n", sum_pr);

    /*
    printf("\n--- VETTORE PAGERANK FINALE ---\n");
    for (i = 0; i < NODES; i++) {
        printf("Nodo %d: %.6f\n", i + 1, prold[i]);
    }
    printf("=============================================\n");
    */

    // Liberazione totale della memoria allocata per non lasciare residui nella RAM del sistema
    free(all_prnew);
    free(val); free(rowind); free(colptr);
    free(prold); free(prnew); free(damp1); free(damp2); free(diff); free(sum);

    pthread_barrier_destroy(&our_barrier);
    pthread_barrier_destroy(&our_barrier2);
    return 0;
}

// ================================================================================
// FUNZIONE CORE DEI THREAD (La logica matematica parallela)
// ================================================================================
void *mat_vec(void *rank) {
    long tid = (long)rank; // Otteniamo l'ID numerico del thread corrente (da 0 a 5)

    int i, j, col;
    double current_norm;

    /* 1. DIVISIONE DELLE COLONNE (Fase SpMV):
       Dividiamo le colonne (i nodi sorgente dei link) equamente tra i 6 thread.
       Ogni thread sa esattamente da quale colonna iniziare (`col_start`) e dove finire (`col_end`).
    */
    int col_chunks = NODES / CORES; // calcolo le colonne che spettano ad ogni thread
    int col_start = tid * col_chunks;
    int col_end = (tid == CORES - 1) ? NODES : (tid + 1) * col_chunks; // se sei l'ultimo thread il punto di fine è il valore dell'ultimo nodo, altrimenti il punto di fine è l'inizio del thread successivo

    /* 2. DIVISIONE DEI NODI/RIGHE (Fase di Riduzione finale):
       Per unire i dati dei 6 blocchi note alla fine, dividiamo i nodi destinazione.
       Ogni thread si occuperà di ricostruire il PageRank finale solo per una fetta di nodi (`node_start` - `node_end`).
    */
    int node_chunks = NODES / CORES; // calcolo le righe che spettano ad ogni thread
    int node_start = tid * node_chunks;
    int node_end = (tid == CORES - 1) ? NODES : (tid + 1) * node_chunks;

    /* PUNTARE ALLA PROPRIA PAGINA PRIVATA:
       Spostiamo l'inizio del puntatore all'interno della matrice gigante `all_prnew`.
       Il Thread 0 punterà all'inizio (indice 0), il Thread 1 punterà all'indice (1 * 685230), ecc.
       `my_private_prnew` è il "blocco note personale" di questo thread.
    */
    double *my_private_prnew = &all_prnew[tid * NODES];

    // Inizio del grande ciclo della Power Iteration (continua finché non converge)
    do {
        /* [Fase A] AZZERAMENTO DELLA MEMORIA PRIVATA:
           Prima di calcolare, ogni thread cancella i vecchi dati esclusivamente dal proprio
           blocco note privato. Questa operazione è velocissima (`memset`) e non ha conflitti.
        */
        memset(my_private_prnew, 0, NODES * sizeof(double));

        /* [Fase B] LA MOLTIPLICAZIONE SPARSA (SpMV) PRIVATA:
           I thread scorrono le colonne assegnate. Quando trovano un collegamento verso un nodo
           destinazione (`rowind[j]`), scrivono il contributo.
           NOTARE BENE: Scrivono dentro `my_private_prnew`. Poiché l'array è privato,
           NON C'È BISOGNO DI NESSUN LOCK MUTEX! I core viaggiano alla velocità della luce.
        */
        for(col = col_start; col < col_end; col++) {
            for(j = colptr[col]; j < colptr[col+1]; j++) {
                my_private_prnew[rowind[j]] += val[j] * prold[col];
            }
        }

        /* BARRIERA 1: IL PUNTO DI RACCOLTA
           Qui tutti i thread devono fermarsi. Non possiamo fare l'unione finché anche l'ultimo
           dei thread non ha finito di scrivere tutti i contributi nel proprio blocco note.
        */
        pthread_barrier_wait(&our_barrier);

       /* * ====================================================================
         * [FASE C] RIDUZIONE PARALLELA & CALCOLO DELLA NORMA LOCALE
         * ====================================================================
         * OBIETTIVO: Raccogliere i frammenti di punteggio sparsi nelle memorie
         * private (Fase B), calcolare il PageRank definitivo e l'errore di
         * convergenza, tutto in parallelo e senza mai usare mutex.
         */
        double local_norm_sq = 0.0; // Accumulatore locale dell'errore quadratico del thread

        // Ogni thread si occupa esclusivamente della sua fetta di nodi destinazione (Righe)
        for(i = node_start; i < node_end; i++) {

            // Inizializza il "semilavorato": conterrà la somma dei voti grezzi in ingresso
            double raw_pagerank = 0.0;

            /* * IL CUORE DELLA RIDUZIONE: Lettura "Verticale" (Strided Access)
             * Il thread scorre le memorie private di TUTTI i core (t) cercando la riga 'i'.
             * Somma i contributi che ogni singolo core ha registrato per questo specifico nodo.
             */
            int t;
            for(t = 0; t < CORES; t++) {
                raw_pagerank += all_prnew[t * NODES + i];
            }

            /* * APPLICAZIONE DELLA FORMULA DI PAGERANK
             * Il valore grezzo viene smorzato dall'85% (damp1) e sommato al 15% residuo
             * distribuito equamente (damp2). Il risultato è scritto nel tabellone ufficiale.
             */
            prnew[i] = raw_pagerank * damp1[i] + damp2[i];

            /* * CALCOLO DELLO SCARTO (ERRORE)
             * Calcola la differenza (d) tra il PageRank appena scoperto e quello del ciclo precedente.
             * Elevando il valore al quadrato (d * d) evitiamo i numeri negativi e prepariamo
             * i dati per la Norma Euclidea (L2 Norm).
             */
            double d = prnew[i] - prold[i];
            local_norm_sq += d * d;

            /* * AGGIORNAMENTO STORICO
             * Il valore nuovo diventa ufficialmente "vecchio", preparando la cella
             * per l'inizio del prossimo ciclo della Power Iteration.
             */
            prold[i] = prnew[i];
        }

        /* Ogni thread salva l'errore quadratico accumulato dalla sua porzione di nodi
           nella cella dell'array globale dedicata al suo ID. Senza usare mutex!
        */
        part_norm[tid] = local_norm_sq;

        /* BARRIERA 2: SICUREZZA DEI DATI DELLA NORMA
           Ci fermiamo di nuovo. Dobbiamo essere certi che tutti e 6 i thread abbiano scritto
           il proprio errore parziale dentro l'array `part_norm` prima di poterlo leggere.
        */
        pthread_barrier_wait(&our_barrier2);

        /* [Fase D] CALCOLO GLOBALE DELLA CONVERGENZA (Lock-Free):
           Invece di far calcolare la radice globale solo al Master, TUTTI i thread fanno un
           rapido ciclo di 6 passaggi sommandosi le norme parziali a vicenda ed eseguendo `sqrt()`.
           Estrarre una radice quadrata in locale sui registri della CPU costa zero e ci risparmia
           l'uso di un costoso Mutex e di una terza barriera di sincronizzazione.
        */
        double total_norm_sq = 0.0;
        int t;
        for(t = 0; t < CORES; t++) {
            total_norm_sq += part_norm[t];
        }
        current_norm = sqrt(total_norm_sq);

    } while(current_norm > err); // Se l'errore globale è maggiore della soglia, tutti i thread ricominciano il ciclo insieme

    // Quando l'errore scende sotto la soglia, tutti i thread escono contemporaneamente e in modo pulito
    pthread_exit(NULL);
}