/*
 * pagerank_definitivo.c
 * Calcolo del PageRank con matrice di adiacenza in formato CSC.
 * Dataset: coppie "sorgente destinazione" senza righe di intestazione.
 * Compilazione:  gcc -o pagerank pagerank_definitivo.c -lm
 * Esecuzione:    ./pagerank data0.dat
 */

/* ── riga 9-12: inclusione librerie ────────────────────────────────────────
   Il preprocessore C sostituisce ogni #include con il contenuto
   del file .h corrispondente prima della compilazione vera e propria.
   Questo inserisce nel codice le "firme" (dichiarazioni) delle funzioni
   di libreria, in modo che il compilatore sappia come usarle.

   <stdio.h>  → printf, fprintf, fscanf, fopen, fclose, FILE
   <math.h>   → sqrt, fabs   (funzioni matematiche in virgola mobile)
   <stdlib.h> → malloc, calloc, free, exit
   <string.h> → memset, memcpy
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── riga 24-28: costanti con #define ──────────────────────────────────────
   #define è una direttiva del preprocessore: ogni occorrenza del nome
   viene sostituita testualmente con il valore prima della compilazione.
   Non occupa memoria a runtime, non ha tipo, non può essere modificata.
   Vantaggio: cambiare una sola riga aggiorna tutto il programma.

   NODES    → numero di nodi (pagine) del grafo
   EDGES    → numero di archi (link) totali
   DAMPING  → fattore di smorzamento d della formula PageRank (tipicamente 0.85)
   err      → soglia di convergenza: ci fermiamo quando la norma scende sotto
*/
#define NODES   6
#define EDGES   19
#define DAMPING 0.85
#define err     0.00001
#define FILEPATH "../dataset/data0.dat"

/* ── riga 42: dichiarazione esplicita di sqrt ──────────────────────────────
   In C standard, includere <math.h> è sufficiente per usare sqrt.
   Questa riga ridichiarava la funzione esplicitamente come faceva
   l'originale (non causa errori, ma è ridondante).
   La manteniamo per restare fedeli allo stile del repository.
*/
double sqrt(double x);

/* ════════════════════════════════════════════════════════════════════════════
   FUNZIONE PRINCIPALE: main
   In C ogni programma ha esattamente un main(). L'esecuzione parte da qui.

   Firma:  int main(int argc, char *argv[])
   - int          → il main restituisce un intero al sistema operativo
                    (0 = successo, qualsiasi altro valore = errore)
   - argc         → "argument count": numero di argomenti passati da
                    riga di comando, incluso il nome del programma stesso.
                    Es: "./pagerank data0.dat" → argc = 2
   - char *argv[] → "argument vector": array di stringhe (puntatori a char).
                    argv[0] = nome del programma
                    argv[1] = primo argomento (il nome del file nel nostro caso)
   ════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /* ── riga 55: stampa di benvenuto ──────────────────────────────────────
       printf(formato, argomenti...) stampa testo formattato su stdout.
       "\n" è la sequenza di escape per il carattere newline (a capo).
    */
    printf("Program start\n");

    /* ── riga 58-62: dichiarazione variabili scalari ───────────────────────
       In C (standard C89/C90) le variabili locali si dichiarano all'inizio
       del blocco, prima di qualsiasi istruzione eseguibile.
       In C99 e successivi si possono dichiarare ovunque (come faremo
       in qualche punto più avanti per chiarezza).

       FILE *fp      → puntatore alla struttura FILE della libreria standard.
                       fopen() restituisce questo puntatore; tutte le funzioni
                       di I/O (fscanf, fclose...) lo ricevono come argomento.
       colindex      → colonna (nodo sorgente) letta dal file, 1-based
       link          → riga (nodo destinazione) letta dal file, 1-based
       i, j          → indici generici per i cicli for
       col           → indice di colonna nel loop di moltiplicazione
       colmatch      → tiene traccia dell'ultima colonna vista durante
                       la lettura del file, per capire quando cambia colonna
       localsum      → conta quanti non-zero appartengono alla colonna corrente
    */
    FILE *fp;
    int colindex, link, i, j = 0, col, colmatch = 0, localsum = 0;

    /* ── riga 64-65: variabili per normalizzazione ─────────────────────────
       co    → numero di link uscenti della colonna i (= out-degree del nodo i)
       index → indice corrente in val[] durante la normalizzazione
    */
    int co, index;

    /* ── riga 68-71: allocazione array CSC con calloc ──────────────────────
       calloc(n, size) alloca n elementi di `size` byte ciascuno nell'heap
       e azzera tutta la memoria allocata (tutti i byte a 0).
       Restituisce un puntatore void* all'inizio del blocco, che convertiamo
       con il cast al tipo puntatore desiderato.

       Differenza con malloc: malloc non azzera → memoria contiene valori
       casuali (garbage). calloc azzera → valori iniziali garantiti a 0.

       val[EDGES]        → i valori non-zero della matrice (1.0/out_degree
                           dopo normalizzazione): uno per ogni link
       rowind[EDGES]     → per ogni non-zero, la riga (nodo destinazione)
       colptr[NODES+1]   → indici di partenza di ogni colonna in val/rowind.
                           Ha NODES+1 elementi: l'ultimo (colptr[NODES])
                           contiene il totale dei non-zero (= EDGES)
    */
    double *val    = (double *) calloc(EDGES,       sizeof(double));
    int    *rowind = (int *)    calloc(EDGES,        sizeof(int));
    int    *colptr = (int *)    calloc(NODES + 1,    sizeof(int));

    /* ── riga 74-78: allocazione vettori PageRank e damping ────────────────
       malloc(n) alloca n byte senza azzerare. Usiamo malloc per i vettori
       che inizializziamo subito dopo nel ciclo for, quindi l'azzeramento
       sarebbe sprecato.

       prold[NODES]  → vettore PageRank all'iterazione corrente (t)
       prnew[NODES]  → vettore PageRank all'iterazione successiva (t+1)
       damp1[NODES]  → array del fattore d (tutti uguali a DAMPING = 0.85)
       damp2[NODES]  → array del termine (1-d)/N (tutti uguali)
       diff[NODES]   → differenza prnew - prold per calcolare la norma
    */
    double *prold = (double *) malloc(NODES * sizeof(double));
    double *prnew = (double *) calloc(NODES,  sizeof(double));
    double *damp1 = (double *) malloc(NODES * sizeof(double));
    double *damp2 = (double *) malloc(NODES * sizeof(double));
    double *diff  = (double *) calloc(NODES,  sizeof(double));

    /* ── riga 80-81: variabili per la norma ────────────────────────────────
       norm_sq → somma dei quadrati delle differenze: Σ(prnew[i]-prold[i])²
       norm    → norma L2 = √norm_sq, usata come criterio di convergenza
    */
    double norm, norm_sq;

    /* ── riga 84: array degli out-degree ───────────────────────────────────
       sum[j] = numero di link uscenti dal nodo j.
       Serve per costruire colptr e per normalizzare val[].
       calloc lo azzera: sum[j]=0 per ogni j prima di contare.
    */
    int *sum = (int *) calloc(NODES, sizeof(int));

    /* ── riga 87-93: controllo allocazioni ─────────────────────────────────
       malloc/calloc restituiscono NULL se la memoria non è disponibile
       (sistema operativo non riesce ad allocare). Dereferenziare NULL
       causa segmentation fault. Controlliamo subito e usciamo con
       messaggio diagnostico se qualcosa è andato storto.

       fprintf(stderr, ...) stampa su standard error invece di stdout:
       i messaggi di errore appaiono in console anche se l'output normale
       è rediretto su file (es: ./pagerank > risultati.txt).
    */
    if (!val || !rowind || !colptr || !prold || !prnew ||
        !damp1 || !damp2 || !diff || !sum) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        return 1;
    }

    /* ── riga 96-101: inizializzazione vettori ─────────────────────────────
       Ciclo for classico: for(inizio; condizione; incremento)
       Esegue il corpo finché la condizione è vera.
       i++ è abbreviazione di i = i + 1.

       prold[i] = 1.0/NODES  → FIX rispetto all'originale (era 0.25 fisso).
                               Il vettore PageRank iniziale deve essere una
                               distribuzione di probabilità: somma = 1.0.
                               Con NODES=6: ogni valore = 1/6 ≈ 0.1667.
                               L'originale metteva 0.25 → somma = 1.5: errato.

       damp1[i] = DAMPING     → 0.85 per ogni nodo (usato come array
       damp2[i] = (1-d)/N     → 0.15/6 ≈ 0.025  per mantenere lo stile originale)
    */
    for (i = 0; i < NODES; i++) {
        prold[i] = 1.0 / NODES;
        damp1[i] = DAMPING;
        damp2[i] = (1.0 - DAMPING) / NODES;
    }
    printf("initialization complete\n");

    /* ── riga 104-107: apertura file ───────────────────────────────────────
       Il nome del file viene preso da argv[1] se fornito, altrimenti
       si usa "data0.dat" come default (operatore ternario: condizione ? A : B).

       fopen(nome, modo) apre il file e restituisce un puntatore FILE*.
       Il modo "r" = sola lettura (read).
       Restituisce NULL se il file non esiste o non è leggibile.

       FIX: l'originale non controllava mai il valore di ritorno di fopen.
       Se il file mancava, la prima chiamata a getc(fp) o fscanf(fp,...)
       su fp=NULL causava un crash non diagnosticabile (segmentation fault).
       Ora stampiamo un messaggio chiaro e usciamo con codice errore 1.
    */
    const char *filename = (argc > 1) ? argv[1] : FILEPATH;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore: impossibile aprire il file'%s'\n", filename);
        return 1;
    }

    /* ── NOTE: blocco rimosso rispetto all'originale ───────────────────────
       L'originale conteneva qui un while loop che leggeva il file
       carattere per carattere e saltava le prime N righe di intestazione:

           while((ch=getc(fp)) != EOF) {
               if(ch == '\n') {
                   newlines += 1;
                   if (newlines == linenum - 1) { break; }
               }
           }

       Quel codice era scritto per un dataset con 4 righe di header.
       Il nostro file non ha intestazioni: inizia direttamente con i dati.
       Mantenere quel loop avrebbe consumato silenziosamente le prime 4
       righe di dati reali, costruendo una CSC completamente sbagliata
       senza alcun messaggio di errore. Rimosso integralmente.
       Rimosse anche le variabili associate: newlines, linenum, ch.
    */

    /* ── riga 125-162: costruzione della struttura CSC ─────────────────────
       Leggiamo il file una coppia alla volta: ogni riga contiene
       "colonna destinazione" e "riga destinazione" (1-based nel file).

       La logica originale costruisce colptr in modo "streaming":
       conta i non-zero per ogni colonna man mano che li legge,
       aggiornando colptr on-the-fly quando colindex cambia.
       Funziona correttamente SE i dati sono ordinati per colonna crescente
       (condizione soddisfatta dal nostro dataset).

       Variabili usate:
         colmatch  → colonna che stiamo contando attualmente (parte da 0)
         localsum  → quanti non-zero abbiamo visto per la colonna corrente
         j         → indice della colonna corrente in colptr e sum
    */
    for (i = 0; i < EDGES; i++) {

        /* ── riga 127: lettura coppia dal file ─────────────────────────────
           fscanf(fp, formato, &var1, &var2) legge dal file puntato da fp
           due interi separati da spazio/newline e li scrive nelle variabili.
           L'operatore & (indirizzo-di) passa il puntatore alla variabile:
           fscanf ha bisogno dell'indirizzo per poter SCRIVERE il valore letto.
           Restituisce il numero di conversioni riuscite (atteso: 2).

           FIX: l'originale ignorava il valore di ritorno (warning del compilatore).
           Se il file finisce prima del previsto o ha un formato diverso,
           fscanf restituisce EOF o un numero < 2. Controlliamo e usciamo
           ordinatamente invece di continuare a leggere valori spazzatura.
        */
        if (fscanf(fp, "%d %d", &colindex, &link) != 2) {
            fprintf(stderr, "Errore lettura file alla riga %d\n", i + 1);
            fclose(fp);
            return 1;
        }

        /* ── riga 133-134: conversione da 1-based a 0-based ────────────────
           Nel file i nodi sono numerati da 1 a NODES.
           In C gli array partono da 0, quindi sottraiamo 1 a entrambi.
           Dopo: colindex e link sono nel range [0, NODES-1].
        */
        colindex = colindex - 1;
        link     = link - 1;

        /* ── riga 137: salvataggio destinazione ────────────────────────────
           rowind[i] memorizza la riga (destinazione) del link i-esimo.
           I non-zero vengono inseriti in ordine di lettura del file,
           che è ordinato per colonna sorgente crescente.
        */
        rowind[i] = link;

        /* ── riga 140-150: aggiornamento colptr on-the-fly ─────────────────
           Se la colonna corrente (colindex) è la stessa dell'ultima vista
           (colmatch), siamo ancora nella stessa colonna: incrementiamo
           localsum (conta i non-zero di questa colonna).

           Se colindex è cambiato, abbiamo trovato una nuova colonna:
           - salviamo localsum nella colonna appena finita: sum[j]
           - calcoliamo colptr[j+1] = colptr[j] + localsum
             (indice in val[] dove inizia la colonna j+1)
           - resettiamo localsum a 1 (già contato il non-zero corrente)
           - aggiorniamo j (indice colonna) e colmatch (colonna corrente)

           Nota: colptr[0] = 0 è già inizializzato da calloc.
        */
        if (colmatch == colindex) {
            localsum += 1;
        } else {
            sum[j]       = localsum;
            colptr[j + 1] = colptr[j] + localsum;
            localsum     = 1;
            j           += 1;
            colmatch     = colindex;
        }

        /* ── riga 153: valore iniziale ─────────────────────────────────────
           Mettiamo 1.0 in ogni non-zero. Verrà normalizzato subito dopo
           dividendo per il numero di link uscenti della colonna (out-degree).
           Questo equivale a mettere 1/out_degree direttamente, ma l'originale
           preferisce farlo in due fasi separate per chiarezza.
        */
        val[i] = 1.0;
    }

    /* ── riga 157-158: chiusura colonna finale e del file ──────────────────
       Il loop termina senza mai incontrare un cambio di colonna per l'ultima
       colonna: quindi sum[j] e colptr[j+1] non vengono mai scritti dal ramo
       else. Li completiamo manualmente.

       sum[j]       = localsum  → out-degree dell'ultima colonna
       colptr[j+1]  = EDGES     → l'ultimo puntatore vale sempre EDGES
                                  (totale non-zero), convenzione CSC standard.

       fclose(fp) rilascia la risorsa file. Dopo questa chiamata fp non
       è più valido: non bisogna usarlo.
    */
    sum[j]      = localsum;
    colptr[j + 1] = EDGES;
    fclose(fp);

    /* ── riga 162-170: normalizzazione di val[] ─────────────────────────────
       Ogni colonna j ha sum[j] non-zero, tutti con valore 1.0.
       Il PageRank richiede che ogni colonna sommi a 1 (matrice stocastica):
       ogni elemento va diviso per sum[j] = out-degree del nodo j.
       Dopo: val[k] = 1.0 / out_degree(col)  per ogni non-zero k della col.

       index tiene traccia di dove inizia ogni colonna in val[]:
       - parte da 0
       - avanza di sum[i] ad ogni iterazione (= numero non-zero della col i)

       co = sum[i] è il numero di non-zero della colonna i.
       Il loop interno scorre val[index .. index+co-1] e divide per co.
    */
    index = 0;
    for (i = 0; i < NODES; i++) {
        co = sum[i];
        for (j = index; j < index + co; j++) {
            val[j] = val[j] / co;
        }
        index += co;
    }


   /* Aggiunta per printare la CSC costruita*/
   // ← AGGIUNGI QUESTA STAMPA CSC
   printf("\n=== STRUTTURA CSC COSTRUITA ===\n");
   printf("colptr: [");
   for (i = 0; i <= NODES; i++) {
      printf("%d", colptr[i]);
      if (i < NODES) printf(", ");
   }
   printf("]\n\n");

   printf("rowind: [");
   for (i = 0; i < EDGES; i++) {
      printf("%d", rowind[i]);
      if (i < EDGES - 1) printf(", ");
   }
   printf("]\n\n");

   printf("val (normalizzati): [");
   for (i = 0; i < EDGES; i++) {
      printf("%.6f", val[i]);
      if (i < EDGES - 1) printf(", ");
   }
   printf("]\n\n");

   printf("Somma per colonna (verifica normalizzazione):\n");
   for (col = 0; col < NODES; col++) {
      double col_sum = 0.0;
      for (int k = colptr[col]; k < colptr[col + 1]; k++) {
         col_sum += val[k];
      }
      printf("  Colonna %d (nodo %d): somma = %.10f\n", col, col, col_sum);
   }
   printf("\n");

   printf("CSC construction complete\n");
   /* FINE STAMPA CSC */

    /* ════════════════════════════════════════════════════════════════════════
       POWER ITERATION
       Ripetiamo il calcolo R(t+1) = d * M * R(t) + (1-d)/N
       finché la norma L2 della differenza ||prnew - prold|| scende
       sotto la soglia `err`.

       do { corpo } while(condizione) esegue il corpo almeno una volta,
       poi controlla la condizione. Opposto del while che controlla prima.
       ════════════════════════════════════════════════════════════════════ */

   do {
        /* ── riga 178: azzeramento prnew ───────────────────────────────────
           Prima di accumulare i contributi, azzeriamo prnew.
           memset(ptr, valore_byte, n_byte) scrive il byte `valore_byte`
           in tutti gli n_byte a partire da ptr.
           memset(..., 0, ...) azzera: scrive 0x00 in ogni byte.
           Per double, tutti i byte a 0 corrisponde al valore 0.0
           (valido nello standard IEEE 754).
        */
        memset(prnew, 0, NODES * sizeof(double));

        /* ── riga 181-183: stampa vettore corrente ─────────────────────────
           Mostra prold (il PR attuale) prima di calcolare il nuovo.
           %f formatta un double in notazione decimale fissa.
           \t è il carattere di tabulazione (allinea l'output in colonne).
        */
        for (j = 0; j < NODES; j++) {
            printf("%f\t", prold[j]);
        }
        printf("\n");

        /* ── riga 187-191: moltiplicazione sparsa M * prold ────────────────
           Questo è il cuore del PageRank e del vantaggio della CSC.

           Per ogni colonna col (= nodo sorgente):
             colptr[col]   → indice del primo non-zero della colonna
             colptr[col+1] → indice del primo non-zero della colonna SUCCESSIVA
             quindi il range [colptr[col], colptr[col+1]) contiene tutti
             i non-zero della colonna col.

           Per ogni non-zero k in quel range:
             rowind[k]  → riga destinazione del link
             val[k]     → peso del link (= 1/out_degree(col))
             prold[col] → PR corrente del nodo sorgente col

           prnew[rowind[k]] += val[k] * prold[col]
           significa: il nodo destinazione riceve una quota del PR del sorgente,
           proporzionale al peso del link (cioè 1/numero_di_link_uscenti).

           Complessità: O(EDGES) per iterazione invece di O(NODES²)
           della moltiplicazione densa. Con grafi sparsi grandi è cruciale.
        */
        norm = 0.0;
        for (col = 0; col < NODES; col++) {
            for (j = colptr[col]; j < colptr[col + 1]; j++) {
                prnew[rowind[j]] += val[j] * prold[col];
            }
        }

        /* ── riga 194-198: applicazione del damping ─────────────────────────
           Formula completa: prnew[i] = d * prnew[i] + (1-d)/N
           - damp1[i] = d = 0.85: la quota che proviene dai link reali
           - damp2[i] = (1-d)/N ≈ 0.025: la quota di "teleportation"
             (un utente apre una pagina a caso con probabilità 1-d)

           Questo garantisce che prnew sia sempre una distribuzione di
           probabilità valida (somma = 1) anche in presenza di dangling nodes
           (nodi senza link uscenti).
        */
        for (i = 0; i < NODES; i++) {
            prnew[i] = prnew[i] * damp1[i] + damp2[i];
            printf("%f\t", prnew[i]);
        }
        printf("\n");

        /* ── riga 202-210: calcolo norma L2 e copia vettore ────────────────
           Calcoliamo quanto è cambiato il vettore PR tra un'iterazione
           e la successiva: norma L2 = √( Σ (prnew[i] - prold[i])² )

           diff[i]    → differenza scalare per ogni componente
           norm_sq    → somma dei quadrati delle differenze (si accumula)
           norm       → radice quadrata di norm_sq = norma L2

           prold[i] = prnew[i]: aggiorna il vettore "vecchio" con quello
           nuovo, pronto per la prossima iterazione.
           Fatto nello stesso loop per risparmiare un passaggio sull'array.

           sqrt() viene da <math.h>; per questo il linker richiede -lm.
        */
        norm_sq = 0.0;
        for (i = 0; i < NODES; i++) {
            diff[i]  = prnew[i] - prold[i];
            norm_sq += diff[i] * diff[i];
            prold[i] = prnew[i];
        }
        norm = sqrt(norm_sq);

    /* ── riga 213: condizione di uscita dal do-while ──────────────────────
       Il loop continua finché norm > err (0.00001).
       Quando norm scende sotto la soglia, i valori PR sono "stabili":
       un'ulteriore iterazione cambierebbe ogni componente di meno di err.
    */
    } while (norm > err);

    /* ── riga 217-220: stampa risultati finali ──────────────────────────────
       Stampiamo il vettore finale prnew con etichetta nodo (1-based).
       %f con 6 cifre decimali di default.
    */
    printf("\n=== PAGERANK FINALE ===\n");
    for (j = 0; j < NODES; j++) {
        printf("Nodo %d: %f\n", j + 1, prnew[j]);
    }

    /* ── riga 223-227: liberazione memoria ─────────────────────────────────
       Ogni malloc/calloc deve avere il suo free() corrispondente.
       free(ptr) restituisce il blocco di memoria all'heap del sistema
       operativo. Non farlo causa memory leak: la memoria rimane occupata
       finché il processo non termina.
       In un programma che termina subito il SO la recupera comunque,
       ma è buona pratica e obbligatoria in programmi di lunga durata.
    */
    free(val);
    free(rowind);
    free(colptr);
    free(prold);
    free(prnew);
    free(damp1);
    free(damp2);
    free(diff);
    free(sum);



   double sum_pr=0;
   for(int i=0;i<NODES;i++) sum_pr+=prnew[i];
   printf("Somma PR = %.10f\n", sum_pr);
   return 0;

    /* ── riga 229: return 0 ─────────────────────────────────────────────────
       Restituisce 0 al sistema operativo: convenzione Unix per "successo".
       Qualsiasi valore != 0 segnala un errore (usato dagli script shell
       per decidere cosa fare dopo l'esecuzione del programma).
    */
    return 0;
}

