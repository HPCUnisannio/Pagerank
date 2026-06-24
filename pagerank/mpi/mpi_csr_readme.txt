/* ========================================================================
 * RELAZIONE TECNICA: OTTIMIZZAZIONE HIERARCHICAL PAGERANK
 * PARADIGMA: MPI PURE - VERSIONE CSR V2 (COALESCED)
 * ========================================================================
 * Questo blocco documenta l'evoluzione del motore di calcolo distribuito
 * dal formato originale CSC (Compressed Sparse Column) al formato CSR
 * (Compressed Sparse Row) con ottimizzazione delle comunicazioni di rete.
 *
 * 1. ANALISI DEI LIMITI DELLA VERSIONE ORIGINALE OTTIMIZZATA (CSC)
 * ------------------------------------------------------------------------
 * La versione di partenza soffriva di un grave problema di scalabilità sui
 * cluster HPC dovuto al pattern di accesso alla memoria "Scatter" (scrittura sparsa):
 * * - Contesa e Ridondanza: Iterando sulle colonne (nodi sorgente), i contributi
 * per i nodi destinazione venivano scritti in modo sparso su tutto l'array.
 * In ambiente distribuito, questo costringeva ogni processo ad allocare
 * un intero vettore temporaneo "local_sum" lungo quanto tutto il grafo.
 * * - Saturazione della Rete: Per sommare i contributi parziali di tutti i 
 * processi, ad ogni iterazione veniva invocata una MPI_Iallreduce sull'INTERO
 * vettore. Spostare svariati Megabyte di dati per processo ad ogni singolo
 * ciclo costringeva la rete a calcolare somme vettoriali globali ad altissima
 * latenza, bloccando le CPU e saturando la banda.
 *
 * 2. LE TRE RIVOLUZIONI DELL'OTTIMIZZAZIONE CSR V2
 * ------------------------------------------------------------------------
 * A. Cambio Strutturale: Da CSC a CSR (Inversione di Prospettiva)
 * Siamo passati al formato CSR. Invece di chiederci "a chi invia i link
 * questo nodo?", ci chiediamo "da chi riceve i link questo nodo?".
 * Ogni processo è ora responsabile di un blocco lineare e contiguo di RIGHE
 * (nodi destinazione). Il calcolo diventa interamente locale: il processo
 * accumula i contributi direttamente nel proprio spazio di memoria, azzerando
 * i conflitti, i cache miss e permettendo di eliminare l'array "local_sum".
 *
 * B. Abbattimento del Payload: MPI_Allgatherv vs MPI_Iallreduce
 * Risolto il problema del calcolo distribuito tramite CSR, i processi non
 * devono più "sommare" vettori sovrapposti, ma solo "condividere" i propri
 * risultati definitivi. Invece di inviare l'intero vettore (NODES * 8 byte),
 * ogni processo esegue una MPI_Allgatherv inviando solo la propria porzione
 * esclusiva (circa NODES / NPROC elementi). Su 64 core, questo riduce il
 * traffico di rete in uscita del 98.4%. Lo scambio avviene in modalità pura
 * copia hardware (RDMA su InfiniBand), senza calcoli matematici di rete.
 *
 * C. Message Coalescing e Delayed Dangling Mass (Da 3 a 2 Comunicazioni)
 * Nelle reti HPC la latenza di attivazione di un messaggio è un collo di
 * bottiglia. Per eliminare la dipendenza dei dati tra Dangling Mass globale
 * e calcolo dell'errore (che costringeva a due riduzioni separate), abbiamo
 * sfasato l'applicazione della Dangling Mass di un'iterazione.
 * All'iterazione k si applica la massa calcolata all'iterazione k-1.
 * Trattandosi di un algoritmo convergente a punto fisso, l'impatto sul numero
 * di iterazioni totali è trascurabile, ma ci permette di calcolare nello
 * stesso ciclo sia l'errore locale (norm_sq_local) sia la nuova massa locale
 * (dm_local). I due scalari vengono impacchettati in un array di 2 elementi
 * ed eseguiti in un'UNICA chiamata MPI_Allreduce, dimezzando la latenza.
 *
 * 3. TABELLA COMPARATIVA DELLE PERFORMANCE DI RETE
 * ------------------------------------------------------------------------
 * METRICA                  | VERSIONE CSC OPT       | VERSIONE CSR OPT
 * ------------------------------------------------------------------------
 * Chiamate MPI nel loop    | 3 (1 Vettoriale)      | 2 (1 Scalare, 1 Gatherv)
 * Payload operazione princ.| Globale: NODES        | Locale: ~ NODES / NPROC
 * Tipo di operazione rete  | Computazionale        | Puro movimento dati (RDMA)
 * Impronta in Memoria      | Alta (local_sum)      | Minima (Assegnamento diretto)
 * ========================================================================
 */



   ---

   ## 1. Il Cuore dell'Algoritmo: L'Approccio "Pull"

   Nel calcolo del PageRank, esistono due modi per propagare il "rango" (il punteggio) da un nodo all'altro: l'approccio *Push* e l'approccio *Pull*. Il nostro codice utilizza rigorosamente l'**approccio Pull**, modellato matematicamente come una moltiplicazione Matrice Sparsa-Vettore (SpMV).

   * **Come funziona:** Invece di far sì che un nodo "sorgente" spinga (push) il suo valore verso i nodi a cui è collegato, facciamo l'opposto. Ogni thread/processo prende un nodo "destinazione" (una riga della matrice) e **"tira a sé" (pull)** le frazioni di PageRank di tutti i nodi che puntano verso di lui (le colonne non-zero di quella riga).
   * **Perché lo abbiamo scelto:**
   1. **Zero Conflitti (Lock-Free):** Dato che un processo MPI (o un thread OpenMP) è l'unico responsabile per la scrittura del risultato finale di quella specifica riga, non c'è mai il rischio che due processi cerchino di sovrascrivere lo stesso dato contemporaneamente. Non servono operazioni `atomic` o `lock`.
   2. **Sinergia con la CSR:** L'approccio Pull si sposa perfettamente con il formato *Compressed Sparse Row* (CSR), che è ottimizzato per scorrere velocemente la matrice riga per riga.



   ---

   ## 2. La Struttura Dati: Perché la Matrice CSR?

   L'uso della **CSR (Compressed Sparse Row)** è la prima grande ottimizzazione rispetto all'uso di una semplice lista di archi.
   Invece di memorizzare una matrice quadrata piena di zeri (che occuperebbe terabyte per grafi grandi), la CSR memorizza solo i valori esistenti usando 3 vettori:

   * `val`: I valori dei pesi (nel PageRank puro, spesso impliciti o calcolati dinamicamente, ma qui gestiti per generalità).
   * `colind`: L'indice della colonna (il nodo sorgente) per ogni elemento.
   * `rowptr`: Un array di "puntatori" che ci dice esattamente a quale indice di `val` e `colind` inizia e finisce una determinata riga.

   Questa struttura riduce drasticamente l'occupazione in RAM e aumenta in modo massiccio la **località spaziale della cache** della CPU: quando scorri gli elementi di una riga, sono tutti contigui in memoria.

   ---

   ## 3. Logica Passo-Passo del Codice

   ### FASE 1: Setup e Distribuzione (Il "Divide")

   1. **Allocazione Contigua (Ottimizzazione):** Invece di allocare `rowptr` e `out_degree` separatamente, allochiamo un unico grande blocco di memoria (`metadata_buffer`). Questo ci permette di inviare entrambi gli array a tutti i processi con **una singola chiamata** di rete (`MPI_Bcast`), dimezzando la latenza.
   2. **Lettura Centralizzata:** Solo il `MASTER` legge il file su disco e costruisce la matrice globale. Gli altri processi aspettano.
   3. **Partizionamento Intelligente:** Usiamo `compute_column_distribution` e `compute_nnz_distribution`. Non dividiamo semplicemente il grafo per "numero di nodi", ma calcoliamo quanti *archi* (`sendcnts`) spettano a ciascun processo. Questo garantisce che ogni processo faccia la stessa quantità di calcoli matematici, evitando sbilanciamenti (Load Balancing).
   4. **Distribuzione (Scatter):** Il master "taglia" i grandi array `val` e `colind` e invia a ciascun processo solo il pezzetto di matrice su cui dovrà lavorare tramite `MPI_Scatterv`. Fatto ciò, il master svuota la memoria globale per non sprecare RAM.

   ### FASE 2: La Computazione Iterativa (L' "Imperar")

   Questo è il ciclo `do-while`. Prima di entrarci, calcoliamo la *Dangling Mass* iniziale (i nodi senza link in uscita che disperdono il loro punteggio).
   All'interno del ciclo usiamo 3 funzioni chiave create per incapsulare la logica:

   1. **`csr_spmv_range`:** È l'approccio Pull. Moltiplica il pezzetto locale di matrice CSR per il vettore `prold`. Il risultato grezzo viene messo in `prnew`.
   2. **`pagerank_update_and_norm_range`:** Prende il risultato grezzo, vi applica la formula matematica completa del PageRank con il *Damping Factor* ($DAMPING = 0.85$), aggiunge la ridistribuzione della Dangling Mass globale, e calcola l'errore locale (la norma euclidea al quadrato locale).
   3. **`pagerank_compute_dangling_mass_range` (Look-ahead):** Questa è un'ottimizzazione algoritmica. Invece di aspettare il prossimo ciclo per capire quanta Dangling Mass c'è, la pre-calcoliamo *ora* guardando il vettore appena generato (`prnew`).

   **La Fusione delle Comunicazioni (Message Coalescing):**
   Alla fine dei calcoli, ogni processo ha un pezzetto di Errore (Norma) e un pezzetto di Dangling Mass. Invece di fare due `MPI_Allreduce` distinte (una per la norma, una per la massa), le infiliamo in un piccolo array di 2 elementi (`local_data[2]`). Con **una singola operazione di rete** sommiamo tutto a livello globale.

   Infine, `MPI_Allgatherv` si assicura che tutti i processi ricevano i pezzettini aggiornati di `prnew` calcolati dagli altri, ricostruendo il vettore completo per l'iterazione successiva.

   ### FASE 3: Validazione e Metriche

   Il master verifica che la somma totale del PageRank sia matematicamente corretta (deve essere $\approx 1.0$) per provare che non si è persa "massa" durante i calcoli. Successivamente, stampa il cruscotto con Speedup, Efficienza e il reale Bilanciamento del Carico basato sugli elementi non-zero (NNZ).

   ---

   ## 4. Riepilogo delle Scelte Architetturali (Da difendere in sede di esame)

   Se ti chiedono "Perché lo avete fatto così?", ecco i punti chiave:

   * **Perché niente `MPI_Type_struct` per il coalescing iniziale?** Perché creare un tipo di dato derivato per unire `double` (val) e `int` (colind) aggiunge overhead di impacchettamento sulla CPU che supera il vantaggio della rete, dato che l'invio della matrice avviene *una sola volta* in tutta l'esecuzione.
   * **Perché non parallelizzare le scritture con `atomic`?** Perché parallelizzare sui nodi destinazione (le righe in CSR) rende il codice intrinsecamente "Lock-Free". Evitiamo la contesa della memoria (Cache Ping-Pong) e sfruttiamo i registri della CPU per gli accumulatori locali.
   * **Perché due `MPI_Scatterv` invece di mandare l'intero grafo a tutti?** Per la scalabilità. Se il grafo è enorme (es. miliardi di archi), la memoria di un singolo nodo del cluster esploderebbe (Out Of Memory). Distribuendo la matrice, l'impronta di memoria scala perfettamente con l'aumentare dei nodi computazionali. Solo i vettori del PageRank rimangono replicati per intero.