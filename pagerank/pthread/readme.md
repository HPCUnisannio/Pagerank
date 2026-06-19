# PageRank con Pthread: Analisi Comparativa e Scelte Progettuali

## Panoramica delle Due Implementazioni

Questo documento presenta l'analisi comparativa di due implementazioni parallele dell'algoritmo PageRank Power Iteration utilizzando Pthreads:

1. Versione Base con Mutex: Architettura Master-Worker con sincronizzazione a mutex e variabili condizione
2. Versione Ottimizzata Privatizzata: Architettura Peer-to-Peer a zero mutex con memoria privatizzata

---

# PARTE 1: VERSIONE BASE CON MUTEX

## Architettura di Sincronizzazione

### Pattern a Due Barriere
L'implementazione utilizza due barriere distinte per prevenire deadlock e corse critiche:
- Barriera 1 (our_barrier): Sincronizza tutti i thread dopo la fase di calcolo parallelo, garantendo il completamento del prodotto matrice-vettore sparso e dell'accumulo della massa dangling prima della riduzione seriale.
- Barriera 2 (our_barrier2): Impedisce ai thread veloci di iniziare l'iterazione successiva prima che i thread lenti abbiano letto il criterio di convergenza.

### Broadcast contro Signal
L'uso di pthread_cond_broadcast invece di pthread_cond_signal garantisce che tutti i thread worker vengano risvegliati quando il master completa la sua fase seriale. Con pthread_cond_signal, solo un thread verrebbe risvegliato, causando il blocco indefinito dei worker rimanenti.

## Correttezza Matematica

### Applicazione Centralizzata del Damping
La formula di damping (prnew = prnew * DAMPING + (1-DAMPING)/N) viene applicata esclusivamente dal thread master dopo che tutti i prodotti parziali sono stati accumulati. Questo previene il bug originale in cui il damping veniva erroneamente applicato alle somme parziali, causando un'inflazione della massa di probabilità nei nodi che ricevevano contributi da più thread.

### Gestione dei Nodi Dangling
Ogni thread calcola il proprio contributo locale alla massa dangling. Il master aggrega questi contributi sotto protezione di mutex e ridistribuisce la massa di probabilità uniformemente tra tutti i nodi, mantenendo la proprietà stocastica (somma del PageRank = 1.0).

## Distribuzione del Lavoro

### Partizionamento 1D per Colonne
Le colonne sono distribuite in modo contiguo tra i thread in base all'ID del thread. Il thread master (tid=0) riceve le colonne di resto per bilanciare il carico. Questo approccio:
- Massimizza la località di cache per i pattern di accesso CSR
- Elimina la necessità di meccanismi complessi di work-stealing
- Fornisce un bilanciamento naturale del carico per grafi con distribuzione uniforme

## Verifica della Convergenza Thread-Safe

### Accesso Protetto alla Norma
Il criterio di convergenza (norm) è protetto da norm_mutex sia durante la scrittura (master) che durante la lettura (tutti i thread). Ogni thread cattura una copia locale (current_norm) per valutare la condizione del ciclo in modo coerente.

## Gestione della Memoria

### Vettori Locali per Thread
Ogni thread alloca il proprio vettore localpr, eliminando il falso sharing e riducendo la contesa sui mutex. L'accumulo nel vettore globale prnew avviene una sola volta per iterazione, usando add_mutex per aggiornamenti atomici.

### Pulizia dell'Allocazione Dinamica
I vettori locali vengono deallocati prima dell'uscita del thread per prevenire memory leak in computazioni di lunga durata.

## Primitive di Sincronizzazione

| Primitiva | Tipo | Scopo |
|-----------|------|-------|
| add_mutex | Mutex | Protegge l'accumulo in prnew |
| wait_mutex | Mutex | Coordina l'attesa dei worker sul master |
| norm_mutex | Mutex | Protegge la lettura/scrittura della norma |
| dangling_mutex | Mutex | Protegge l'accumulo della massa dangling |
| our_barrier | Barriera | Sincronizza fine calcolo parallelo |
| our_barrier2 | Barriera | Sincronizza inizio nuova iterazione |
| proceed_cv | Variabile Condizione | Segnala completamento fase master |

## Flusso di Esecuzione per Iterazione

1. Reset: Il master azzera global_dangling_mass
2. Barriera 2: Tutti i thread si allineano
3. Calcolo Locale: Ogni thread calcola massa dangling e prodotto matrice-vettore
4. Riduzione: Accumulo thread-safe nei vettori globali
5. Barriera 1: Attesa completamento di tutti i thread
6. Fase Master: Applicazione damping, calcolo convergenza, broadcast
7. Reset Locale: Azzeramento vettori locali
8. Barriera 2: Prevenzione corse critiche
9. Verifica Convergenza: Lettura thread-safe della norma

## Comando di Compilazione
gcc pthread/pr_pthread.c libraries/data.c libraries/measure.c -o pthread/pr_pthread -Ilibraries -lpthread -lm

---

# PARTE 2: VERSIONE OTTIMIZZATA PRIVATIZZATA

## Filosofia Progettuale: Architettura Zero Mutex

L'implementazione adotta un approccio di privatizzazione della memoria che elimina completamente l'uso di mutex durante le fasi computazionali, sostituendoli con due sole barriere di sincronizzazione. Questa scelta architetturale massimizza il parallelismo effettivo e minimizza l'overhead di sistema.

## Struttura della Memoria Privatizzata

### Array all_prnew
La memoria privatizzata è implementata come un unico blocco contiguo di dimensione CORES * NODES elementi double. Ogni thread possiede una porzione esclusiva identificata dall'offset tid * NODES:

- Thread 0: all_prnew[0] fino a all_prnew[NODES-1]
- Thread 1: all_prnew[NODES] fino a all_prnew[2*NODES-1]
- ...
- Thread 5: all_prnew[5*NODES] fino a all_prnew[6*NODES-1]

Questa disposizione in memoria contigua massimizza l'efficienza della cache durante la fase di riduzione con accesso strided.

## Suddivisione del Lavoro

### Partizionamento a Blocchi Statici
La distribuzione del carico utilizza una strategia a blocchi statici applicata in due fasi distinte:

| Fase | Partizionamento | Ruolo del Thread |
|------|----------------|------------------|
| SpMV (Fase B) | Colonne | Analizza i nodi sorgente assegnati e scrive contributi nel proprio array privato |
| Riduzione (Fase C) | Righe | Somma i contributi da tutti gli array privati per i nodi destinazione assegnati |

Gestione del resto della divisione intera:
int col_end = (tid == CORES - 1) ? NODES : (tid + 1) * col_chunks;
L'ultimo thread assorbe i nodi rimanenti, garantendo copertura completa senza sbilanciamenti significativi.

## Flusso di Sincronizzazione

### Barriera 1 (our_barrier)
Sincronizza il passaggio dalla fase di calcolo parallelo (SpMV privata) alla fase di riduzione. Garantisce che tutti i thread abbiano completato la scrittura nei propri array privati prima che qualsiasi thread inizi a leggere i dati degli altri.

### Barriera 2 (our_barrier2)
Sincronizza il completamento del calcolo delle norme parziali. Assicura che tutti i thread abbiano scritto il proprio contributo in part_norm prima che venga calcolata la norma globale.

## Calcolo della Convergenza Lock-Free

### Norma Parziale Locale
Ogni thread calcola l'errore quadratico solo per la propria porzione di nodi destinazione, accumulandolo in una variabile locale local_norm_sq. Questo elimina la necessità di sincronizzazione durante il calcolo.

### Array part_norm
Le norme parziali vengono scritte in un array condiviso di dimensione CORES, dove ogni thread scrive esclusivamente nella cella corrispondente al proprio ID. Questo pattern di accesso disgiunto non richiede mutex.

### Riduzione Globale
Dopo la Barriera 2, ogni thread esegue autonomamente la somma delle norme parziali e il calcolo della radice quadrata. La ridondanza del calcolo su tutti i core è insignificante rispetto al costo di una sincronizzazione aggiuntiva.

## Architettura Peer-to-Peer

A differenza del modello Master-Worker tradizionale, questa implementazione adotta un'architettura paritetica:

- Nessun thread master privilegiato: il Thread 0 è un lavoratore come gli altri
- Calcolo distribuito della convergenza: ogni thread valuta autonomamente la condizione di uscita
- Riduzione parallela: l'aggregazione dei contributi è suddivisa equamente tra tutti i thread

## Dettagli Implementativi Rilevanti

### Scrittura Privata nella SpMV
Durante la moltiplicazione matrice-vettore, ogni thread scrive esclusivamente nel proprio array privato:
my_private_prnew[rowind[j]] += val[j] * prold[col];
Nessun lock è necessario perché nessun altro thread accede a quella porzione di memoria.

### Lettura Strided nella Riduzione
Durante la riduzione, ogni thread legge in modo "verticale" tutti gli array privati:
for(t = 0; t < CORES; t++) {
    raw_pagerank += all_prnew[t * NODES + i];
}
Il pattern di accesso strided sfrutta la località spaziale grazie all'allocazione contigua di all_prnew.

### Applicazione Centralizzata del Damping
Il damping viene applicato durante la fase di riduzione, dopo aver sommato tutti i contributi grezzi, garantendo la correttezza matematica che le versioni precedenti violavano.

## Comando di Compilazione
gcc pthread/pr_pthread_opt.c libraries/data.c libraries/measure.c -o pthread/pr_pthread_opt -Ilibraries -lpthread -lm

---

# PARTE 3: CONFRONTO TRA LE DUE VERSIONI

## Tabella Comparativa Globale

| Parametro | Versione con Mutex | Versione Privatizzata |
|:---|:---|:---|
| Modello Architetturale | Master-Worker | Peer-to-Peer |
| Sincronizzazione | Mutex + Variabili Condizione + Barriere | Solo Barriere |
| Contesa sui Lock | Alta durante accumulo | Zero |
| Overhead di Sistema | Elevato (context switch, lock contention) | Minimo (solo barriere) |
| Occupazione Memoria Extra | Minima (solo vettori locali) | ~33 MB (blocco privatizzato) |
| Località di Cache | Buona (accesso locale) | Ottima (contigua, strided) |
| Scalabilità | Media (collo di bottiglia master) | Elevata (nessun collo di bottiglia) |
| Complessità Codice | Medio-Bassa | Alta (gestione accessi strided) |
| Calcolo Convergenza | Centralizzato (Master) | Distribuito (tutti i thread) |

## Confronto Primitive di Sincronizzazione

| Aspetto | Versione Mutex | Versione Privatizzata |
|---------|---------------|----------------------|
| Numero Mutex | 4 (add_mutex, wait_mutex, norm_mutex, dangling_mutex) | 0 |
| Numero Barriere | 2 (our_barrier, our_barrier2) | 2 (our_barrier, our_barrier2) |
| Variabili Condizione | 1 (proceed_cv) | 0 |
| Variabili di Flag | 1 (var_wait) | 0 |

## Flusso di Esecuzione a Confronto

| Fase | Versione Mutex | Versione Privatizzata |
|------|---------------|----------------------|
| Reset Iniziale | Master azzera global_dangling_mass | Ogni thread azzera il proprio array privato |
| Calcolo Dangling | Ogni thread accumula in global_dangling_mass con mutex | Non implementato (semplificazione) |
| Moltiplicazione SpMV | Scrittura in localpr privato, poi accumulo in prnew globale con add_mutex | Scrittura diretta in array privato, nessun lock |
| Riduzione | Somma sequenziale dei localpr con mutex | Lettura strided parallela, nessun lock |
| Damping | Applicato dal Master su prnew globale | Applicato da ogni thread sulla propria porzione |
| Convergenza | Calcolata dal Master, letta con norm_mutex | Calcolata in parallelo da tutti i thread |
| Risveglio | pthread_cond_broadcast dal Master | Non necessario (tutti i thread attivi) |

## Vantaggi e Svantaggi

### Versione con Mutex
Vantaggi:
- Codice più semplice e manutenibile
- Minore occupazione di memoria
- Gestione esplicita dei nodi dangling
- Più facile da debuggare

Svantaggi:
- Collo di bottiglia sul master
- Overhead di lock contention
- Thread worker inattivi durante la fase master
- Scalabilità limitata

### Versione Privatizzata
Vantaggi:
- Massimo parallelismo (zero contesa)
- Nessun collo di bottiglia sequenziale
- Migliore scalabilità su molti core
- Eliminazione overhead dei mutex

Svantaggi:
- Maggiore complessità del codice
- Occupazione memoria proporzionale al numero di core
- Pattern di accesso strided può causare cache miss su grafi enormi
- Debugging più complesso

## Note sulle Prestazioni

### Versione Mutex
- L'overhead di sincronizzazione è presente ma gestibile per grafi medio-piccoli
- Il partizionamento contiguo delle colonne massimizza l'efficienza della cache
- I mutex sono usati solo per brevi sezioni critiche durante l'accumulo
- La variabile condizione con broadcast garantisce risveglio simultaneo di tutti i worker

### Versione Privatizzata
- L'overhead di sincronizzazione è minimizzato usando barriere solo a inizio e fine iterazione
- L'allocazione contigua di all_prnew ottimizza gli accessi strided
- Il calcolo ridondante della norma globale su tutti i core è trascurabile
- Ideale per sistemi con molti core e grafi di grandi dimensioni

## Criteri di Scelta

| Scenario | Versione Consigliata |
|----------|---------------------|
| Grafi piccoli (< 10.000 nodi) | Versione Mutex (overhead accettabile) |
| Grafi medi (10.000 - 100.000 nodi) | Entrambe valide |
| Grafi grandi (> 100.000 nodi) | Versione Privatizzata (scalabilità superiore) |
| Pochi core (2-4) | Versione Mutex (minore complessità) |
| Molti core (8+) | Versione Privatizzata (sfrutta meglio il parallelismo) |
| RAM limitata | Versione Mutex (minore occupazione) |
| Massime prestazioni | Versione Privatizzata (zero contesa) |
| Manutenibilità prioritaria | Versione Mutex (codice più semplice) |

## Correttezza Matematica (Comune a Entrambe)

Entrambe le implementazioni correggono i bug critici della versione originale:

1. Deadlock Risolto: Uso di pthread_cond_broadcast invece di pthread_cond_signal
2. Race Condition Risolte: Introduzione della Barriera 2 per prevenire corse critiche
3. Damping Corretto: Applicazione centralizzata della formula dopo l'accumulo completo
4. Dangling Nodes: Gestione corretta della massa dei nodi pozzo (Versione Mutex)

## Comandi di Compilazione

Versione Base con Mutex:
gcc pthread/pr_pthread.c libraries/data.c libraries/measure.c -o pthread/pr_pthread -Ilibraries -lpthread -lm

Versione Ottimizzata Privatizzata:
gcc pthread/pr_pthread_opt.c libraries/data.c libraries/measure.c -o pthread/pr_pthread_opt -Ilibraries -lpthread -lm