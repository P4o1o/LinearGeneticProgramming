#### Problemi psb2 su float

* ##### bouncing balls:
    Given a starting height and a height after the first bounce of a dropped ball, calculate the bounciness index (height of first bounce / starting height). Then, given a number of bounces, use the bounciness index to calculate the total distance that the ball travels across those
    bounces.
* ##### dice game:
    Peter has an n sided die and Colin has an m sided die. If they both roll their dice at the same time, return the probability that Peter rolls strictly higher than Colin.
* ##### shopping list:
    Given a vector of floats representing the prices of various shopping goods and another vector of floats representing the percent discount of each of those goods, return the total price of the shopping trip after applying the discount to each item. 
* ##### snow day:
    Given an integer representing a number of hours and 3 floats representing how much snow is on the ground, the rate of snow fall, and the proportion of snow melting per hour, return the amount of snow on the ground after the amount of hours given. Each hour is considered a discrete event of adding snow and then melting, not a continuous process.
* ##### vector distance:
    Given two n-dimensional vectors of floats, return the Euclidean distance between the two vectors in n-dimensional space.
#

#### Risultati

il programma trova la soluzione facilmente per __shopping list__ e __vector distance__ quando le dimensioni dell'input sono piccole
Su __bouncing balls__ e __snow day__ invece sembra non riesce a trovare la soluzione, infatti servirebbe un ciclo for per calcolare la soluzione di questi due problemi. Anche per __dice game__ fa fatica, infatti la soluzione a questo problema necessiterebbe di un if-else.
#

#### Valgrind

ho testato con valgrind e non ci sono memory leaks, se non alcuni che però mi sembrano essere causati da OpenMP.
mentre cachegrind sembrerebbe rilevare un buon utilizzo della cache e, mentre i branch condizionali vengono in gran parte previsti correttamente, i branch indiretti quasi la metà delle volte vengono predetti in modo errato. Questi branch indiretti mal predetti sembrano tutti essere causati al momento di esecuzione di un individuo, quando tramite l'indice dell'operazione da utilizzare, salvato nell'individuo, si accede all'array di operazioni e si esegue l'operazione al dato indice.
#

#### Selezioni

le selezioni a __torneo__ ed ad __elitismo__ funzionano bene.
La __roulette__ che utilizza resampling e si serve del "fitness" definito come come: __(mse dell'individuo peggiore della popolazione) - (mse dell'individuo di cui si vuole calcolare il fitness)__ sembrerebbe funzionare, ma riporta risultati pessimi e questo mi fa dubitare della sua corretezza.

le selezioni con __fitness sharing__ per popolazioni grandi sono estreamente lente e spesso il malloc per la tabella di distanze fallisce a causa delle dimensioni eccessive che può raggiungere (Nell'attuale implementazione si costruisce una vera e propria tabella, quindi la distanza tra due individui diversi viene salvata due volte. Questo si potrebbe migliorare salvando le distanze una volta sola, ma implicherebbe qualche calcolo in più che potrebbe rallentarlo ultreriormente).
Inoltre per calcolarlo l'ho definito come: __((mse) ^ beta) * sharing__ . Nel processo di selezione, quindi, vado in cerca del fitness sharing più basso.
Ho il dubbio che questo calcolo del fitness sharing possa non risultare corretto.
Per il fitness sharing ho definito una distanza di edit (definita in selections.c) tra gli individui, ma non sono certo di aver definito i pesi correttamente (1 se l'operazione è diversa, se invece è la stessa +0.25 se l'indice risultato è diverso e +0.75/arity per ogni argomento diverso).
#

#### Domande sui tipi di dato e sulla funzione di hash adottata
Ho definito le lunghezze come __lenght_t = uint32_t__ ovvero come interi senza segno a 32 byte, può andare bene?

Inoltre i vari numeri che identificano un'istruzione (uno per il tipo di operazione, uno per il registro del risultato e uno o due per gli argomenti) gli ho definiti come __op_type = uint16_t__, __env_index = uint16_t__, ovvero interi senza segno a 16 bit, dato che così facendo risulta semplice scrivere la funzione di hash che va a considerare un'istuzione singola (operazione, risultato, argomenti) come un'unica "parola" da 64 bit; questo però implica che la funzione di hash supporta operazioni con al massimo 2 argomenti.
Va bene come approccio? 2 argomenti supportati possono essere sufficenti?

Per l'implementazione della funzione hash mi sono basato sull'algoritmo siphash.
Nella mia impementazione rappresento ogni istruzione come 64 byte, non considero la dimensione dell'individuo nel calcolo dell'hash (L'algoritmo tradizionale aggiunge __strlen % 256__ come ultimo byte della sequenza). La mia implementazione può risultare comunque corretta o sarebbe meglio considerarla nel calcolo?

#### Conteggio degli individui valutati
Nell'attuale implementazione, solamente al momento della creazione della popolazione iniziale, un individuo viene conteggiato tra gli individui valutati se e solo se per nessuna delle singole istanze di un problema l'individuo genera il valore DBL_MAX (oppure NaN, +- inf) in output (in caso contrario l'individuo viene scartato), si veda funzione get_mse() in genetics.c. Può essere corretto o è meglio considerare anche loro nel conteggio?

#### Struttura dei file

main.c
    ├── float_psb2.h/float_psb2.c
    ├── evolution.h/evolution.c

evolution.h/evolution.c 
    ├── selections.h/selections.c
    ├── creations.h/creations.c

selections.h/selections.c
    ├── genetics.h/genetics.c

creations.h/creations.c
    ├── genetics.h/genetics.c

float_psb2.h/float_psb2.c
    ├── genetics.h/genetics.c

genetics.h/genetics.c
    ├── prob.h/prob.c
    ├── logger.h/logger.c
    ├── operations.h/operations.c


* ##### logger.h/logger.c :
    contiene le funzioni per loggare gli errori.
* ##### prob.h/prob.c :
    contiene le funzioni per generare numeri random e per gestire le probabilità.
* ##### operations.h/operations.c :
    contiene le varie operazioni matematiche che possono essere utilizzate per comporre un individuo.
* ##### genetics.h/genetics.c :
    contiene le definizioni di:
    - __genetic_environment__: operazioni e numero di "registri" utilizzabili, 
    - __individual__: struct rappresentante un individuo,
    - __populaion__: struct rappresentante un popolazione,
    - __genetic_input__:i dati e i rispettivi risultati attesi per un problema da risolvere con LGP
    - - __genetic_result__:risultati ottenuti dopo aver eseguito LGP
* ##### selections.h/selections.c :
    contiene le funzioni rappresentanti i metodi di selezione.
* ##### cretions.h/cretions.c :
    contiene le funzioni rappresentanti i metodi di creazione di una nuova popolazione.
* ##### evolution.h/evolution.c :
    contiene la funzione __evolve__ che esegue LGP e __genetic_options__, i vari parametri per la funzione.
* ##### float_psb2.h/float_psb2.c :
    contiene le funzioni per creare i genetic_input per i problemi psb2 su float.
* ##### benchmarks.h/benchmarks.c :
    contiene le funzioni per mostrare i risultati di un'evoluzione o per stamparli su file e per ogni problema psb2 è stata scritta una funzione che tenta di risolverlo.
* ##### main.c :
    qualche esempio di utilizzo.
#