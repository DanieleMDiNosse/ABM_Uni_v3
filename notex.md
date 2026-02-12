### Limitations

- L'arbitragista è eccessivamente efficiente e ha un costo di esecuzione troppo basso. Oltre alla fee da pagare all'AMM
e il costo del flash loan, non ha alcuna spesa di bidding per block space. Minori sono le frizioni dell'arbitraggio,
maggiore sarà la frequenza di tali eventi e maggiore sarà l'adverse selection che i liquidity provider  affrontano.
- Gli arbitragisti sono relegati ad operare solamente all'inizio del blocco. Sebbene questa dinamica sia effettivamente 
presente, un arbitraggio può essere eseguito anche come backrun ad uno swap che ha spostato in maniera significativa il
prezzo del DEX dal prezzo del CEX
- Active LPs hanno un comportamento abbastanza semplice e *reagiscono* solamente al segnale sulla volatilità. In altre
parole, non hanno alcuna previsione futura.
- Oltre al termine di noise nella scelta dell'ampiezza del mint, gli LP sono molto omogenei (i.e. reagiscono allo stesso
segnale e l'unica cosa che evita una sincronizzazione collettiva sono i clock interni)
- Anche il Jiter presenta poche frizioni
- Il rebalancing degli LP è eseguito senza alcun tipo di impatto sul CEX
- Attualmente, non ho eseguito alcun stress test (ad esempio, cosa accade se ottengo un rendimento a 5 sigma dalla media?)



abm_results/scenarios/test/png/49804_toxicity_LPpassiveshare0.5_pjit1.0_8_normalized_lvr_steps10000.png





