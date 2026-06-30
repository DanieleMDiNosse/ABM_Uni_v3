# Citation insertion plan for `paper/ABM_paper.tex`

This file does not modify the paper. It marks where the new or upgraded citations from `biblio_additions.md` are most useful in the current manuscript, with local line references from `paper/ABM_paper.tex` as inspected in this working tree.

## Evidence used

- Local paper inspected: `paper/ABM_paper.tex`.
- Local bibliography inspected: `paper/bibliography.bib`.
- User report inspected: `biblio_additions.md`.
- OpenAlex checks used to verify several high-priority references and publication metadata.

Important metadata caveat: OpenAlex confirms several venue-validated entries, e.g. Heimbach et al. AFT 2022 DOI `10.1145/3558535.3559772`, Fan et al. AFT 2023 DOI `10.4230/LIPIcs.AFT.2023.25`, Fritsch and Canidio Web Conference Companion 2024 DOI `10.1145/3589335.3651961`, Ho and Stoll Journal of Finance DOI `10.1111/j.1540-6261.1983.tb02282.x`, and Milionis et al. arXiv DOI `10.48550/arxiv.2208.06046`. For the Cartea--Drissi--Monga AMM papers, OpenAlex currently returns SSRN records (`10.2139/ssrn.4144743` and `10.2139/ssrn.4273989`); the report's “forthcoming in JEDC/SIAM” statement should be verified against the authors' pages or publisher pages before writing it as a bibliographic fact in the paper.

## Recommended citation keys

Existing keys already in `paper/bibliography.bib`:

- `adams2021`: Uniswap v3 whitepaper.
- `Adams2020UniswapVC`: Uniswap v2 core.
- `angeris2020`: Uniswap market analysis.
- `heimbach2022`: Risks and returns of Uniswap v3 LPs; should be kept, and the current entry already uses the AFT DOI.
- `milionis2023`: LVR paper; should be kept despite being arXiv because it is field-defining.
- `fritsch2024`: Measuring arbitrage losses and AMM liquidity profitability; the entry should be upgraded from “arXiv preprint” to the ACM Web Conference Companion 2024 proceedings metadata.
- `cartea2023`: Predictable loss and optimal liquidity provision; consider updating metadata after verifying final/forthcoming publication status.
- `fan2023`: Strategic liquidity provision in Uniswap v3; should include DOI `10.4230/LIPIcs.AFT.2023.25`.
- `maire2024`: Market neutral liquidity provision; already a published Ledger entry.
- `cohen2023`, `hafner2024`, `zhang2023`, `canidio2025`: useful but more preprint-heavy; do not let these dominate the revised review.

Suggested new keys to add to the `.bib` later, if you decide to implement the plan:

- `cartea2022execution`: Cartea, Drissi, and Monga, “Decentralised Finance and Automated Market Making: Execution and Speculation”.
- `westerhoff2006`: Westerhoff and Dieci, transaction taxes in heterogeneous-agent markets, JEDC.
- `mannaro2008`: Mannaro, Marchesi, and Setzu, artificial financial market and Tobin-like transaction taxes, JEBO.
- `pellizzari2009`: Pellizzari and Westerhoff, transaction taxes under different microstructures, JEBO.
- `brock2009`: Brock, Hommes, and Wagener, destabilizing effects of more hedging instruments, JEDC.
- `leal2016`: Leal, Napoletano, Roventini, and Fagiolo, low- and high-frequency trading ABM.
- `gao2024`: Gao et al., high-frequency financial market simulation and flash-crash scenarios, JASSS.
- `yagi2020`: Yagi, Masuda, and Mizuta, HFT and artificial market liquidity, IEEE TCSS.
- `carro2015`: Carro, Toral, and San Miguel, markets, herding, and external information, PLoS ONE.
- `mizuta2020`: Mizuta, ABM for designing financial markets that work well, IEEE SSCI.
- `raberto2001`: Raberto, Cincotti, Focardi, and Marchesi, agent-based simulation of a financial market, Physica A.
- `chakraborti2011`: Chakraborti et al., econophysics review of agent-based models, Quantitative Finance.
- `ho1983`: Ho and Stoll, dealer markets under competition, Journal of Finance.
- `avellaneda2008`: Avellaneda and Stoikov, high-frequency trading in a limit order book, Quantitative Finance.
- `cartea2015book`: Cartea, Jaimungal, and Penalva, Algorithmic and High-Frequency Trading.
- `gueant2016book`: Guéant, The Financial Mathematics of Market Liquidity.

## Global bibliography strategy

The main weakness of the current draft is not only the number of references, but the composition: the bibliography is heavily DeFi/protocol/working-paper oriented. The revised citation structure should make the paper read as a market-design and market-microstructure contribution that happens to study an AMM. The most useful new layer is therefore not more Uniswap papers, but non-DeFi ABM and microstructure references that support three claims:

1. ABMs are a standard tool for studying financial-market rule changes and heterogeneous-agent feedback.
2. Fee changes in AMMs are analogous to transaction taxes, spreads, and market-design frictions in traditional market models.
3. Latency, priority, high-frequency behavior, and information release are established ABM/microstructure themes, not ad hoc crypto-specific complications.

## High-priority insertion points

### 1. Abstract: strengthen the “ABM as market-design tool” framing

Location: lines 43--55, abstract.

Current relevant text:

> This paper develops a granular Agent-Based Model (ABM) of a Uniswap v3 pool interacting with a stochastic reference market governed by Heston volatility dynamics.

Recommended insertion:

- Usually the abstract should not carry many citations, but if the target journal allows citations in abstracts, add one compact methodological anchor after “Agent-Based Model (ABM)”.
- Candidate citation: `\cite{raberto2001,mizuta2020}` or no citation in abstract and instead cite these in the Introduction.

Why here:

- This signals from the first page that the contribution is not only a protocol simulation; it belongs to financial-market ABM and market-design methodology.

Priority: optional. Prefer adding these citations in the Introduction if the journal discourages abstract citations.

### 2. Introduction, first AMM paragraph: separate protocol mechanics from market-making literature

Location: lines 63--72.

Current relevant text:

> Automated market makers (AMMs) have emerged as a key innovation in Decentralized Finance (DeFi), allowing users to trade assets through liquidity pools instead of traditional order books. ... Uniswap v3 introduced concentrated liquidity ... `\cite{adams2021}`.

Recommended insertion:

- Keep `\cite{adams2021}` for Uniswap v3 implementation mechanics.
- Add `\cite{angeris2020,cartea2022execution}` near the sentence that describes AMMs as trading venues rather than order books.
- If the paper is positioned for a finance audience, add one bridging sentence after the first sentence:

Suggested prose:

> From a market-microstructure perspective, AMMs can be interpreted as automated dealer markets in which the fee plays a role analogous to a spread or transaction cost, while inventory adjustment is embedded in the pricing rule rather than in discretionary dealer quotes `\cite{ho1983,avellaneda2008,angeris2020,cartea2022execution}`.

Why here:

- This directly addresses the supervisor's concern by making the introduction less “crypto whitepaper first” and more finance/microstructure first.
- It also prepares readers for dynamic fees as a market-design variable.

Priority: very high.

### 3. Introduction, LVR paragraph: keep Milionis but add adverse-selection/market-making anchors

Location: lines 73--85.

Current relevant text:

> Recent research `\cite{milionis2023}` has formalized the notion of LP profitability ... LVR is the loss an LP suffers due to adverse selection ...

Recommended insertion:

- Keep `\cite{milionis2023}` as the central LVR citation.
- Add traditional market-making citations after the explanation that LPs are “picked off” by informed arbitrageurs.
- Candidate citations: `\cite{ho1983,avellaneda2008,cartea2015book,gueant2016book}`.

Suggested prose:

> This adverse-selection interpretation parallels the classical market-making literature, where liquidity suppliers require compensation for inventory risk and informed order flow through spreads and execution premia `\cite{ho1983,avellaneda2008,cartea2015book,gueant2016book}`.

Why here:

- The current paragraph makes LVR sound like a purely DeFi concept. This addition shows that the economic mechanism is recognizable to finance referees.

Priority: very high.

### 4. Introduction, empirical LP profitability paragraph: upgrade and consolidate venue-validated DeFi references

Location: lines 86--101.

Current relevant text:

> For example, an in-depth analysis of Uniswap v3 by Heimbach et al. `\cite{heimbach2022}` ... Likewise ... Fritsch and Canidio `\cite{fritsch2024}` ...

Recommended insertion:

- Keep `\cite{heimbach2022}`; OpenAlex confirms AFT 2022 DOI.
- Keep `\cite{fritsch2024}` but update the `.bib` entry later to the ACM Web Conference Companion 2024 DOI.
- Add `\cite{maire2024}` only if discussing hedged/market-neutral LP strategies, not just empirical profitability.

Suggested prose addition after the Fritsch sentence:

> The empirical evidence is therefore consistent with the theoretical view that AMM fees must compensate not only for price volatility, but also for the timing and informational content of order flow `\cite{milionis2023,fritsch2024,maire2024}`.

Why here:

- The paragraph already does the right job; the main improvement is metadata quality and a cleaner link from empirical underperformance to the paper's fee-control question.

Priority: high.

### 5. Introduction, ABM motivation paragraph: add the non-DeFi ABM layer

Location: lines 102--118.

Current relevant text:

> Agent-Based Models (ABMs) and simulations offer a way to study complex interactions in AMM ecosystems ... ABM can be interpreted in two distinct ways ...

Recommended insertion:

Add citations after the general ABM motivation and after the “controlled laboratories” sentence.

Candidate citations:

- General artificial financial markets: `\cite{raberto2001,chakraborti2011}`.
- Market design / policy ABMs: `\cite{westerhoff2006,mannaro2008,pellizzari2009,mizuta2020}`.
- HFT/latency/cancellation ABMs: `\cite{leal2016,gao2024,yagi2020}`.
- External information/herding: `\cite{carro2015}`.

Suggested replacement/addition:

> This use of ABMs as controlled laboratories is well established in artificial financial-market research, where heterogeneous-agent simulations have been used to study transaction taxes, market microstructure, high-frequency trading, cancellation rules, fragmentation, and the propagation of external information `\cite{raberto2001,chakraborti2011,westerhoff2006,mannaro2008,pellizzari2009,leal2016,carro2015,mizuta2020}`.

Why here:

- This is the single most important place to answer the supervisor's criticism. It makes the paper's methodology legible outside DeFi and reduces the relative weight of arXiv/protocol citations.

Priority: very high.

### 6. Introduction, statement of objective: connect dynamic AMM fees to transaction-tax / fee-policy ABMs

Location: lines 118--122.

Current relevant text:

> Our goal is to investigate whether, and in what way, dynamically adjusted fee schedules influence the economics of liquidity provision ...

Recommended insertion:

Add a market-design sentence immediately after this objective.

Suggested prose:

> In this sense, AMM fee adaptation is the DeFi analogue of a market-design intervention: changing a transaction cost or spread-like rule and observing how liquidity, routing, arbitrage, and welfare-relevant outcomes respond in a heterogeneous-agent system `\cite{westerhoff2006,mannaro2008,pellizzari2009,mizuta2020}`.

Why here:

- It gives a strong conceptual bridge to JEDC-style computational economics and financial market-design literature.

Priority: very high.

### 7. Introduction, contributions/findings paragraph: cite the closest analytical AMM papers

Location: lines 135--144.

Current relevant text:

> The simulations deliver three main findings. First, under block-time execution, static fees are insufficient ... Second, dynamic fees can improve ... Third, fee adaptivity creates ...

Recommended insertion:

- Add citations in the lead-in sentence or immediately before the findings paragraph, not in every finding.
- Candidate citations: `\cite{milionis2023,cartea2023,cartea2022execution,fritsch2024}`.

Suggested prose:

> These findings extend the analytical and empirical AMM literature on predictable loss, LVR, and arbitrage losses by studying how adaptive fee rules interact with block-time execution and strategic liquidity provision `\cite{milionis2023,cartea2023,cartea2022execution,fritsch2024}`.

Why here:

- This explicitly positions the contribution relative to published or publication-bound AMM work rather than only listing model components.

Priority: high.

## Literature review restructuring points

### 8. Literature review opening: announce four blocks, not three

Location: lines 147--148.

Current relevant text:

> We first discuss analytical contributions ... empirical evidence ... simulation-based and agent-based approaches ... dynamic fee mechanisms ...

Recommended insertion:

Revise the roadmap to include:

1. AMM microstructure and LP profitability.
2. Empirical AMM LP performance.
3. ABM market-design and policy analysis in traditional financial markets.
4. Dynamic fee mechanisms in AMMs.

Candidate citations in the opening sentence: none necessary, or `\cite{raberto2001,chakraborti2011,mizuta2020}` after the ABM block description.

Why here:

- The current structure has an ABM subsection, but it is almost entirely AMM-specific. A separate market-design ABM block is the clearest response to “bibliography too scarce and too technical”.

Priority: very high.

### 9. Analytical and theoretical approaches: add execution/speculation and traditional market-making references

Location: lines 149--153.

Current relevant text:

> The foundational models of AMMs and liquidity provider returns have benefited from insights in financial economics ... Milionis ... Cartea ... Aqsha ... Bayraktar ...

Recommended insertion:

- Add `\cite{cartea2022execution}` near the Cartea paragraph, because it is the strongest “AMMs in finance/computational economics” bridge from the report.
- Add `\cite{ho1983,avellaneda2008,cartea2015book,gueant2016book}` in the sentence linking AMM LPs to market makers.

Suggested prose:

> This framing connects AMM liquidity provision to the broader market-making literature, where spreads and execution policies compensate dealers for inventory risk, asymmetric information, and adverse selection `\cite{ho1983,avellaneda2008,cartea2015book,gueant2016book}`.

Why here:

- It converts “theoretical AMM approaches” into “AMM theory as a branch of market microstructure”.

Priority: very high for JBF/JEDC positioning.

### 10. Analytical and theoretical approaches: be careful with Cartea publication status

Location: line 150 paragraph, sentence beginning “For example, Cartea et al. `\cite{cartea2023}` ...”

Recommended insertion:

- Cite both Cartea papers if you add `cartea2022execution`.
- Do not write “forthcoming in JEDC/SIAM” unless verified outside OpenAlex.

Suggested wording:

> Cartea, Drissi, and Monga study AMM execution, speculation, predictable loss, and optimal liquidity provision, providing a direct bridge between DeFi mechanisms and continuous-time market-making models `\cite{cartea2022execution,cartea2023}`.

Why here:

- Stronger than current wording because it uses both papers and does not overclaim venue status without verification.

Priority: high.

### 11. Empirical studies: add venue language for Heimbach and Fritsch

Location: lines 154--158.

Current relevant text:

> The aforementioned study by Heimbach et al. ... Subsequent empirical studies ... Fritsch and Canidio ...

Recommended insertion:

- Mention that Heimbach et al. appears in AFT 2022 and Fritsch--Canidio in ACM Web Conference Companion 2024, if desired.
- Candidate citations remain existing keys, but update `fritsch2024` BibTeX later.

Suggested prose:

> These venue-validated empirical studies are important because they move LP profitability from protocol commentary into reproducible measurement using on-chain data `\cite{heimbach2022,fritsch2024}`.

Why here:

- It directly improves the perceived quality of the bibliography without adding many citations.

Priority: medium-high.

### 12. Add new subsection: “ABMs for market design and financial-policy analysis”

Location: immediately after the current “Agent-Based Modeling and Simulations” subsection, before `\subsection{Dynamic fees in deployed AMMs}` at line 174. Alternatively, split the current ABM subsection into AMM-specific and non-DeFi ABM parts.

Recommended citations:

- Foundational artificial markets: `\cite{raberto2001,chakraborti2011}`.
- Transaction taxes and fee-policy analogues: `\cite{westerhoff2006,mannaro2008,pellizzari2009}`.
- Market-design destabilization and hedging instruments: `\cite{brock2009}`.
- HFT/latency/cancellation: `\cite{leal2016,gao2024,yagi2020}`.
- External information: `\cite{carro2015}`.
- Market-design ABM methodology: `\cite{mizuta2020}`.

Suggested subsection skeleton:

> A related strand of artificial financial-market research uses ABMs to evaluate market-design interventions. Early artificial markets showed how stylized price dynamics can emerge from heterogeneous interacting agents `\cite{raberto2001,chakraborti2011}`. Subsequent work uses ABMs to study how transaction taxes and fee-like frictions affect volatility, liquidity, and market stability under different microstructures `\cite{westerhoff2006,mannaro2008,pellizzari2009}`. Other studies examine how rule changes related to hedging instruments, high-frequency trading, order cancellation, latency, and external information can amplify or dampen market instability `\cite{brock2009,leal2016,carro2015,gao2024,yagi2020}`. Our use of an ABM to test dynamic AMM fees follows this market-design logic: the fee rule is treated as a policy lever whose impact depends on heterogeneous agent responses and feedback through prices, routing, liquidity provision, and arbitrage.

Why here:

- This is probably the most important structural improvement to the paper. It creates a journal-facing bibliography layer and makes the dynamic-fee ABM methodology look established rather than idiosyncratic.

Priority: highest.

### 13. Dynamic fees in deployed AMMs: add academic dynamic-fee work before protocol docs

Location: lines 174--191.

Current relevant text:

> Beyond academic proposals, several large AMM families already deploy fee schedules ... Curve ... Trader Joe ... Meteora ... Algebra ...

Recommended insertion:

- Add a short first paragraph distinguishing academic dynamic-fee proposals from deployed protocol mechanics.
- Candidate citations: `\cite{cartea2023}` and, if added to the bibliography, the optimal dynamic-fee paper noted in the report (Baggiani, Herdegen, and Sánchez-Betancourt) after verifying metadata.
- Keep protocol documentation citations but make them secondary implementation evidence.

Suggested prose:

> Academic work on AMM fee choice emphasizes that the fee is not merely a revenue parameter but a control variable that trades off volume against adverse selection and arbitrage losses `\cite{milionis2023,cartea2023}`. Deployed protocols implement this idea through simpler, bounded controllers based on imbalance, volatility, or activity proxies.

Why here:

- This prevents the subsection from looking like a list of documentation pages.

Priority: high.

## Model section insertion points

### 14. Uniswap v3 mechanics: keep protocol citations narrow

Location: lines 193--292.

Current citations: `\cite{adams2021}`, `\cite{Adams2020UniswapVC}`.

Recommended action:

- Do not overload this section with academic citations. Protocol docs/whitepapers are appropriate here.
- If you want one academic citation near the beginning, add `\cite{angeris2020}` for AMM mechanics/arbitrage, but the current Section 2 literature review can carry most of the academic load.

Why here:

- This keeps the bibliography strategy clean: docs for mechanics, peer-reviewed papers for claims.

Priority: low.

### 15. Mempool / latency mechanism: cite MEV and HFT/latency literature

Location: lines 293--343.

Current relevant text:

> This captures confirmation latency ... stochastic transaction ordering ... arbitrageurs enjoying systematic priority.

Recommended insertion:

- Add `\cite{flashboys}` for MEV/priority/frontrunning where priority execution is introduced.
- Add traditional/artificial-market latency citations where block-time execution is framed as a latency/rule-design feature: `\cite{leal2016,gao2024,yagi2020}`.

Suggested prose:

> This design abstracts from the full block-building market, but preserves the market-design feature that latency and priority ordering can materially affect volatility, arbitrage, and liquidity provision, as emphasized in both MEV studies and artificial-market analyses of high-frequency trading and cancellation rules `\cite{flashboys,leal2016,gao2024,yagi2020}`.

Why here:

- The model's block-time and priority-arbitrage assumptions need external support beyond DeFi technical papers.

Priority: high.

### 16. Reference market / Heston model: add canonical Heston citation

Location: lines 352--380.

Current relevant text:

> To capture stylized facts ... we introduce a stochastic volatility regime based on the Heston model.

Recommended insertion:

- Add the canonical Heston paper citation (`heston1993`) if not already in the bibliography.

Suggested citation:

> ... based on the Heston model `\cite{heston1993}`.

Why here:

- This is a standard scientific citation currently missing from the provided excerpt. It is not in `biblio_additions.md`, but it is an obvious bibliography-quality improvement.

Priority: high.

### 17. Market impact: current citations are good, but fix prose and add canonical context if needed

Location: lines 381--394.

Current relevant text:

> Empirically, large trades tend to move prices in a concave way, consistent with the square-root law ... `\cite{PhysRevX.1.021006,Maitrier03042026}`.

Recommended insertion:

- Existing citations are good and journal-published.
- No additional citation is necessary unless you want a broader market-impact book/review.

Scientific note:

- The formula lists `\xi >= 0` in text but does not use `\xi` in Equation (impact). This is not a citation issue, but it is worth fixing later.

Priority: citation low; scientific cleanup medium.

### 18. LVR accounting: cite Milionis plus Cartea predictable loss

Location: lines 397--475.

Current citations: mostly `\cite{milionis2023}`.

Recommended insertion:

- Keep Milionis as primary.
- Add `\cite{cartea2023}` when discussing predictable loss and fee/LVR compensation.
- If adding a broader market-making sentence, cite `\cite{ho1983,avellaneda2008,gueant2016book}`.

Suggested prose near line 397:

> This stale-quote cost is the AMM analogue of adverse selection faced by liquidity suppliers in dealer and limit-order-book markets `\cite{ho1983,avellaneda2008,milionis2023,cartea2023}`.

Why here:

- This reinforces that LVR is not an isolated DeFi metric but an instance of adverse selection in liquidity supply.

Priority: high.

### 19. Arbitrageur model: cite AMM no-arbitrage and execution/speculation

Location: lines 514--624.

Current citation: `\cite{angeris2020}` near the no-arbitrage band.

Recommended insertion:

- Keep `\cite{angeris2020}`.
- Add `\cite{cartea2022execution}` where the arbitrageur chooses size to exploit DEX/CEX mispricing.
- Add `\cite{milionis2023,fritsch2024}` when describing arbitrage extraction as a cost to LPs.

Suggested prose:

> The arbitrageur is therefore both the mechanism enforcing price alignment and the channel through which stale AMM quotes become realized LP losses `\cite{angeris2020,milionis2023,fritsch2024,cartea2022execution}`.

Why here:

- It connects a model component to the empirical/theoretical literature on arbitrage losses.

Priority: medium-high.

### 20. LP strategies: cite strategic LP and active market-making literature

Location: lines 660--797.

Current relevant text:

> We model three LP cohorts ... passive ... active ... JIT ... Active LPs provide more liquidity near the current price and manage positions dynamically.

Recommended insertion:

- Cite `\cite{fan2023}` in the active LP paragraph.
- Cite `\cite{heimbach2022}` for short-lived/narrow positions if discussing active LP risks.
- Cite `\cite{avellaneda2008,gueant2016book}` if framing active LPs as automated market makers managing inventory/range risk.

Suggested prose:

> This active cohort is motivated by empirical evidence that Uniswap v3 LP outcomes depend strongly on range width and position timing, and by strategic-liquidity models in which LPs dynamically adjust ranges to optimize fee income and inventory exposure `\cite{heimbach2022,fan2023}`.

Why here:

- The active LP model is currently plausible but under-cited at the exact location where the mechanism is introduced.

Priority: high.

### 21. JIT liquidity: cite existing JIT/MEV references

Location: lines 798--807.

Current relevant text:

> The Jiter is a MEV-style liquidity provider ...

Recommended insertion:

- Add `\cite{wan2022just,flashboys}` in the first sentence.
- If discussing fee diversion or strategic short-lived liquidity, also cite `\cite{heimbach2022}` for empirical short-lifetime LP outcomes.

Suggested prose:

> The Jiter is a MEV-style liquidity provider, motivated by documented JIT liquidity and priority-ordering behavior in AMMs `\cite{wan2022just,flashboys}`.

Why here:

- JIT is a technical DeFi mechanism; the technical/MEV references are appropriate here.

Priority: high.

### 22. Smart router: cite execution/speculation and best execution / routing context

Location: lines 926--967.

Current relevant text:

> To endogenize venue choice, we introduce a smart-routing trader ... best-execution criterion ...

Recommended insertion:

- Add `\cite{cartea2022execution}` near the best-execution/routing framing.
- Consider adding a mainstream algorithmic trading execution reference: `\cite{cartea2015book}`.

Suggested prose:

> The smart router is a parsimonious version of best-execution logic in algorithmic trading, here specialized to the choice between an AMM and a reference venue `\cite{cartea2015book,cartea2022execution}`.

Why here:

- This reduces the sense that the router is an ad hoc model component.

Priority: medium.

### 23. Noise trader and order arrival: cite ABM artificial markets if adding methodological support

Location: lines 969--1005.

Current relevant text:

> The noise trader represents uninformed or urgency-driven flow ... arrivals are sampled independently ...

Recommended insertion:

- Add `\cite{raberto2001,chakraborti2011}` only if you want to justify stylized heterogeneous-agent order flow.
- Avoid over-citing this paragraph; the new ABM subsection should do most of this work.

Why here:

- Useful but not essential. Too many citations in every model paragraph can make the paper look defensive.

Priority: low-medium.

### 24. Dynamic fee schedules: connect controller design to academic fee-policy and protocol controllers

Location: lines 1007--1090.

Current relevant text:

> A central feature of the model is a time-varying taker fee ... We consider three modes ...

Recommended insertion:

- Add `\cite{cartea2023}` in the opening sentence about fees as a control variable.
- Add `\cite{westerhoff2006,mannaro2008,pellizzari2009}` in a short methodological sentence saying this is a fee-policy intervention.
- Keep protocol docs in the previous deployed-fees subsection, not necessarily in equations.

Suggested prose:

> In market-design terms, this controller treats the AMM fee as an endogenous transaction-cost rule, analogous to fee and tax interventions studied in artificial financial markets, but here targeted at AMM adverse selection `\cite{westerhoff2006,mannaro2008,pellizzari2009,cartea2023}`.

Why here:

- This is another direct answer to the supervisor's request: fee policy is an established ABM object.

Priority: very high.

## Results and conclusion insertion points

### 25. Microstructure diagnostics: cite empirical DeFi stylized facts plus HFT/latency ABMs

Location: lines 1303--1325.

Current relevant text:

> The DEX series exhibits pronounced spikes ... spike-and-reversion dynamics documented empirically in `\cite{dinosse_gatta}`.

Recommended insertion:

- Keep `\cite{dinosse_gatta}` for the specific DeFi stylized fact.
- Add `\cite{leal2016,gao2024,yagi2020}` if describing the general latency/fast-trader mechanism.

Suggested prose:

> More broadly, this channel is consistent with artificial-market evidence that latency and high-frequency priority mechanisms can generate short-horizon volatility and instability even when they improve price alignment in normal periods `\cite{leal2016,gao2024,yagi2020}`.

Why here:

- It strengthens the diagnostic section and prevents the empirical validation from relying only on the author's own related paper.

Priority: medium-high.

### 26. Cross-scenario PnL summary: cite Fritsch, Milionis, Cartea when interpreting LVR vs fees

Location: lines 1365--1395.

Current relevant text:

> This establishes the baseline microstructure result ... Dynamic fees materially change this ranking ...

Recommended insertion:

- Add citations in interpretation sentences, not next to every numerical result.
- Candidate citations: `\cite{milionis2023,fritsch2024,cartea2023}`.

Suggested prose:

> The negative static-fee benchmark is consistent with the LVR and arbitrage-loss literature: when quotes are stale and the fee wedge is fixed, arbitrage losses can dominate fee income for standing liquidity `\cite{milionis2023,fritsch2024,cartea2023}`.

Why here:

- It ties simulation evidence back to established external evidence.

Priority: medium.

### 27. Competitiveness / DEX share paragraph: cite smart routing / execution

Location: lines 1407--1424.

Current relevant text:

> The smart router trades off execution quality against fees, so higher endogenous fees can reduce AMM competitiveness.

Recommended insertion:

- Candidate citations: `\cite{cartea2015book,cartea2022execution}`.
- If emphasizing transaction-cost policy tradeoffs, add `\cite{westerhoff2006,pellizzari2009}`.

Suggested prose:

> This is the routing-side counterpart of the classic market-design trade-off: raising transaction costs can protect liquidity suppliers but may also reduce venue competitiveness and trading activity `\cite{westerhoff2006,pellizzari2009,cartea2015book}`.

Why here:

- It shows the simulation outcome is not only AMM-specific; it is the same economics as market-design fee interventions.

Priority: medium-high.

### 28. LVR vs fees vs block size: cite LVR theory and latency literature

Location: lines 1728--1839.

Current relevant text:

> The LVR theory implies that ... `\mathbb{E}[LVR] \propto \sigma^2 \Delta t` ... Larger blocks ...

Recommended insertion:

- Add `\cite{milionis2023}` immediately after “The LVR theory implies”.
- Add `\cite{leal2016,gao2024}` when discussing block length / latency / priority effects.

Suggested prose:

> The LVR scaling follows the continuous-time adverse-selection analysis of AMM liquidity provision `\cite{milionis2023}`, while the block-size experiment is analogous to latency and market-speed counterfactuals in artificial-market studies of high-frequency trading `\cite{leal2016,gao2024}`.

Why here:

- The section currently contains a theoretical scaling claim that should be directly cited.

Priority: high.

### 29. Conclusions: restate contribution as market-design ABM, not only DeFi simulation

Location: lines 1848--1925.

Current relevant text:

> This paper introduced a granular agent-based model ... Overall, the experiments support three broad conclusions ...

Recommended insertion:

Add one sentence in the first conclusion paragraph or just before “Overall”.

Suggested prose:

> Methodologically, the paper contributes to the artificial financial-market literature by treating AMM fee adaptation as a market-design intervention whose effects are mediated by heterogeneous liquidity takers, arbitrageurs, routers, and liquidity providers `\cite{raberto2001,chakraborti2011,westerhoff2006,mizuta2020}`.

Why here:

- This gives the final page the same broader positioning that should appear in the introduction.

Priority: very high.

## Bibliography metadata cleanup priorities

These are not paper-text insertion points, but they matter for the supervisor's criticism.

1. Upgrade `fritsch2024` from arXiv metadata to ACM Web Conference Companion metadata:
   - Title: “Measuring Arbitrage Losses and Profitability of AMM Liquidity”.
   - DOI: `10.1145/3589335.3651961`.
   - OpenAlex record: `https://openalex.org/W4396843707`.

2. Upgrade `fan2023` with DOI:
   - DOI: `10.4230/LIPIcs.AFT.2023.25`.
   - OpenAlex record: `https://openalex.org/W4287114683`.
   - Check author names carefully; current `.bib` author list appears inconsistent with OpenAlex/report spelling.

3. Keep `heimbach2022` as a conference paper, not as an arXiv paper:
   - DOI: `10.1145/3558535.3559772`.
   - OpenAlex record: `https://openalex.org/W4280596911`.

4. Keep `milionis2023` despite arXiv status:
   - This is a foundational LVR paper and is directly central to the paper.
   - Add DOI `10.48550/arxiv.2208.06046` if desired.
   - OpenAlex record: `https://openalex.org/W4291961646`.

5. For `cartea2023`, do not overstate final venue status unless verified:
   - OpenAlex record for Predictable Loss: `https://openalex.org/W4312956485`, DOI `10.2139/ssrn.4273989`.
   - OpenAlex record for Execution and Speculation: `https://openalex.org/W4285216148`, DOI `10.2139/ssrn.4144743`.

6. Add traditional microstructure and ABM references with complete journal/book metadata before expanding with more DeFi preprints.

## Minimal implementation order if editing later

If you later decide to modify the paper, I would implement in this order:

1. Add the new ABM market-design subsection after the current AMM simulation subsection (before deployed dynamic fees).
2. Add the Introduction bridge sentence that frames AMM fees as market-design variables.
3. Add traditional market-making citations around LVR/adverse selection.
4. Add dynamic-fee/transaction-tax citations around the fee controller.
5. Clean BibTeX metadata for existing DeFi references.
6. Only then add additional model-section citations.

This order gives the largest bibliographic improvement with the smallest risk of turning the manuscript into a citation dump.
