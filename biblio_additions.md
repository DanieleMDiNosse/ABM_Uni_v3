# Updated Bibliography Strategy for Your ABM DeFi Paper

## Executive positioning

Based on the draft you uploaded, the bibliography should be rebuilt around three layers. First, keep a **small** protocol layer for implementation facts only, such as the Uniswap v3 whitepaper and the protocol-specific documentation for adaptive fee rules. Second, build a **scientific DeFi/AMM core** around papers that are already published, conference-published, or clearly publication-bound. Third, add a **non-DeFi ABM market-design layer** showing that your paper belongs to a broader literature that uses agent-based models to study the effects of market rule changes such as transaction fees, latency, order cancellation, fragmentation, and external information. That is the move your supervisor is asking for, and it is also the move that will make the paper look much more journal-ready. citeturn60view0turn47view2turn46view1turn52view1turn57search7turn56view0

My strongest recommendation is to write the paper as a **JEDC-first submission**, with *Simulation Modelling Practice and Theory* as the backup that becomes more attractive only if you emphasize simulator design, calibration, experimentation, and scenario analysis more heavily. *Journal of Banking & Finance* is possible only if you materially strengthen the bridge to mainstream market-microstructure and liquidity-provision theory, because that journal has a broader capital-markets and finance remit, while JEDC is directly aligned with computational economics, dynamic models, and heterogeneous-agent approaches. citeturn32search0turn32search1turn60view0turn47view2

The most useful new fact for your positioning is that there is now a **direct AMM paper forthcoming in JEDC**—Cartea, Drissi, and Monga, *Decentralised Finance and Automated Market Making: Execution and Speculation*. That gives you a high-value “same-journal” anchor. In parallel, the same authors’ *Predictable Loss and Optimal Liquidity Provision* is marked as forthcoming in *SIAM Journal on Financial Mathematics*, which strengthens the case that AMM liquidity provision is now entering established quantitative-finance venues rather than remaining only in whitepapers and arXiv notes. citeturn60view0turn47view2

## The literature narrative you should build

The cleanest literature story is this. AMMs are not just “crypto engineering”; they are a **market-making technology**. Their fee rule is a **market-design policy variable**, analogous to spread design, transaction taxes, cancellation rules, or latency/fragmentation rules in traditional markets. ABMs are particularly well-suited to study such policy changes because they let you model heterogeneous agents, bounded rationality, endogenous liquidity, feedback effects, and out-of-equilibrium dynamics directly. This is also how recent AMM papers frame themselves: next to optimal trading, market making, adverse selection, and algorithmic-liquidity provision in traditional markets. citeturn60view0turn47view2turn46view3turn49view2

That narrative becomes much stronger if you explicitly split the literature review into four blocks: **AMM microstructure and LP profitability**, **dynamic fees and adverse selection**, **ABM market-design papers on policy changes**, and **traditional market-making / microstructure theory**. Your current draft already has the first block, but it underweights the last three. That is exactly why it currently reads as too close to protocol documentation and too far from the literature that journal referees will expect. citeturn46view3turn47view2turn60view0turn49view2

For JBF in particular, you should not let the review stop at DeFi citations. Cartea, Drissi, and Monga explicitly place their AMM work in the broader algorithmic-trading and liquidity-provision tradition, pointing back to dealer-market and market-making references such as Ho and Stoll, Avellaneda and Stoikov, Cartea–Jaimungal–Penalva, and Guéant. Even if you stay with an ABM contribution, adding that layer is what signals that you are contributing to market microstructure rather than only to protocol engineering. citeturn49view2

## The references I would add immediately

### DeFi and AMM papers that materially improve the bibliography

The highest-value upgrades on the DeFi side are the papers that are already venue-validated. Fan et al. explicitly state on arXiv that *Strategic Liquidity Provision in Uniswap v3* appeared in the **Proceedings of AFT ’23**, and Heimbach et al. have a publication DOI linked from arXiv for *Risks and Returns of Uniswap V3 Liquidity Providers*, which matches the AFT paper already listed in your draft. These are much better journal/conference-facing anchors than leaving the literature review dominated by protocol docs. citeturn46view1turn52view1

The two Cartea–Drissi–Monga papers are especially valuable for your paper’s positioning. *Execution and Speculation* is marked **forthcoming in JEDC**, and *Predictable Loss and Optimal Liquidity Provision* is marked **forthcoming in SIAM Journal on Financial Mathematics**. Those two papers give you an extremely strong bridge from DeFi into recognized finance and computational-economics outlets. citeturn60view0turn47view2

Milionis, Moallemi, Roughgarden, and Zhang remain essential because *Automated Market Making and Loss-Versus-Rebalancing* is still one of the cleanest adverse-selection interpretations of LP losses, and it directly supports a dynamic-fee paper like yours. Even if it remains an arXiv reference in your bibliography, it is a field-defining one and should stay. The same is true, though at a lower priority, for newer dynamic-fee work such as Baggiani, Herdegen, and Sánchez-Betancourt on optimal dynamic fees. citeturn46view3turn60view3

### Non-DeFi ABM papers on policy and market design

For the non-DeFi ABM layer, the most natural analogy to your contribution is the **transaction-tax / fee** literature. The classic references here are Westerhoff and Dieci on Keynes–Tobin taxes in *Journal of Economic Dynamics and Control*, Mannaro–Marchesi–Setzu on Tobin-like transaction taxes in *Journal of Economic Behavior & Organization*, and Pellizzari–Westerhoff on transaction taxes under different microstructures, also in *Journal of Economic Behavior & Organization*. Those papers are precisely the type of “fee-policy in financial ABMs” references your supervisor had in mind. citeturn57search7

A second excellent analogue is the literature on **market-design changes that affect speed, cancellation, or fragmentation**. Leal, Napoletano, Roventini, and Fagiolo show in their ABM that high-frequency trading increases volatility, contributes to flash crashes, and that higher HFT order-cancellation rates increase the incidence of flash crashes while shortening their duration. Gao et al. later develop an HFT ABM for flash-crash scenarios and publish it in *Journal of Artificial Societies and Social Simulation*, explicitly emphasizing its use for policy advice and robustness testing. Yagi, Masuda, and Mizuta then study liquidity effects of HFT in an artificial market and publish in *IEEE Transactions on Computational Social Systems*. Together, these papers give you the “latency / fast trading / cancellation” side of the requested literature. citeturn62view2turn52view4turn52view5

A third parallel is **information release / exogenous information**. Carro, Toral, and San Miguel develop a stochastic ABM with herding and external information, calibrate it with the ZEW indicator and DAX, and publish it in *PLoS ONE*. This is exactly the kind of non-DeFi citation you can use when you want to say that ABMs are routinely used to study how exogenous signals or public information propagate into market outcomes. citeturn56view0turn56view1

Finally, a good methodological umbrella citation is Mizuta’s paper on using ABMs to design markets that “work well,” published in the 2020 IEEE SSCI proceedings. It is not a top-field journal paper, but it is useful because it says explicitly that ABMs are increasingly used to study the consequences of detailed market rules and regulations. That sentence alone helps justify your paper’s use of an ABM for fee-policy design. citeturn51view0

## A submission-oriented bibliography shortlist

The list below is the bibliography I would prioritize. I have separated it into **must-add**, **strong support**, and **background only**. For some older economics papers, I am highly confident on title/journal pairing but I would still do one last DOI/page-range verification when you finalize the `.bib`.

### Must-add for the revised literature review

Cartea, Á., Drissi, F., & Monga, M. *Decentralised Finance and Automated Market Making: Execution and Speculation*. Forthcoming in *Journal of Economic Dynamics and Control*. citeturn60view0

Cartea, Á., Drissi, F., & Monga, M. *Decentralised Finance and Automated Market Making: Predictable Loss and Optimal Liquidity Provision*. Forthcoming in *SIAM Journal on Financial Mathematics*. citeturn47view2

Fan, Z., Marmolejo-Cossío, F., Moroz, D. J., Neuder, M., Rao, R., & Parkes, D. C. *Strategic Liquidity Provision in Uniswap v3*. In *Proceedings of AFT ’23*. citeturn46view1

Heimbach, L., Schertenleib, E., & Wattenhofer, R. *Risks and Returns of Uniswap V3 Liquidity Providers*. Published version linked on arXiv via DOI 10.1145/3558535.3559772. citeturn52view1

Milionis, J., Moallemi, C. C., Roughgarden, T., & Zhang, A. L. *Automated Market Making and Loss-Versus-Rebalancing*. arXiv. citeturn46view3

Maire, B., & Wunsch, M. (2024). *Market Neutral Liquidity Provision*. *Ledger*, 9, 115–134.

Westerhoff, F., & Dieci, R. (2006). *The Effectiveness of Keynes-Tobin Transaction Taxes when Heterogeneous Agents Can Trade in Different Markets: A Behavioral Finance Approach*. *Journal of Economic Dynamics and Control*. citeturn57search7

Mannaro, K., Marchesi, M., & Setzu, A. (2008). *Using an Artificial Financial Market for Assessing the Impact of Tobin-Like Transaction Taxes*. *Journal of Economic Behavior & Organization*, 67(2), 445–462. citeturn57search7

Pellizzari, P., & Westerhoff, F. (2009). *Some Effects of Transaction Taxes under Different Microstructures*. *Journal of Economic Behavior & Organization*, 72(3), 850–863.

Brock, W. A., Hommes, C. H., & Wagener, F. O. O. (2009). *More Hedging Instruments May Destabilize Markets*. *Journal of Economic Dynamics and Control*. citeturn40search8

### Strong support for the ABM policy framing

Leal, S. J., Napoletano, M., Roventini, A., & Fagiolo, G. *Rock around the Clock: An Agent-Based Model of Low- and High-Frequency Trading*. Published paper corresponding to the 2014 arXiv version; use this to support arguments on latency, order cancellation, and flash-crash dynamics. citeturn62view2

Gao, K., Vytelingum, P., Weston, S., Luk, W., & Guo, C. (2024). *High-frequency Financial Market Simulation and Flash Crash Scenarios Analysis: An Agent-Based Modelling Approach*. *Journal of Artificial Societies and Social Simulation*, 27(2), 8. DOI 10.18564/jasss.5403. citeturn52view4

Yagi, I., Masuda, Y., & Mizuta, T. (2020). *Analysis of the Impact of High-Frequency Trading on Artificial Market Liquidity*. *IEEE Transactions on Computational Social Systems*. citeturn52view5

Carro, A., Toral, R., & San Miguel, M. (2015). *Markets, Herding and Response to External Information*. *PLoS ONE*, 10(7), e0133287. DOI 10.1371/journal.pone.0133287. citeturn56view0turn56view1

Mizuta, T. (2020). *An Agent-Based Model for Designing a Financial Market that Works Well*. In *2020 IEEE Symposium Series on Computational Intelligence*. DOI 10.1109/SSCI47803.2020.9308376. citeturn51view0

Raberto, M., Cincotti, S., Focardi, S. M., & Marchesi, M. (2001). *Agent-Based Simulation of a Financial Market*. *Physica A*. DOI 10.1016/S0378-4371(01)00312-0. citeturn62view0turn62view1

### Mainstream microstructure anchors for a stronger JBF-style pitch

Ho, T., & Stoll, H. R. (1983). *The Dynamics of Dealer Markets under Competition*. *The Journal of Finance*.

Avellaneda, M., & Stoikov, S. (2008). *High-Frequency Trading in a Limit Order Book*. *Quantitative Finance*.

Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

Guéant, O. (2016). *The Financial Mathematics of Market Liquidity: From Optimal Execution to Market Making*. CRC Press.

These are not optional if you want the paper to look like a finance-market-microstructure contribution rather than only a DeFi simulation paper. That bridge is explicitly invoked by recent AMM theory papers. citeturn49view2

### Keep, but move to a lighter protocol-only role

Adams, H., Zinsmeister, N., Salem, M., Keefer, R., & Robinson, D. (2021). *Uniswap v3 Core Whitepaper*.

Angeris, G., Kao, H.-T., Chiang, R., Noyes, C., & Chitra, T. (2020). *An Analysis of Uniswap Markets*. arXiv.

Algebra, Curve, Meteora, and Trader Joe documentation on adaptive / dynamic fees.

These should remain **only** where you need to document protocol mechanics or implementation details. They should not dominate the literature review anymore.

## How I would rewrite the literature review

Open the review with AMMs as a new market-making institution, not with protocol documents. Then cite Angeris, Milionis, Heimbach, Fan, Maire–Wunsch, and the two Cartea papers as the DeFi core. After that, add a short subsection titled something like **“ABM for market-design and policy analysis in financial markets”** and place there the transaction-tax papers, Brock–Hommes–Wagener on destabilizing design changes, Leal et al. on HFT/cancellation, Gao et al. on flash-crash policy scenarios, and Carro et al. on external information. That subsection is the one that answers your supervisor directly. citeturn46view3turn47view2turn60view0turn57search7turn62view2turn52view4turn56view0

Then explicitly state your paper’s contribution in those terms: you are not only studying DeFi LP profitability; you are using an ABM to study the effect of a **fee-policy intervention** on market quality, LP behavior, arbitrage, and liquidity allocation in a market-making system with endogenous feedback. Once phrased that way, the analogy to transaction-tax ABMs, HFT-latency ABMs, and information-response ABMs becomes immediate. citeturn57search7turn62view2turn56view0turn51view0

For JEDC, I would make the lead sentence of the positioning paragraph essentially say that your paper joins a literature in computational finance and agent-based market design that studies how rule changes alter equilibrium-like outcomes in systems with heterogeneous interacting agents, and that AMM fee rules are the DeFi counterpart of classic market-design levers such as transaction costs, venue structure, or information conditions. That will read as much more mature than a narrow “adaptive fee for CLMMs” framing. citeturn32search0turn60view0turn47view2turn57search7turn56view0

## Open questions and limitations

Some older economics references in the transaction-tax and JEDC literature were easy to identify at the level of **title, author, and journal placement**, but I did not re-verify every page range and DOI directly from the publisher during this pass. In particular, before final submission you should still do a last mechanical cleanup of the `.bib` file for older JEDC / JEBO items such as Westerhoff–Dieci, Brock–Hommes–Wagener, and Pellizzari–Westerhoff. The conceptual recommendation is high-confidence; the final typesetting of those entries should still be checked against the publisher record.

The strongest same-journal alignment I found is clearly with **JEDC**, not with *Simulation Modelling Practice and Theory* or *Journal of Banking & Finance*. That does not mean the other two are impossible targets; it means that, with the literature now available, JEDC gives you the cleanest and most defensible bibliographic narrative. citeturn32search0turn32search1turn60view0