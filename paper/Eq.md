Eq. (34) gives the token deposits needed to mint liquidity $L$ in range $$$i_a,i_b$$$. In every case, both components are *linear in $L$* (they’re always $L\times(\text{something that depends only on }s_t,s_a,s_b)$). So, for fixed $(s_t,s_a,s_b)$,
$$
\Delta x(L)=L\,\Delta x(1),\qquad \Delta y(L)=L\,\Delta y(1),
$$
where $(\Delta x(1),\Delta y(1))$ is just Eq. (34) evaluated at $L=1$.

Because the LP’s wallet $\mathcal W^1_{j,t}$ is entirely in token 1, minting $L$ costs (in token 1 units):
- $m_t\,\Delta x(L)$ to buy the required token 0 amount $\Delta x(L)$ at CEX price $m_t$ (token1 per token0),
- plus $\Delta y(L)$ token 1 deposited directly.

So the total token-1 budget needed is
$$
m_t\Delta x(L)+\Delta y(L)=m_t\bigl(L\Delta x(1)\bigr)+L\Delta y(1)=L\bigl(m_t\Delta x(1)+\Delta y(1)\bigr).
$$

Setting this equal to the available cash and calling the largest feasible $L$ “$L^{\max}_{j,t}($$i_a,i_b$$)$” gives Eq. (35):
$$
\mathcal W^1_{j,t}=L^{\max}_{j,t}(i_a,i_b)\bigl(m_t\Delta x(1)+\Delta y(1)\bigr).
$$



Eq. 17 in ABM_paper.pdf is consistent with the instantaneous LVR formula in LVR.pdf.

- In LVR.pdf, Theorem 1 defines instantaneous LVR as

$$
\ell(\sigma,P)=\frac{\sigma^2 P^2}{2},\bigl|x^{*,\prime}(P)\bigr|.
$$

Here ($P$) is the external/reference (true) market price process used by the rebalancing benchmark (the CEX price in your ABM notation), not the
potentially-stale on-chain marginal price.
- In ABM_paper.pdf, the external reference price is denoted ($m_t$). That’s why they write “with ($p=m_t$)”: it’s just a notation mapping $(P_t \leftrightarrow
m_t)$ (and evaluating the demand-curve slope at the current reference price).
- For a Uniswap v3 position in-range, token-0 inventory as a function of sqrt-price $(s=\sqrt{p})$ is (their Eq. 15)

$$
x(s)=L_t\Bigl(\frac{1}{s}-\frac{1}{s_b}\Bigr),
$$
so
$$
\left|\frac{dx}{dp}\right|
=\left|\frac{dx}{ds}\frac{ds}{dp}\right|
=\left|\left(-\frac{L_t}{s^2}\right)\left(\frac{1}{2s}\right)\right|
=\frac{L_t}{2p^{3/2}}.
$$
Plugging into (\ell) and evaluating at (p=m_t):
$$
\ell_t^{v3}=\frac{\sigma_t^2 m_t^2}{2}\cdot \frac{L_t}{2m_t^{3/2}}
=\frac{\sigma_t^2 L_t\sqrt{m_t}}{4},
$$
which matches their Eq. (17).

So the “(p=m_t)” step isn’t an extra assumption; it’s that in the LVR theory (p) (or (P_t)) is the reference market price, and in the ABM that reference
market price is called (m_t).