# Markov Chain Monte Carlo (MCMC)

Markov Chain Monte Carlo methods take its name from the following facts.

1. One designs a *Markov chain* $X_0, X_1, X_2, \ldots,$ with the property that any initial distribution $X_0$ converges to a desired distribution,
```math
X_n \rightarrow X_\infty \sim \mathbb{P}.
```
2. One performs a *Monte-Carlo* sample by computing a (single) trajectory $x_n,$ such that, after an initial transient time called *burn-in time* $N,$ the distribution of the (tail of the) sample trajectory,
```math
    x_N, x_{N+1}, \ldots,
```
has a distribution close to the stationary one $X_\infty \sim \mathbb{P}.$

For these two properties to hold, the desired distribution has to be a stationary distribution of the Markov chain, and the Markov chain has to be ergodic, so that (i) this is the only stationary distribution; (ii) any initial distribution converges to the stationary distribution; (iii) a single trajectory suffices to walk about, and statistically represent, the desired stationary distribution.

There are a few such MCMC methods, such as the original *Metropolis algorithm;* its extension known as *Metropolis-Hastings method;* and the *Hamiltonian Monte-Carlo (HMC) method.*
