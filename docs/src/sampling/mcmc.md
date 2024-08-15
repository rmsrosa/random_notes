# Markov Chain Monte Carlo (MCMC)

Markov Chain Monte Carlo methods take its name from the following facts.

1. One designs a *Markov chain* $X_0, X_1, X_2, \ldots,$ with the property that any initial distribution $X_0$ converges to a desired distribution,
```math
X_n \rightarrow X_\infty \sim \mathbb{P}.
```
2. The Markov chain has to be ergodic so that we only need one *Monte-Carlo* sample trajectory $x_n,$ so that, after an initial transient time called *burn-in time* $N,$ the distribution of the (tail of the) sample trajectory
```math
    x_N, x_{N+1}, \ldots,
```
has a distribution close to the stationary one $X_\infty \sim \mathbb{P}.$

There are a few such MCMC methods, such as the original *Metropolis algorithm;* its extension known as *Metropolis-Hastings method;* and the *Hamiltonian Monte-Carlo (HMC) method.*
