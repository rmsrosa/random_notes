# Markov Chain Monte Carlo (MCMC)

Markov Chain Monte Carlo methods take its name from the following characteristics.

1. One designs a *Markov chain* $X_0, X_1, X_2, \ldots,$ with the property that any initial distribution $X_0$ converges to a desired distribution $P,$ $X_n \rightarrow X_\infty \sim P.$
2. One performs a *Monte-Carlo* sample by computing a (single) trajectory $(x_n)_n$ such that, after an initial transient time called *the burn-in time* $N,$ the distribution of the (tail of the) sample trajectory, $x_{N+1}, x_{N+2}, \ldots,$ is close to the stationary distribution $X_\infty \sim P.$

For these two properties to hold, the desired distribution has to be a stationary distribution of the Markov chain, and the Markov chain has to be ergodic, so that (i) this is the only stationary distribution; (ii) any initial distribution converges to the stationary distribution; (iii) a single trajectory (almost surely) suffices to walk about the event space and statistically represent the desired stationary distribution.

There are a few such MCMC methods, such as the original *Metropolis algorithm;* its extension known as *Metropolis-Hastings method;* and the *Hamiltonian Monte-Carlo (HMC) method.*
