# Irreducibility and recurrence

Irreducibility is a fundamental concept related to the uniqueness of an invariant measure, when it exists, but it does not necessarily implies that it exists.

## Irreducibility

A time-homogeneous Markov chain $(X_n)_n$ with $n$-step transition probability $K_n(x, \cdot)$ is called **$P$-irreducible,** with respect to a probability distribution $P,$ if
```math
    E\in\mathcal{B}(\mathcal{X}), \;P(E) > 0 \Longrightarrow \sum_{n\in \mathbb{N}} A_n(x, E) > 0, \quad \forall x\in \mathcal{X}.
```
This is equivalent to assuming that, 
```math
    x\in\mathcal{X}, \;E\in\mathcal{B}(\mathcal{X}), \; P(E) > 0 \Longrightarrow \exists n=n(x, E)\in\mathbb{N}, \; K_n(x, E) > 0.
```
The Markov chain is called **strongly $P$-irreducible** when $n(x, E) = 1$ for all such $x$ and $E.$

Irreducibility means that any measurable set with positive measure is eventually reached by the chain, with positive probability, starting from any point in $\mathcal{X}.$

There are several concepts related to irreducibility

### First return


### First return time and stopping time

The integer $n(x, E)$ in the equivalent definition of irreducibility can be taken to be the *first* such integer, which gives the notion of *first return* time. We do not need irreducibility to define the first return as long as we agree that it is infinity when it does not return.

More precisely, given a Borel set $E\subset \mathcal{X}$ and a point $x,$ of a given time-discrete Markov chain $(X_n)_n,$ the **first return** of $x$ to $E$ is defined by
```math
    n(x, E) = \inf\left\{n\in\mathcal{N}\cup\{+\infty\}; \; n = \infty or X_n|_{X_0 = x} \in E\right\}.
```

Given a Borel set $E\subset \mathcal{X},$ the **stopping time** $\tau_E$ at $E$ of the Markov chain is defined to be
```math
    \tau_E = \inf\{ n\in\mathbb{N}\cup\{+\infty\}; \; n = +\infty \textrm{ or } X_n\in E\}.
```
It should be clear that $\tau_E = \infty,$ if the chain never reaches $E,$ or $\tau_E = $ first time $n$ such that $X_n$ reaches $E.$ 

The quantity
```math
    P(\tau_E < \infty | X_0 = x)
```
is the *probability of return to $E$ in a finite number of steps.*

### Number of passages

Another useful quantity is the **number of passages** in $E,$
```math
    \eta_E = \sum_{n=1}^\infty \mathbb{1}_{X_n \in A}.
```

Alternating chain example (e.g. $X_{n+1} = X_n \pm 2,$, so only even or odd integers are reached, so it is not irreducible). Continuous example (e.g. something like $X_{n+1} = [X_n] \pm [X_n] + 2 + Beta$)

## Recurrence
