# Irreducibility

Irreducibility is a fundamental concept related to the uniqueness of an invariant measure, when it exists, but it does not necessarily implies that it exists.

## Definition

A time-homogeneous Markov chain $(X_n)_n$ with $n$-step transition probability $K_n(x, \cdot)$ is called **$P$-irreducible,** with respect to a probability distribution $P,$ if
```math
    E\in\mathcal{B}(\mathcal{X}), \;P(E) > 0 \Longrightarrow \sum_{n\in \mathbb{N}} K_n(x, E) > 0, \quad \forall x\in \mathcal{X}.
```
This is equivalent to assuming that, 
```math
    x\in\mathcal{X}, \;E\in\mathcal{B}(\mathcal{X}), \; P(E) > 0 \Longrightarrow \exists n=n(x, E)\in\mathbb{N}, \; K_n(x, E) > 0.
```
The Markov chain is called **strongly $P$-irreducible** when $n(x, E) = 1$ for all such $x$ and $E.$

Irreducibility means that any measurable set with positive measure is eventually reached by the chain, with positive probability, starting from any point in $\mathcal{X}.$

There are several concepts related to irreducibility

### First return time

The integer $n(x, E)$ in the equivalent definition of irreducibility can be taken to be the *first* such integer, which gives the notion of *first return* time. We do not need irreducibility to define the first return as long as we agree that it is infinity when it does not return.

More precisely, given a Borel set $E\subset \mathcal{X}$ and a point $x,$ of a given time-discrete Markov chain $(X_n)_n,$ the **first return** of $x$ to $E$ is defined by
```math
    n(x, E) = \inf\left\{n\in\mathcal{N}\cup\{+\infty\}; \; n = \infty or X_n|_{X_0 = x} \in E\right\}.
```

### Stopping time

Instead of a function $n(x, E)$ from $\mathcal{X}\times\mathcal{B}(\mathcal{X})$ to $\mathbb{N},$ we can consider a random variable version of the first return map, which is a random variable from $\Omega$ to $\mathbb{N},$ defined as follows.

Given a Borel set $E\subset \mathcal{X},$ the **stopping time** $\tau_E$ at $E$ of the Markov chain is the random variable defined by
```math
    \tau_E = \inf\{ n\in\mathbb{N}\cup\{+\infty\}; \; n = +\infty \textrm{ or } X_n\in E\}.
```
It should be clear that $\tau_E = \infty,$ if the chain never reaches $E,$ or it is the first time $n$ such that $X_n$ reaches $E.$

The first return time and the stopping time are related by $\tau_E(\omega) = n(X_0(\omega), E),$ for any sample $\omega\in\Omega.$

The quantity
```math
    P(\tau_E < \infty)
```
is the *probability of return to $E$ in a finite number of steps.*

### Number of passages

Another useful quantity is the random variable for the **number of passages** in $E,$
```math
    \eta_E = \sum_{n=1}^\infty \mathbb{1}_{X_n \in A}.
```

### Example 

Alternating chain example (e.g. $X_{n+1} = X_n \pm 2,$, so only even or odd integers are reached, so it is not irreducible). Continuous example (e.g. something like $X_{n+1} = [X_n] \pm [X_n] + 2 + Beta$)

## Irreducibility in the discrete-state case

In the discrete-state case, $P$-irreducibility can be stated with point sets, so that the Markov chain is $P$-irreducible if, and only if,
```math
    x, y\in\mathcal{X},\; P(y) > 0 \Longrightarrow \exists n=n(x, y)\in\mathbb{N}, \; K_n(x, y) > 0.
```

### Positivity of an invariant probability distribution

If a finite-state Markov chain has an invariant probability distribution $P$ and it is $P$-irreducible, then $P$ must be everywhere strictly positive, i.e. $P(x) > 0,$ for all $x\in \mathcal{X}.$

Indeed, since $P$ is nontrivial, there exists $y\in\mathcal{X}$ such that $P(y) > 0.$ Now, for any $x,$ since the chain is $P$ irreducible, we have $K_n(x, y) > 0,$ for some $n\in\mathbb{N}.$

Since $P$ is invariant, we have $P K_n = P,$ and then
```math
    P(x) = \sum_{z\in\mathcal{X}} K_n(z, x) P(z).
```
This can be estimated from below by restricting the summation to $z = y,$ so that
```math
    P(x) = \sum_{z\in\mathcal{X}} K_n(z, x) P(z) \geq K_n(y, x) P(z) > 0,
```
where we used that $K_n(y, x) > 0$ and $P(z) > 0.$ Thus,
```math
    P(x) > 0, \quad \forall x\in\mathcal{X},
```
showing that $P$ is everywhere strictly positive.

### Uniqueness of the invariant probability distribution

Suppose, again, that we have a finite-state Markov chain with an invariant probability distribution $P$ which is $P$-irreducible. We have just seen that $P$ must be everywhere strictly positive, i.e. $P(x) > 0,$ for all $x\in \mathcal{X}.$ We now use this to show that the invariant probability distribution must be unique.


