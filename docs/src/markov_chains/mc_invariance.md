# Invariant distributions

Of particular interest, for time-homogeneous Markov chains, are the stationary distributions, i.e. such that $X_n = X_0,$ for all $n=0, 1, 2, \ldots.$ This is the same as saying that they all have the same distribution $P,$ which is invariant by the Markov operator.

## Definition

More precisely, a probability distribution $P$ on $\mathcal{X}$ is **(time-)invariant** or **stationary** for a time-homogenous discrete-time Markov chain $(X_n)_n$ when $P = PK,$ i.e.
```math
    P(E) = \int_{\mathcal{X}} K(x, E);\mathrm{d}P(x),
```
where $K(x, E) = K_1(x, E) = K_{n,n+1}(x, E)$ is the one-step transition probability of the Markov chain and the $K$ in the equation $P = PK$ is the associated Markov operator on the space $\mathcal{P}(\mathcal{X})$ of probability distributions on $\mathcal{X}.$

Sometimes it might be useful to consider measures with are not necessarily probability distributions. They may be finite or infinite, and $\sigma$-finite or not. In any case, we call a measure $\mu$ **(time-)invariant** or **stationary** for the Markov chain if $\mu = \mu K.$

## Examples

A Markov chain may or may not have an invariant distribution and it may have a unique invariant distribution or several ones. This is a main topic of interest in the theory of Markov chains. Below, we give a few examples with these cases of uniqueness, non-uniqueness, and non-existence of invariant probability measures, also called stationary distributions.

### A two-state Markov chain

Let $\mathcal{X} = \{1, 2\}$ and consider the Markov chain characterized by the one-step transition distribution
```math
    K = (K(i, j))_{i,j=1}^2 = \begin{bmatrix}
        1 - \alpha & \alpha \\
        \beta & 1 - \beta
    \end{bmatrix},
```
where $0 < \alpha, \beta < 1.$ We can look for an invariant state by solving $P = PK,$ with $P = (p, 1-p).$ We have
```math
    \begin{cases}
        p = p(1-\alpha) + (1-p)\beta, \\
        1 - p = p\alpha + (1-p)(1-\beta)
    \end{cases}.
```
This can be simplified to
```math
    p(\alpha + \beta) = \beta,
```
hence
```math
    p = \frac{\beta}{\alpha + \beta}.
```
Thus, there is a unique invariant distribution, which is given by
```math
    P = \left(\frac{\beta}{\alpha + \beta}, \frac{\alpha}{\alpha + \beta}\right).
```

If $\alpha > \beta,$ then state $1$ is more likely to go to state $2$ than the other way around. And this also means that, in the stationary state, state $2$ is more likely than state $1$. Similarly for $\beta > \alpha.$ If both are equal, then the stationary state is uniform in the state variables. In any case, we will see further on that any initial distribution converges to this unique stationary distribution.

When either or both parameters $\alpha$ and $\beta$ assume one of the extreme values $0$ and $1,$ then we may or may not have a unique invariant measure.

Indeed, if only $\alpha = 0,$ with $0 < \beta \leq 1,$ then $p=1$ and there is only one stationary distribution, concentrated at state $x=1.$ If only $\beta = 0,$ with $0 < \alpha \leq 1,$ then $p=0$ and there is, again, only one stationary distribution, which this time is concentrated at state $x=2.$

If both $\alpha = \beta = 0,$ then $0\leq p \leq 1$ is arbitrary, and there are an infinite number of stationary distributions, one concentrated at $x=1,$ another concentrated at $x=2,$ and many others as convex combinations of both. Effectivaly, in this case of both $\alpha=\beta=0,$ the Markov chain can be decoupled into two separate chains, one restricted to state $x=1$ and the other restricted to state $x=2.$ Each has its own unique invariant measure, with the whole system having any convex combination of both as invariant measure.

Finally, if both $\alpha = \beta = 1,$ then $p=1/2$ and we have again a unique invariant measure, with uniform probability in both states. The peculiarity of this example is that it has some periodic solutions in time. Indeed, if we start with $X_0 = 1$ or $X_0 = 2,$ then it oscillates between both states. They do not converge to the stationary distribution.

### Random walk running to infinity

In the finite-state case, we always have at least one invariant probability distribution. For a case with no invariant distribution, we need to go to an infinite-state case, either discrete or continuous. Here we consider an infinite but discrete example. Similary continuous-state examples can be easily constructed.

Consider a random walk $X_n = X_n + W_n,$ on $\mathcal{X} = \mathbb{N},$ where the $W_n$ are i.i.d. Bernoulli-type random variables with equal probabilities of being $0$ or $+1.$ This means the one-step transition probability is
```math
    K(i, j) = \frac{1}{2}\delta_{i, j} + \frac{1}{2}\delta_{i,j-1},
```
where $\delta_{i,j}$ is the Kroenecker delta, equal to $1,$ when $i=j,$ and to $0,$ otherwise. Thus any sample path has $1/2$ probability of staying at $j=1$ and $1/2$ probability to move from $i$ up to $j=i+1.$ In (infinite-)matrix form, it has both the diagonal and the superdiagonal with entries $1/2$ and all the other entries equal to zero.

If there were an equilibrium probability distribution $p = (p_1, p_2, \cdots),$ where $p_n \geq 0$ and $\sum_{n\in\mathbb{N}} p_n = 1,$ we would have the equations
```math
    p_n = \frac{p_n + p_{n+1}}{2}, \quad n\in\mathbb{Z}.
```
This means
```math
    p_{n+1} = p_n = p_1,
```
so that
```math
    \sum_{n\in\mathbb{N}} p_n = \sum_{n\in\mathbb{N}} p_1 = \infty,
```
which makes it impossible to be a probability distribution, with $\sum_{n\in\mathbb{N}} p_n = 1.$ This means the chain admits no invariant probability distribution. The problem here is the fact that we ask the probability to be finite. In fact, the counting measure, or any multiple of it, is an invariant measure, which is not finite. (See the next example, which is similar).

### A symmetric random walk on the integers

Even if we are allowed to move up or down, we may not have an invariant probability distribution. Consider, for instance, the random walk $X_n = X_n + W_n,$ on $\mathcal{X} = \mathbb{Z},$ where the $W_n$ are i.i.d. Bernoulli-type random variables with equal probabilities of being $+1$ or $-1.$ This means the one-step transition probability is
```math
    K(i, j) = \frac{1}{2}\delta_{i, j-1} + \frac{1}{2}\delta_{i,j+1},
```
where $\delta_{i,j}$ is the Kroenecker delta, equal to $1,$ when $i=j,$ and to $0,$ otherwise, so it has $1/2$ probability to move from $i$ to $j=i+1$ and $1/2$ probability to move from $i$ to $j=i-1.$ In (infinite-)matrix form, it has both the subdiagonal and superdiagonal with entries $1/2$ and all the other entries equal to zero.

If there were an equilibrium probability distribution $p = (\cdots, p_{-2}, p_{-1}, p_0, p_1, p_2, \cdots),$ where $p_n \geq 0$ and $\sum_{n\in\mathbb{Z}} p_n = 1,$ we would have the equations
```math
    p_n = \frac{p_{n-1} + p_{n+1}}{2}, \quad n\in\mathbb{Z}.
```
But this equation contradicts the conditions $p_n \geq 0$ and $\sum_n p_n = 1.$ Indeed, the finite summation implies that $\liminf_{|n|\rightarrow \infty} p_n = 0.$ If some $p_n > 0,$ then we would have $n_- < n$ and $n_+ > n$ such that $p_{n_-}, p_{n_+} < p_n.$ This means that the maximum of $p_j$ for $n_- \leq j \leq n_+$ occurs somewhere within the extreme values and also that there must be a point $m$ with $p_{m-1}, p_{m+1} \leq p_m$ and $\min\{p_{m-1}, p_{m+1}\} < p_m$. In this case,
```math
    p_m > \frac{p_{m-1} + p_{m+1}}{2},
```
contradicting the inequality above for the stationary distribution. Thus, one cannot have an invariant probability distribution for this Markov chain.

If we look for an invariant measure which is not necessarily finite, then we find out that the *counting distribution* is invariant. In fact, the counting distribution can be written as
```math
    P(E) = \sum_{n\in \mathbb{Z}} \delta_n(E),
```
where $\delta_n$ is the Dirac distribution at $n,$ i.e. $\delta_n(E) = 1,$ if $n\in E, and $= 0,$ otherwise. Then,
```math
    \begin{align*}
        (PK)(E) & = \sum_{n\in \mathbb{Z}} \frac{1}{2}\left(\delta_{n-1}(E) + \delta_{n+1}(E)\right) = \frac{1}{2}\sum_{n\in \mathbb{N}} \delta_{n-1}(E) + \frac{1}{2}\sum_{n\in\mathbb{N}}\delta_n(E) \\
        & = \frac{1}{2}\sum_{i\in \mathbb{N}} \delta_{i}(E) + \frac{1}{2}\sum_{i\in\mathbb{N}}\delta_i(E) = \frac{1}{2}P(E) + \frac{1}{2}P(E) = P(E),
    \end{align*}
```
showing that $P$ is invariant (or any constant multiple of $P$).