# Markov chains

Many important probabilistic models fit the framework of a Markov chain, including the Markov Chain Monte Carlo (MCMC) methods. Here we explore some of its concepts and properties.

## Discrete-time Markov chains

Markov chains are families of random variables $(X_n)_{n\in N},$ over an index set $N,$ such that, essentially, only the most recent known state determines the future of the chain. The index set $I$ can be continuous or discrete, but here we are only interested on the discrete case $\mathbb{Z}_{\geq 0} = \{0, 1, 2, \ldots, \},$ or sometimes just $\mathbb{N}=\{1, 2, \ldots, \}.$ The index set is usually referred to as the *time* variable, while the values of the random variables live on a space $\mathcal{X}$ called the *state* or *event* space.

The event space $\mathcal{X}$ can be either discrete (finite or infinite) or continuous. We are mostly interested in the case $\mathcal{X} = \mathbb{R}^d,$ $d\in\mathbb{N},$ but some examples are given with $\mathcal{X} = \{1, \ldots, n\},$ $\mathcal{X}=\mathbb{Z},$ for illustrative purposes. In any case, we always assume it is a topological space

The random variables are functions $X_n:\Omega \rightarrow \mathcal{X}$ which are assumed to be measurable from a probability space $(\Omega, \mathcal{F}, \mathbb{P}),$ with $\sigma$-algebra $\mathcal{F}$ and probability distribution $\mathbb{P},$ to a measurable space, which we take here to be $(\mathcal{X}, \mathcal{B}(\mathcal{X})),$ where $\mathcal{B}(\mathcal{X})$ denotes the Borel $\sigma$-algebra of the topological space $\mathcal{X}.$

In the discrete-time case, the Markov property for the discrete-time process $(X_n)_n$ to be a Markov chain can be written as
```math
    \mathbb{P}(X_{n+1}|X_0\in E_0, X_1 \in E_1, \ldots, X_n\in E_n) = \mathbb{P}(X_{n+1}|X_n\in E_n),
```
for all Borel sets $E_0, \ldots, E_n.$

## Transition probabilities

Markov chains can be described by the transition probabilities.

### The non-homogeneous time-discrete case

A Markov chain can be fully characterized by the **transition probabilities**
```math
    K_{n, m}(E, F) = \mathbb{P}(X_m\in F|X_n\in E),
```
for any pair  $E, F\subset \mathcal{X}$ of Borel sets, where $n, m = 0, 1, \ldots.$ In particular, we denote
```math
    K_{n, m}(x, F) = \mathbb{P}(X_{n+1}\in F|X_n = x),
```
for any Borel set $E\subset \mathcal{X}$ and any point $x\in\mathcal{X}.$ Notice that 

* The set map $K_{n, m}(x, \cdot)$ is a probability measure on $\mathcal{X},$ for each $x\in\mathcal{X}.$
* The map $K_{n, m}(\cdot, F)$ is a measurable function from $\mathcal{X}$ to $[0, 1],$ for each Borel set $F\subset \mathcal{X}.$

When $\mathcal{X} = \mathbb{R}^d,$ $d\in \mathbb{N},$ we say that the transition probability $K_{n, m}$ has a **transition kernel** or **transition density** $k_{n, m} = k_{n, m}(x, y)$ when $K_{n, m}(x, \cdot)$ is absolutely continous with respect to the Lebesgue measure, so that
```math
    K_{n, m}(x, F) = \int_F k_{n, m}(x, y) \;\mathrm{d}y,
```
and
```math
    K_{n, m}(E, F) = \int_E \int_F k_{n, m}(x, y) \;\mathrm{d}y\;\mathrm{d}x.
```
Since $K_{n, m}(x, \cdot)$ is a probability distribution, we have
```math
    \int_{\mathcal{X}} k_{n,m}(x, y) \;\mathrm{d}y = K_n(x, \mathcal{X}) = 1.
```
But $K_{n,m}(E, F)$ is contined on $X_n\in E,$ so $K_{n, m}(\cdot, y)$ need not be a probability distribution.

### The time-homogeneous case

Very often, as in most examples we are interested in, here, the transition probabilities only depend on the "time difference" $m-n,$ i.e. it is *time homogeneous,* or *autonomous.* More precisely, the Markov chain is called **time homogeneous** when
```math
    K_{k, k + n}(E, F) = K_{0, n}(E, F), \quad \forall n, k = 0, 1, 2, \ldots, \;\forall E, F\in\mathcal{B}(\mathcal{X}).
```
In this case, we define the **$n$th-step transition probability** by
```math
    K_n(E, F) = K_{0, n}(E, F) = K_{k, k+n}(E, F),
```
The corresponding density $k_n(x, y)$ is called the **$n$th-step transition density.**

The one-step transition probability and density are simply denoted without any subscript,
```math
    \begin{align*}
        K(E, F) & = K_1(E, F) = \mathbb{P}(X_{n+1} \in F | X_n\in E), \\
        K(x, F) & = K_1(x, F) = \mathbb{P}(X_{n+1} \in F | X_n = x), \\
        k(x, y) = k_1(x, y).
    \end{align*}
```

### The evolution of distributions

We denote by $P_n$ the probability distribution of the random variable $X_n,$ i.e.
```math
    P_n(E) = \mathbb{P}(X_n\in E), \quad \forall E\in\mathcal{B}(\mathcal{X}).
```
Due to the nature of a Markov process, the distribution of $P_{n+1}$ only depends on the distribution of $P_n$ and on the transition distribution. More generally, $P_{k+n}$ depends on $P_{k}$ and, in particular, the distribution $P_n$ of $X_n$ depends on the distribution $P_0$ of $X_0.$

More precisely, we can express this dependence via the Chapman-Kolmogorov equation, which, in the case the transition probabilities admit a density, reads
```math
    K_{m+n}(x, F) = \int_{\mathcal{X}} K_n(y, F)k_m(x, y)\;\mathrm{d}y,
```
for any Borel set $F\subset\mathcal{X}.$

### The Markov operator

Given the one-step transition probability $K(\cdot, \cdot),$ in the time-homogeneous case, on defines the associated **Markov operator** $K:\mathcal{P}(\mathcal{X})\rightarrow \mathcal{P}(\mathcal{X})$ that takes a distribution $P\in \mathcal{P}(\mathcal{X})$ on $\mathcal{X}$ to the "next step" distribution $PK\in\mathcal{P}(\mathcal{X}),$ defined by
```math
    (PK)(E) = \int_{\mathcal{X}} K(x, E);\mathrm{d}P(x).
```
In particular, if $P_n$ is the distribution of $X_n,$ then the distribution of $X_{n+1}$ is $KP_n,$ i.e.
```math
    P_{n+1} = P_nK, \quad \textrm{for } X_j \sim P_j.
```

**Rmk:** Another common notation in the integration is with $P(\mathrm{d}x)$ instead of $\mathrm{d}P(x).$ 

**Rmk:** We use the same $K$ to denote the Markov operator on $\mathcal{P}(\mathcal{X})$ and the one-step probability distribution $K(\cdot, \cdot)$ on $\mathcal{X}\times\mathcal{X}.$

### Finite-state case

When the state is finite, say $\mathcal{X} = \{1, 2, \ldots, n\},$ the transition operator can be characterized by the transitions $K(i, j)$ of going from state $i$ to state $j$. This defines a matrix,
```math
    K = \left(K_1(i, j)\right)_{i,j=1}^n = \begin{bmatrix}
        k_{1,1} & k_{1,2} & \cdots & k_{1,n} \\
        k_{2,1} & k_{2,2} & \ddots & \vdots \\
        \vdots & \ddots & \ddots & k_{n-1,n} \\
        k_{n,1} & \cdots & k_{n,n-1} & k_{n,n}
    \end{bmatrix},
```
where $k_{i, j} = K(i, j).$ Each row sums up to one:
```math
    \sum_{j=1}^n k_{i,j} = \sum_{j=1}^n K_1(i, j) = K_1(i, \mathcal{X}) = 1.
```

The $n$-step transition operator is obtained by simply composing the $1$-step transition operator,
```math
    \left(K_n(i, j)\right)_{i,j} = \left(K_{0,n}(i, j)\right)_{i,j} = \left(K_{n-1, n}(i, j)\right)_{i,j}\left(K_{n-2, n-1}(i, j)\right)_{i,j}\cdots \left(K_{0, 1}(i, j)\right)_{i,j} = \left(K_1(i, j)\right)_{i,j}^n = K^n.
```

If we denote the distribution of a given state $X_n$ by a row vector
```math
    P_n = [p_1, \ldots, p_n], \quad p_i = \mathbb{P}(X_n = i),
```
then
```math
    \begin{align*}
        P_{n+1} & = [\mathbb{P}(X_{n+1} = j)]_{j=1, \ldots, n} = \left[\sum_{i=1}^n \mathbb{P}(X_{n+1} = j | X_n = i) \mathbb{P}(X_n = 1)\right]_{j=1, \ldots, n} \\ 
        & = \left[\sum_{i=1}^n K_{i, j} p_i\right]_{j=1, \ldots, n} = \begin{bmatrix} p_1 & p_2 & \cdots & p_n\end{bmatrix} \begin{bmatrix}
            k_{1,1} & k_{1,2} & \cdots & k_{1,n} \\
            k_{2,1} & k_{2,2} & \ddots & \vdots \\
            \vdots & \ddots & \ddots & k_{n-1,n} \\
            k_{n,1} & \cdots & k_{n,n-1} & k_{n,n}
        \end{bmatrix} = P_n K,
    \end{align*}
```
where the product $P_n K$ is to be understood as the matrix product of a row vector $[p_1, \ldots, p_n]$ of the distribution of $X_n$ with the transition matrix $K,$ to yield another row vector with the distributions for the next state. Thus, $K$ is also a representation of the Markov operator.

## Invariant distribution

Of particular interest are the stationary distributions, i.e. such that $X_n = X_0,$ for all $n=0, 1, 2, \ldots.$ This is the same as saying that they all have the same distribution $P,$ which is invariant by the Markov transition operator.

### Definition

More precisely, a distribution $P$ on $\mathcal{X}$ is invariant when $PK = P,$ i.e.
```math
    P(E) = \int_{\mathcal{X}} K_1(x, E);\mathrm{d}P(x).
```

### Examples

#### A two-state Markov chain

Let $\mathcal{X} = \{1, 2\}$ and consider the Markov chain characterized by the transition distribution
```math
    K_1 = (K_1(i, j))_{i,j=1}^2 = \begin{bmatrix}
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

#### Random walk running to infinity

In the finite-state case, we always have at least one invariant probability distribution. For a case with no invariant distribution, we need to go to an infinite-state case, either discrete or continuous. Here we consider an infinite but discrete example. Similary continuous-state examples can be easily constructed.

Consider a random walk $X_n = X_n + W_n,$ on $\mathcal{X} = \mathbb{N},$ where the $W_n$ are i.i.d. Bernoulli-type random variables with equal probabilities of being $0$ or $+1.$ This means the one-step transition probability is
```math
    K_1(i, j) = \frac{1}{2}\delta_{i, j} + \frac{1}{2}\delta_{i,j-1},
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
which makes it impossible to be a probability distribution, with $\sum_{n\in\mathbb{N}} p_n = 1.$ This means the chain admits no invariant probability distribution.

#### A symmetric random walk on the integers

Even if we are allowed to move up or down, we may not have an invariant distribution. Consider, for instance, the random walk $X_n = X_n + W_n,$ on $\mathcal{X} = \mathbb{Z},$ where the $W_n$ are i.i.d. Bernoulli-type random variables with equal probabilities of being $+1$ or $-1.$ This means the one-step transition probability is
```math
    K_1(i, j) = \frac{1}{2}\delta_{i, j-1} + \frac{1}{2}\delta_{i,j+1},
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

## Stopping time and number of passages

Given a Borel set $E\subset \mathcal{X},$ the **stopping time** $\tau_E$ at $E$ of the Markov chain is defined as
```math
    \tau_E = \inf\{ n\in\mathbb{N}\cup\{+\infty\}; \; n = +\infty \textrm{ or } X_n\in E\}.
```
It should be clear that $\tau_E = \infty,$ if the chain never reaches $E,$ or $\tau_E = $ first time $n$ such that $X_n$ reaches $E.$ 

The quantity
```math
    P(\tau_E < \infty | X_0 = x)
```
is the *probability of return to $E$ in a finite number of steps.*

Another useful quantity is the **number of passages** in $E,$
```math
    \eta_E = \sum_{n=1}^\infty \mathbb{1}_{X_n \in A}.
```

Alternating chain example (e.g. $X_{n+1} = X_n \pm 2,$, so only even or odd integers are reached, so it is not irreducible). Continuous example (e.g. something like $X_{n+1} = [X_n] \pm [X_n] + 2 + Beta$)

## Recurrence and 

## Convergence notions

### Weak convergence

### Strong convergence

### Total variation norm

### Weiserstein distance

## Aperiodicity

