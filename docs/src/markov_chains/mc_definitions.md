# Discrete-time Markov chains

Many important probabilistic models fit the framework of a Markov chain, including the Markov Chain Monte Carlo (MCMC) methods, as the name says it. Here we explore some of its concepts and properties. Markov chains can be indexed with either discrete or continous "time" variables, but here we only consider the discrete-time case.

## Definition

Markov chains are families of random variables $(X_n)_{n\in I},$ over an index set $I,$ such that, essentially, only the most recent known state determines the future of the chain. The index set $I$ can be continuous or discrete, but here we are only interested on the discrete case $I = \mathbb{Z}_{\geq 0} = \{0, 1, 2, \ldots, \}.$ The index set is usually referred to as the *time* variable, while the values of the random variables live on a space $\mathcal{X}$ called the *state* or *event* space.

The event space $\mathcal{X}$ can be either countable (finite or infinite, with the discrete topology) or continuous (e.g. $\mathcal{X} = \mathbb{R}^d,$ $d\in\mathbb{N},$ or some infinite dimensional Hilbert or Banach space). We are mostly interested in the continuous case $\mathcal{X} = \mathbb{R}^d,$ $d\in\mathbb{N},$ but some examples are given with $\mathcal{X} = \{1, \ldots, n\},$ $n\in\mathbb{N},$ or $\mathcal{X}=\mathbb{Z},$ for illustrative purposes and intuitive assessment. In any case, we always assume it is a topological space.

The random variables are functions $X_n:\Omega \rightarrow \mathcal{X}$ which are assumed to be measurable from a probability space $(\Omega, \mathcal{F}, \mathbb{P}),$ with $\sigma$-algebra $\mathcal{F}$ and probability distribution $\mathbb{P},$ to a measurable space, which we take here to be $(\mathcal{X}, \mathcal{B}(\mathcal{X})),$ where $\mathcal{B}(\mathcal{X})$ denotes the Borel $\sigma$-algebra of the topological space $\mathcal{X}.$

In the discrete-time case, the Markov property for the discrete-time process $(X_n)_n$ to be a Markov chain can be written as
```math
    \mathbb{P}(X_{n+1}|X_0\in E_0, X_1 \in E_1, \ldots, X_n\in E_n) = \mathbb{P}(X_{n+1}|X_n\in E_n),
```
for all Borel sets $E_0, \ldots, E_n.$ The continuos-time version has a similar condition, based on the notion of filtration of a $\sigma$-algebra, but we do not need to worry about this in the time-discrete case.

## Transition probabilities

Markov chains can be described by transition probabilities. We will define them in both non-homogeneous and homogeneous cases in time, but after that we will only consider the homogeneous case.

### The non-homogeneous time-discrete case

A Markov chain can be fully characterized by the **transition probabilities**
```math
    K_{n, m}(E, F) = \mathbb{P}(X_m\in F|X_n\in E),
```
for any pair $E, F\subset \mathcal{X}$ of Borel sets, where $n, m = 0, 1, \ldots.$ In particular, we denote
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

## The Markov operator

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
    \begin{align*}
        \left(K_n(i, j)\right)_{i,j} & = \left(K_{0,n}(i, j)\right)_{i,j} = \left(K_{n-1, n}(i, j)\right)_{i,j}\left(K_{n-2, n-1}(i, j)\right)_{i,j}\cdots \left(K_{0, 1}(i, j)\right)_{i,j} \\
        & = \left(K(i, j)\right)_{i,j}^n = K^n.
    \end{align*}
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

