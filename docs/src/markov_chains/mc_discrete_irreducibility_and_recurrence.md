# Irreducibility in the countable-space case

Irreducibility is a fundamental concept related to the uniqueness of an invariant measure. We explore such concept here.

## Setting

As before, we assume that $(X_n)_n$ is a time-homogeneous, discrete-time Markov chain with a countable state space. More precisely, we assume the indices are $n=0, 1, 2, \ldots,$ and that the space $\mathcal{X}$ is finite or countably infinite. The sample space is the probability space $(\Omega, \mathcal{F}, \mathbb{P}),$ where $\mathcal{F}$ is the $\sigma$-algebra on the set $\Omega$ and $\mathbb{P}$ is the probability distribution. The one-step transition distribution is denoted by $K(x, y) = \mathbb{P}(X_{n+1} = y | X_n = x),$ and is independent of $n=0, 1, \ldots,$ thanks to the time-homogeneous assumption. Similary, the $n$-step transition distribution is denoted $K_n(x, y) = \mathbb{P}(X_{k+n} = y | X_k = x),$ for $n=1, 2, \ldots,$ independently of $k=0, 1, \ldots.$

## Definitions

We start with some fundamental definitions.

### Connected points

Markov chains are about the probability of states changing with time. If starting at some state, some of the other states might be more likely to be observed in the future than others, and some might never be observed. We distinguish them by the notion of connectedness.

!!! note "Definition (connected points)"
    We say that **$x$ is connected to $y$** when there exists $n=n(x, y)\in\mathbb{N}$ such that $K_n(x, y) > 0.$ When $x$ is connected to $y$, we write $x \rightarrow y.$ When $x$ is connected to $y$ and $y$ is connected to $x,$ we write $x \leftrightarrow y.$

### Characterization of connection in terms of the first return time and the number of visits

Connection can be characterized with respect to other random variables.
!!! note "Fact"
    A point $x$ is connected to $y$ if, and only if, there is a positive probability of reaching $y$ from $x$ in a finite number of steps, which can be written as
    ```math
        x \rightarrow y \quad \Longleftrightarrow \quad \mathbb{P}(\tau_y < \infty | X_0 = x) > 0.
    ```

When $x$ is connected to $y,$ there is a positive probability that there is at least one passage from $x$ to $y,$ i.e. $\eta_y \geq 1$ is greater than one, with positive probability. Thus, we have the following equivalences.

!!! note "Fact"
    ```math
        x \rightarrow y \quad \Longleftrightarrow \quad \mathbb{P}(\eta_y \geq 1 | X_0 = x) > 0,
    ```
    and
    ```math
        (X_n)_n \textrm{ is recurrent } \quad \Longleftrightarrow \quad \mathbb{P}(\eta_y \geq 1 | X_0 = x) > 0, \quad \forall x, y\in\mathcal{X}.
    ```

### Irreducibility

!!! note "Definition (Irreducibility)"
    The chain is called **irreducible** when
    ```math
        \forall x, y\in\mathcal{X}, \;\exists n=n(x, y)\in\mathbb{N}, \; K_n(x, y) > 0.
    ```
    It is called **strongly irreducible** when $n(x, y) = 1$ for all $x, y\in\mathcal{X}.$

Irreducibility means that there is a positive probability to go from any one state to any other state, or back to itself, after a finite number of steps. Strong irreducibility means that with positive probability the states are reached after a single time step.

For example, the random walk on the integers $\mathcal{X}=\mathbb{Z}$ defined by $X_{n+1} = X_n + B_n,$ where $B_n$ are Bernoulli i.i.d. with states $+1$ with probability $p$ and $-1$ with probability $1 - p,$ where $0 < p < 1,$ is irreducible, since from any given state $x,$ any other state $y\neq x$ can be reached, with positive probability, in $n = |y - x|$ steps, while if $y = x,$ then two steps are needed.

If the jump is twice larger, i.e. $X_{n+1} = X_n + 2B_n,$, then only states $y$ of same parity as $x$ are reached with positive probability, so this chain is not irreducible.

Thus, *the chain is recurrent if, and only if, every point is connected to every other point in space.*

Then, we can write that
```math
    (X_n)_n \textrm{ is recurrent } \quad \Longleftrightarrow \quad x \rightarrow y, \;\forall x, y\in\mathcal{X} \quad \Longleftrightarrow \quad x \leftrightarrow y, \;\forall x, y\in\mathcal{X}.
```

We also have

!!! note "Fact"
    ```math
        (X_n)_n \textrm{ is irreducible } \quad \Longleftrightarrow \quad \mathbb{P}(\tau_y < \infty | X_0 = x) > 0, \quad \forall x, y\in\mathcal{X}.
    ```


### Recurrent chain

!!! note "Definition (recurrent chain)"
    The Markov chain is called **recurrent** when every state is recurrent, i.e.
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1, \quad \forall x\in\mathcal{X}.
    ```

For instance, we have seen that the random walk $X_{n+1} = X_n + B_n,$ where $B_n$ are Bernoulli i.i.d. with states $+1$ with probability $p$ and $-1$ with probability $1 - p,$ where $0 < p < 1,$ but it is not recurrent. For any $x, \in\mathcal{X}=\mathbb{Z}$ and $n\in\mathbb{N},$ the transition probability $K_n(x, y)$ is zero if $|y - x|$ and $n$ have different parity or when $n < |y - x|,$ and is given by the binomial distribution
```math
    p_n(m) = \begin{pmatrix} n \\ i \end{pmatrix} p^i(1-p)^j,
```
with
```math
    i = \frac{n + m}{2}, \quad j = n - i = \frac{n - m}{2}, \quad m = y - x,
```
when
```math
    |m| \leq n, \quad m, n \textrm{ with same parity}
```
Thus, we can write
```math
    K_n(x, y) = \begin{cases}
        \begin{pmatrix} n \\ \frac{n + y - x}{2} \end{pmatrix} p^{(n+y-x)/2}(1-p)^{(n + x - y)/2}, & |x - y| \leq n, \; x - y, n \textrm{ same parity}, \\
        0, & \textrm{otherwise}.
    \end{cases}
```
In particular, the probability of returning to $x$ is
```math
    K_n(x, x) = \begin{cases}
        \begin{pmatrix} n \\ \frac{n}{2} \end{pmatrix} p^{n/2}(1-p)^{n/2}, & n \textrm{ is even}, \\
        0, & \textrm{otherwise}.
    \end{cases}
```
We have
```math
    \{X_n = x \textrm{ infinitely often} | X_0 = x\} = \bigcap_{n \in\mathbb{N}}\bigcup_{m \geq n, m\in \mathbb{N}} \{X_m = x | X_0 = x\}.
```
Thus, by the Borel-Cantelli Lemma,
```math
    \mathbb{P}\left(X_n = x \textrm{ infinitely often} | X_0 = x\right) = 0,
```
since
```math
    \sum_{n\in\mathbb{N}} \mathbb{P}\left(X_m = x | X_0 = x\right) = \sum_{n\in\mathbb{N}, n \textrm{ even}} \begin{pmatrix} n \\ \frac{n}{2} \end{pmatrix} p^{n/2}(1-p)^{n/2} = \sum_{n\in\mathbb{N}} \begin{pmatrix} 2n \\ n \end{pmatrix} p^n(1-p)^n \leq \sum_{n\in\mathbb{N}} \frac{(2n)!}{2(n!)} (1/2)^n
```

## Uniqueness of the invariant distribution

## Kac's Theorem on invariant distribution of an irreducible chain

Kac's Theorem says that if $(X_n)_n$ is irreducible and it has a stationary distribution $P,$ then
```math
    P(x) = \frac{1}{\mathbb{E}[\tau_x | X_n = x]}, \quad \forall x\in\mathcal{X}.
```

The idea is that, $m_x = \mathbb{E}[\tau_x | X_n = x]$ is the average time that the chain takes to start from $x$ and come back to $x.$ Every state $x$ is visited repeatedly, with an average time of $m_x$ between consecutive visits.

## Irreducibility in the discrete-space case

In the discrete-space case, $P$-irreducibility can be stated with point sets, so that the Markov chain is $P$-irreducible if, and only if,
```math
    x, y\in\mathcal{X},\; P(y) > 0 \Longrightarrow \exists n=n(x, y)\in\mathbb{N}, \; K_n(x, y) > 0.
```

### Positivity of an invariant probability distribution

If a discrete-space Markov chain has an invariant probability distribution $P$ and it is $P$-irreducible, then $P$ must be everywhere strictly positive, i.e. $P(x) > 0,$ for all $x\in \mathcal{X}.$

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



## References

1. [C. P. Robert, G. Casella (2004), "Monte Carlo Statistical Methods", Second Edition, Springer Texts in Statistics, Springer New York, NY](https://doi.org/10.1007/978-1-4757-4145-2)
1. [G. F. Lawler(2006), "Introduction to Stochastic Processes", 2nd Edition. Chapman and Hall/CRC, New York.](https://doi.org/10.1201/9781315273600)
