# Irreducibility and recurrence in the discrete-space case

Irreducibility and recurrence are fundamental concepts related to the existence and uniqueness of an invariant measure.

## Definitions

We start with some fundamental definitions.

### Setting

Here, we always assume that $(X_n)_n$ is a time-homogeneous, discrete-time Markov chain with a discrete state space. We assume the indices are $n=0, 1, 2, \ldots,$ and by discrete space we mean specifically that $\mathcal{X}$ is a countable space with the discrete topology, being either finite or infinite. The one-step transition distribution is denoted by $K(x, y) = \mathbb{P}(X_{n+1} = y | X_n = x),$ independently of $n=0, 1, \ldots,$ while the $n$-step is denoted $K_n(x, y) = \mathbb{P}(X_{k+n} = y | X_k = x),$ for $n=1, 2, \ldots,$ independently of $k=0, 1, \ldots.$

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

There are several concepts related to irreducibility.

### Connected points

!!! note "Definition (connected points)"
    We say that **$x$ is connected to $y$** when there exists $n=n(x, y)\in\mathbb{N}$ such that $K_n(x, y) > 0.$ Thus, *the chain is recurrent if, and only if, every point is connected to every other point in space.*

    When $x$ is connected to $y$, we write $x \rightarrow y.$ When $x$ is connected to $y$ and $y$ is connected to $x,$ we write $x \leftrightarrow y.$

Then, we can write that
```math
    (X_n)_n \textrm{ is recurrent } \quad \Longleftrightarrow \quad x \rightarrow y, \;\forall x, y\in\mathcal{X} \quad \Longleftrightarrow \quad x \leftrightarrow y, \;\forall x, y\in\mathcal{X}.
```

### Return time

!!! note "Definition (return time)"
    Given a point $x\in\mathcal{X},$ we define the **return time** to $x$ by
    ```math
        \tau_x = \inf\left\{n\in\mathbb{N}\cup\{+\infty\}; \; n = \infty \textrm{ or } X_n = x\right\}.
    ```

The quantity
```math
    \mathbb{P}(\tau_x < \infty)
```
is the *probability of reaching $x$ in a finite number of steps,* while
```math
    \mathbb{P}(\tau_y < \infty | X_0 = x)
```
is the *probability of reaching $y$ from $x$ in a finite number of steps,* and 
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x)
```
is the *probability of returning to $x$ in a finite number of steps.**

!!! note "Fact"
    A point $x$ is connected to $y$ if, and only if, there is a positive probability of reaching $y$ from $x$ in a finite number of steps, which can be written as
    ```math
        x \rightarrow y \quad \Longleftrightarrow \quad \mathbb{P}(\tau_y < \infty | X_0 = x) > 0.
    ```

Combining these characterizations, we have that

!!! note "Fact"
    ```math
        (X_n)_n \textrm{ is recurrent } \quad \Longleftrightarrow \quad \mathbb{P}(\tau_y < \infty | X_0 = x) > 0, \quad \forall x, y\in\mathcal{X}.
    ```

### Number of passages

Another useful quantity is the random variable denoting the *number of passages* through a given state $x\in\mathcal{X},$

!!! note "Definition (number of passages)"
    The **number of passages** through a given state $x\in\mathcal{X}$ is defined by
    ```math
        \eta_x = \sum_{n=1}^\infty \mathbb{1}_{\{X_n = x\}}.
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

### Recurrence and Transience

When the chain is irreducible, every state is attainable from anywhere, including the state itself, so that, due to the time-homogenous assumption, every state is revisited infinitely often. But that doesn't say how significant such visits are, in a probabilistic way. When this happens with probability one, we say the state is recurrent, otherwise we say it is transient.

More precisely, we have the following definition.

!!! note "Definition (recurrent state)"
    a state $x$ is called **recurrent** when
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1.
    ```
    Otherwise, the state $x$ is called **transient,** i.e. if
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) < 1.
    ```

But it turns out that when this probability is smaller than 1, its is actually zero, as we will see at some point.

When every state is recurrent we say that the chain is recurrent.

!!! note "Definition (recurrent chain)"
    The Markov chain is called **recurrent** when every state is recurrent, i.e.
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1, \quad \forall x\in\mathcal{X}.
    ```

Equivalent definitions can be made with the notion of return time.
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x) = 1,
```
for every $x\in\mathcal{X}.$ This means that, starting from any $x\in\mathcal{X},$ the chain almost surely returns to $x$ in finite time. (But this is equivalent to recurrence only in the countable case; see page 222 of [Robert and Casella (2004)](https://doi.org/10.1007/978-1-4757-4145-2).)

When it keeps coming back infinitely often, counting such times adds up to infinity. When this happens with probability one, then the expectation of such counts is also infinite. In other words, we have the following equivalence.

!!! note "Fact"
    A state $x$ is recurrent if, and only if,
    ```math
        \mathbb{E}\left[\eta_x\right | X_0 = x] = \infty.
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

## Existence of invariant distribution

!!! note "Theorem"
    Suppose that, for some $x\in\mathcal{X},$
    ```math
        \mathbb{E}[\tau_{x} | X_0 = x] < \infty.
    ```
    Then
    ```math
        P_x(z) = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=1}^\infty \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x)
    ```
    defines an invariant probability distribution for the Markov chain.
    
The proof below is adapted from Theorem 6.37 of [Robert and Casella (2004)](https://doi.org/10.1007/978-1-4757-4145-2). Notice we are not assuming irreducibility, so we are not claiming that this invariant distribution is unique.

!!! note "Proof"
    For the proof, we consider, for the sake of simplicity, the unnormalized measure
    ```math
        {\tilde P}_x(z) = \sum_{n=1}^\infty \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x).
    ```
    We show that it defines a positive and finite invariant measure, with ${\tilde P}_x(\mathcal{X}) = \mathbb{E}[\tau_{x} | X_0 = x].$ After that, we obtain the desired result by normalizing ${\tilde P}_x$ by the expectation $\mathbb{E}[\tau_{x} | X_0 = x].$

    First of all, since the space is discrete and each ${\tilde P}_x(z) \geq 0,$ for $z\in\mathcal{X},$ it follows that ${\tilde P}_x$ defines indeed a measure on $\mathcal{X}.$ Let us check that ${\tilde P}_x$ is in fact a nontrivial and finite measure. We have
    ```math
        \begin{align*}
            {\tilde P}_x(\mathcal{X}) & = \sum_{z\in\mathcal{X}} P_x(z) \\
            & = \sum_{z\in\mathcal{X}}\sum_{n=1}^\infty \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x) \\
            & = \sum_{n=1}^\infty \sum_{z\in\mathcal{X}} \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x) \\
            & = \sum_{n=1}^\infty \mathbb{P}( \tau_{x} \geq n | X_0 = x) \\
            & = \sum_{n=1}^\infty \sum_{m=n}^\infty \mathbb{P}(\tau_{x} = m | X_0 = x) \\
            & = \sum_{m=1}^\infty \sum_{n=1}^{m} \mathbb{P}(\tau_{x} = m | X_0 = x) \\
            & = \sum_{m=1}^\infty m \mathbb{P}(\tau_{x} = m | X_0 = x) \\
            & = \mathbb{E}[\tau_{x} | X_0 = x]
        \end{align*}
    ```
    Since it is assumed that 
    ```math
        \mathbb{E}[\tau_{x} | X_0 = x] < \infty,
    ```
    it follows that the measure ${\tilde P}$ is finite. And since, by definition, $\tau_x$ is an integer with $1 \leq \tau_x \leq +\infty,$ the measure is non-trivial, i.e. it is positive, with
    ```math
        1 \leq {\tilde P}(\mathcal{X}) < \infty.
    ```

    In particular,
    ```math
        \begin{align*}
            {\tilde P}(\mathcal{x}) & = \sum_{n=1}^\infty \mathbb{P}(X_n = x, \tau_{x} \geq n | X_0 = x) \\
            & = \sum_{n=1}^\infty \mathbb{P}(X_n = x, \tau_{x} = n | X_0 = x) \\
            & = \sum_{n=1}^\infty \mathbb{P}(\tau_{x} \geq n | X_0 = x) \\
            & = \mathbb{P}(\cup_{n\in\mathbb{N}} \{\tau_{x} = n | X_0 = x\}) \\
            & = \mathbb{P}(\tau_{x} < \infty | X_0 = x).
        \end{align*}
    ```
    The condition
    ```math
        \mathbb{E}[\tau_{x} | X_0 = x] < \infty.
    ```
    implies that $\tau_{x}$ must be finite almost surely, so that
    ```math
        {\tilde P}(\mathcal{x}) = \mathbb{P}(\tau_{x} < \infty | X_0 = x) = 1.
    ```

    Now, let us check that ${\tilde P}_x$ is invariant.
    We need to show that
    ```math
        \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) = {\tilde P}_x(z),
    ```
    for every $z\in\mathcal{X}.$ For that, we write
    ```math
        \begin{align*}
            \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \sum_{y\in\mathcal{X}} K(y, z)\sum_{n=1}^\infty \mathbb{P}(X_n = y, \tau_{x} \geq n | X_0 = x) \\
            & = \sum_{n=1}^\infty \sum_{y\in\mathcal{X}} K(y, z) \mathbb{P}(X_n = y, \tau_{x} \geq n | X_0 = x)
        \end{align*}
    ```
    We split the sum according to $y=x$ and $y\neq x,$ so that
    ```math
        \begin{align*}
            \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \sum_{n=1}^\infty K(x, z) \mathbb{P}(X_n = x, \tau_{x} \geq n | X_0 = x) \\
            & \qquad + \sum_{n=1}^\infty \sum_{y\neq x} K(y, z) \mathbb{P}(X_n = y, \tau_{x} \geq n | X_0 = x) \\
            & = K(x, z) {\tilde P}(x) + \sum_{n=1}^\infty \sum_{y\neq x} \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n | X_0 = x) \\
        \end{align*}
    ```
    For $y\neq z,$ we have
    ```math
        \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n | X_0 = x) = \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n+1 | X_0 = x).
    ```
    Thus,
    ```math
        \begin{align*}
            \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = K(x, z) {\tilde P}(x) + \sum_{n=1}^\infty \sum_{y\neq x} \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n+1 | X_0 = x) \\
            & = K(x, z) {\tilde P}(x) + \sum_{n=1}^\infty \mathbb{P}(X_{n+1} = z, \tau_{x} \geq n+1 | X_0 = x) \\
            & = K(x, z) {\tilde P}(x) + \sum_{n=2}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x),
        \end{align*}
    ```
    where in the last step we just reindexed the summation. Now we use that ${\tilde P}(x) = 1$ (as proved above) and that 
    ```math
        K(x, z) = \mathbb{P}(X_1 = z | X_0 = x) = \mathbb{P}(X_1 = z, \tau_{x} \geq 1 | X_0 = x)
    ```
    (since $\tau_{x} \geq 1$ always), to obtain
    ```math
        \begin{align*}
            \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \mathbb{P}(X_1 = z, \tau_{x} \geq 1 | X_0 = x) + \sum_{n=2}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x) \\
            & = \sum_{n=1}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x) \\
            & = {\tilde P}(z),
        \end{align*}
    ```
    proving the invariance.

    Now, as mentioned above, since ${\tilde P}(\mathcal{X}) = \mathbb{E}[\tau_{x} | X_0 = x]$ is finite and positive, we can normalize ${\tilde P}$ by this expectation to obtain the invariant probability distribution
    ```math
        P(z) = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=1}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x).
    ```

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