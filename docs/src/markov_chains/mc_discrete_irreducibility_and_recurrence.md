# Irreducibility and recurrence in the discrete-space case

Irreducibility and recurrence are fundamental concepts related to the uniqueness of an invariant measure, when it exists, but it does not necessarily implies that it exists.

## Definitions

Here, we always assume that $(X_n)_n$ is a time-homogeneous discrete time Markov chain with a discrete state space, with indices $n=0, 1, 2, \ldots,$ and with $\mathcal{X}$ countable, either finite or infinite.

### Irreducibility

In the discrete-state case, a time-homogeneous Markov chain $(X_n)_n$ with $n$-step transition probability $K_n(x, y),$ $x,y\in\mathcal{X},$ is called **irreducible** when
```math
    \forall x, y\in\mathcal{X}, \;\exists n=n(x, y)\in\mathbb{N}, \; K_n(x, y) > 0.
```

Irreducibility means that there is a positive probability to go from any one state to any other state, or back to itself, after a finite number of steps.

The Markov chain is called **strongly irreducible** when $n(x, y) = 1$ for all $x, y\in\mathcal{X}.$

There are several concepts related to irreducibility.

### Connected points

We say that **$x$ is connected to $y$** if, and only if, there exists $n=n(x, y)\in\mathbb{N}$ such that $K_n(x, y) > 0.$ Thus, *the chain is recurrent if, and only if, every point is connected to every other point in space.*

### Return time

Given a point $x\in\mathcal{X},$ we define the **return time** to $x$ to be
```math
    \tau_x = \inf\left\{n\in\mathcal{N}\cup\{+\infty\}; \; n = \infty or X_n = x\right\}.
```

The quantity
```math
    \mathbb{P}(\tau_x < \infty)
```
is the **probability of reaching $x$ in a finite number of steps,** while
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x)
```
is the **probability of returning to $x$ in a finite number of steps.** 

Then, *a point $x$ is connected to $y$ if, and only if, $\mathbb{P}(\tau_y < \infty | X_0 = x) > 0.$*

Combining the characterizations, we have that *a chain is recurrent if, and only if, $\mathbb{P}(\tau_y < \infty | X_0 = x) > 0,$ for any $x, y\in\mathcal{X}.$*

### Number of passages

Another useful quantity is the random variable for the **number of passages** in $x\in\mathcal{X},$
```math
    \eta_x = \sum_{n=1}^\infty \mathbb{1}_{\{X_n = x\}}.
```

### Recurrence

The Markov chain is called **recurrent** when
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x) = 1,
```
for every $x\in\mathcal{X}.$ This means that, starting from any $x\in\mathcal{X},$ the chain almost surely returns to $x$ in finite time.

Since we are assuming this is a time-homogenous Markov chain, once it comes back to $x,$ it keeps returning to $x$ again and again, with probability one. Thus, the definition above is equivalent to
```math
    \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1.
```

When it keeps coming back infinitely often, counting such times adds up to infinity. When this happens with probability one, then the expectation of such counts is also infinite. In other words, the above definitions of recurrence are also equivalent to assuming that
```math
    \mathbb{E}\left[\eta_x\right | X_0 = x] = \infty.
```

### Example 

Alternating chain example (e.g. $X_{n+1} = X_n \pm 2,$, so only even or odd integers are reached, so it is not irreducible). Continuous example (e.g. something like $X_{n+1} = [X_n] \pm [X_n] + 2 + Beta$)

## Existence of invariant distribution

Suppose that, for some $x\in\mathcal{X},$
```math
    \mathbb{E}[\tau_{x} | X_0 = x] < \infty.
```
Then
```math
    P_x(z) = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=0}^\infty \mathbb{P}(X_n = z, \tau_{x} > n | X_0 = x)
```
defines a stationary distribution. The proof below is adapted from Theorem 6.37 of [Robert and Casella (2004)](https://doi.org/10.1007/978-1-4757-4145-2)

First of all, let us check that $P_x$ is indeed a probability distribution. We have
```math
    \begin{align*}
        \sum_{z\in\mathcal{X}} P_x(z) & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{z\in\mathcal{X}}\sum_{n=0}^\infty \mathbb{P}(X_n = z, \tau_{x} > n | X_0 = x) \\
        & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=0}^\infty \sum_{z\in\mathcal{X}} \mathbb{P}(X_n = z, \tau_{x} > n | X_0 = x) \\
        & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=0}^\infty \mathbb{P}( \tau_{x} > n | X_0 = x) \\
        & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=0}^\infty \sum_{m=n+1}^\infty \mathbb{P}(\tau_{x} = m | X_0 = x) \\
        & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{m=0}^\infty \sum_{n=0}^{m-1} \mathbb{P}(\tau_{x} = m | X_0 = x) \\
        & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{m=0}^\infty m \mathbb{P}(\tau_{x} = m | X_0 = x) \\
        & = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]}\mathbb{E}[\tau_{x} | X_0 = x] \\
        & = 1.
    \end{align*}
```

Now, let us check that $P_x$ is invariant. For simplicity, we drop the normalizing constant and show, equivalently, the invariance of the positive measure
```math
    {\tilde P}_x(z) = \sum_{n=0}^\infty \mathbb{P}(X_n = z, \tau_{x} > n | X_0 = x).
```
We need to show that
```math
    \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) = {\tilde P}_x(z),
```
for every $z\in\mathcal{X}.$ For that, we write
```math
    \begin{align*}
        \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \sum_{y\in\mathcal{X}} K(y, z)\sum_{n=0}^\infty \mathbb{P}(X_n = y, \tau_{x} > n | X_0 = x) \\
        & = 
    \end{align*}
```


## Kac's Theorem on invariant distribution of an irreducible chain

Kac's Theorem says that if $(X_n)_n$ is irreducible and it has a stationary distribution $P,$ then
```math
    P(x) = \frac{1}{\mathbb{E}[\tau_x | X_n = x]}, \quad \forall x\in\mathcal{X}.
```

The idea is that, $m_x = \mathbb{E}[\tau_x | X_n = x]$ is the average time that the chain takes to start from $x$ and come back to $x.$ Every state $x$ is visited repeatedly, with an average time of $m_x$ between consecutive visits.


## Irreducibility in the discrete-state case

In the discrete-state case, $P$-irreducibility can be stated with point sets, so that the Markov chain is $P$-irreducible if, and only if,
```math
    x, y\in\mathcal{X},\; P(y) > 0 \Longrightarrow \exists n=n(x, y)\in\mathbb{N}, \; K_n(x, y) > 0.
```

### Positivity of an invariant probability distribution

If a discrete-state Markov chain has an invariant probability distribution $P$ and it is $P$-irreducible, then $P$ must be everywhere strictly positive, i.e. $P(x) > 0,$ for all $x\in \mathcal{X}.$

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