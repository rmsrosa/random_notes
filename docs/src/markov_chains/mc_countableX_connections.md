# Connected states, irreducibility and uniqueness of invariant measures

The notion of connected states is a fundamental concept related to the uniqueness of an invariant measure, when it exists, at least in a local scope. When every pair of states is connected, then we have the notion of irreducibility, which extends the uniquenesse to a global scope.

## Setting

As before, we assume that $(X_n)_n$ is a time-homogeneous, discrete-time Markov chain with a countable state space. More precisely, we assume the indices are $n=0, 1, 2, \ldots,$ and that the space $\mathcal{X}$ is finite or countably infinite. The sample space is the probability space $(\Omega, \mathcal{F}, \mathbb{P}),$ where $\mathcal{F}$ is the $\sigma$-algebra on the set $\Omega$ and $\mathbb{P}$ is the probability distribution. The one-step transition distribution is denoted by $K(x, y) = \mathbb{P}(X_{n+1} = y | X_n = x),$ and is independent of $n=0, 1, \ldots,$ thanks to the time-homogeneous assumption. Similary, the $n$-step transition distribution is denoted $K_n(x, y) = \mathbb{P}(X_{k+n} = y | X_k = x),$ for $n=1, 2, \ldots,$ independently of $k=0, 1, \ldots.$

## Definitions

We start with some fundamental definitions.

### Connected points

Markov chains are about the probability of states changing with time. If starting at some state, some of the other states might be more likely to be observed in the future than others, and some might never be observed. We distinguish them by the notion of connectedness.

!!! note "Definition (connected points)"
    We say that **$x$ is connected to $y$** when there exists $n=n(x, y)\in\mathbb{N}$ such that $K_n(x, y) > 0.$ When $x$ is connected to $y$, we write $x \rightarrow y.$ When $x$ is connected to $y$ and $y$ is connected to $x,$ we say they are **connected to each other,** or **mutually connected,** and we write $x \leftrightarrow y.$

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
        x \rightarrow y \quad \Longleftrightarrow \quad \mathbb{P}(\eta_y \geq 1 | X_0 = x) > 0.
    ```

## Local uniqueness of invariant measures

In fact, if ${\tilde P}$ is a given invariant measure, then, using this invariance,
```math
    {\tilde P}(z) = \sum_{y_1} K(y_1, z){\tilde P}(y_1).
```
Splitting the summation into $y_1=x$ and $y_1\neq x,$ we have
```math
    {\tilde P}(z) = K(x, z){\tilde P}(x) + \sum_{y_1\neq x} K(y_1, z){\tilde P}(y).
```
Using again the invariance for the term ${\tilde P}(y)$ inside the summation and spliting again the summation, we have
```math
    \begin{align*}
        {\tilde P}(z) & = K(x, z){\tilde P}(x) + \sum_{y_1\neq x} K(y_1, z)\left( \sum_{y_2} K(y_2, y_1){\tilde P}(y_2) \right) \\
        & = K(x, z){\tilde P}(x) + \sum_{y_1\neq x} K(x, y_1)K(y_1, z){\tilde P}(x) + \sum_{y_1\neq x}\sum_{y_2\neq x} K(y_2, y_1)K(y_1, z){\tilde P}(y_2).
    \end{align*}
```
By induction, we obtain
```math
    \begin{align*}
        {\tilde P}(z) & = K(x, z){\tilde P}(x) \\
        & \quad + \sum_{y_1\neq x} K(x, y_1)K(y_1, z){\tilde P}(x) \\
        & \quad + \sum_{y_1\neq x}\sum_{y_2\neq x} K(x, y_1)K(y_1, z){\tilde P}(x) \\
        & \quad + \cdots \\
        & \quad + \sum_{y_1\neq x}\cdots \sum_{y_{k-1}\neq x} K(x, y_{k-1})K(y_{k-2}, y_{k-1})\cdots K(y_1, z){\tilde P}(x) \\
        & \quad + \sum_{y_1\neq x}\cdots \sum_{y_k\neq x} K(y_k, y_{k-1})K(y_{k-1}, y_{k-2})\cdots K(y_1, z){\tilde P}(x),
    \end{align*}
```
for every $k\in\mathbb{N}.$ Negleting the last term at each iteration $k,$ we find that
```math
    \begin{align*}
        {\tilde P}(z) & \geq \bigg(K(x, z) + \sum_{y_1\neq x} K(x, y_1)K(y_1, z) + \cdots \\
        & \qquad + \sum_{y_1\neq x}\cdots \sum_{y_{k-1}\neq x} K(x, y_{k-1})K(y_{k-2}, y_{k-1})\cdots K(y_1, z)\bigg){\tilde P}(x).
    \end{align*}
```
Now notice that, for $k=1,$
```math
    K(x, z) = \mathbb{P}(X_1 = z | X_0 = x) = \mathbb{P}(X_1 = z, \tau_x \geq 1 | X_0 = x),
```
for $k=2,$
```math
    \sum_{y_1\neq x} K(x, y_1)K(y_1, z) = \mathbb{P}(X_2 = z, X_1 \neq x | X_0 = x) = \mathbb{P}(X_2 = z, \tau_x \geq 2 | X_0 = x),
```
and, more generally, for any $k\in\mathbb{K},$
```math
    \begin{align*}
        \sum_{y_1\neq x}\cdots \sum_{y_{k-1}\neq x} K(x, y_{k-1})K(y_{k-2}, y_{k-1})\cdots K(y_1, z) & = \mathbb{P}(X_k = z, X_{k-1}\neq x, \ldots, X_1 \neq x | X_0 = x) \\
        & = \mathbb{P}(X_k = z, \tau_x \geq k | X_0 = x).
    \end{align*}
```
The summation of such $k$ terms is precisely ${\tilde P}_x(z),$ and we find
```math
    {\tilde P}(z) \geq {\tilde P}_x(z){\tilde P}(x).
```
Later we will see that this inequality (assuming $x$ is recurrent and ${\tilde P}(x) > 0$) implies that ${\tilde P}(z)$ is, in fact, a constant multiple of ${\tilde P}_x(z),$ from where we obtain the uniqueness up to a multiplicative constant. In any case, we see, from this calculation, that the expression for ${\tilde P}_x(z)$ appears naturally from the hypothesis of invariance.
