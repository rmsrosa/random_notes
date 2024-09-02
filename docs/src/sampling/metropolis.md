# Metropolis and Metropolis-Hastings methods

The Metropolis algorithm was proposed by [N. Metropolis, A. Rosenbluth, M. Rosenbluth, A. Teller, E. Teller (1953)](http://dx.doi.org/10.1063/1.1699114) as a means to compute the equilibrium state of a many-particle system. The actual computations were done in the [MANIAC I (Mathematical Analyzer Numerical Integrator and Automatic Computer Model I)](https://en.wikipedia.org/wiki/MANIAC_I), an early computer developed under the direction of Nicholas Metropolis, at the Los Alamos Scientific Laboratory, and used, in particular, for the development of the hydrogen bomb, which most of the authors of this paper were involved with.

In the application, one aims to compute mean quantities of interest with respect to a statistical equilibrium state $P$ of a many-particle system with potential energy $U(x),$ which amounts to computing integrals with the probability density function of the form
```math
    p(x) = \frac{e^{-U(x)}}{Z}.
```
More generally, one considers $f(x)$ such that
```math
    p(x) = \frac{f(x)}{Z},
```
for a possibly unknown normalizing constant $Z.$ In the setting above, $f(x) = e^{-U(x)}.$ The aim, then, is to draw samples of a distribution knowning the PDF only up to a multiplicative constant.

## The Metropolis algorithm

Let us consider the case in which the event space is a finite dimensional space $\mathcal{X} = \mathbb{R}^d,$ $d\in\mathbb{E}$, and the desired probability distribution $P$ is absolutely continuous with respect to the Lebesgue measure $\mathrm{d}x,$ with density $p(x) = f(x)/Z,$ for a known function $f=f(x)$ and a possibly unkown normalization factor $Z.$

The algorithm is built upon the idea of designing a Markov chain $(X_n)_{n=0, 1, \ldots}$ and drawing a (single) sample path $(x_n)_{n=0, 1, \ldots}$ as a representative of the distribution, as follows
1. Choose a *symmetric* (and stationary) proposal transition probability $Q = Q(x, \cdot)$ with kernel density $q(x, y),$ of going from state $X_n = x$ to state $X_{n+1} = y;$
2. Choose a initial state $x_0;$
3. Proceed by induction to define the state $x_{n+1}$ from a given state $x_n$ as follows
    1. Choose a *candidate* $x' \sim Q(x_n, \cdot);$
    2. Compute the acceptance ratio $r(x_n, x'),$ where $r(x, y) = \frac{f(y)}{f(x)} = \frac{p(y)}{p(x)}.$
    3. Draw a sample $u$ from a standard uniform distribution, $u \sim \operatorname{Uniform}[0, 1];$
    4. Accept/reject step:
        1. if $u \leq r(x_n, x'),$ accept the state $x'$ and set $x_{n+1} = x;$
        2. if $u > r(x_n, x'),$ reject the state $x'$ and repeat the previous state, $x_{n+1} = x_n.$
4. Discard a initial transient time $N$ called the *burn-in* time and consider the states $x_{N+1}, x_{N+2}, \ldots,$ as a sample of the desired distribution.

### The transition kernel of the Metropolis algorithm

The density $q(x, y)$ is the kernel density of a *proposal distribution* $Q(x, \cdot).$ The actual kernel of the Markov chain $(X_n)_{n=0, 1, \ldots}$  is affected by the acceptance/rejection step. When going from state $x$ to state $y \neq x,$ assuming $f(x) > 0,$ the chances of the proposed state of being accepted has probability density
```math
    \min\left\{1, \frac{f(y)}{f(x)}\right\}.
```
When $f(x) = 0,$ one should definitely accept any $y$, thus we consider the acceptance factor
```math
    \alpha(x, y) = \begin{cases}
        \min\left\{1, \frac{f(y)}{f(x)}\right\}, & f(x) > 0, \\
        1, & f(x) = 0.
    \end{cases}
```

Then, the actual transition probability $A(x, E) = \mathbb{P}(X_{n+1} \in E | X_n = x)$ of going from state $X_n = x$ to a state $X_{n+1}$ in a measurable set $E$ has density
```math
    a(x, y) = q(x, y)\alpha(x, y).
```

Thus, if $p_{X_n}(x)$ is the density of the $X_n$ state, then the density $p_{X_{n+1}}(x)$ of the next state $X_{n+1}$ is given by
```math
    p_{X_{n+1}}(x) = \int_{\mathcal{X}} a(x, y) p_{X_n}(x)\;\mathrm{d}x.
```

### The equilibrium distribution

Any equilibrium distribution ${\tilde p}(x)$ must, then, satisfy the equation
```math
    {\tilde p}(x) = \int_{\mathcal{X}} a(x, y) {\tilde p}(x)\;\mathrm{d}x, \quad x\in\mathcal{X}.
```
In particular, we must check whether the density $p=p(x)$ of the desired distribution is an equilibrium distribution. This is obtained by checking a stronger condition known as the detailed balance equation.

### Detailed balance equation

A distribution, with density ${\tilde p}(x),$ is said to satisfy the **detailed balance equation** of a Markov chain with transition kernel $a(x, y)$ when
```math
    a(x, y){\tilde p}(x) = a(y, x){\tilde p}(y), \quad \text{a.s. } x, y\in \mathcal{X}.
```
This says that, under a given probability state with density ${\tilde p},$ the chances of going from a measurable set $B$ to a measurable set $C$ are the same as those of going from $C$ back to $B.$ It is a *reversibility* condition. This implies the stationarity of the distribution, as follows.

### Stationarity of a distribution satisfying the detailed balance equation

Given a state $X_n$ with density ${\tilde p}={\tilde p}(x),$ the density of the next state $X_{n+1} = y$ is given by
```math
    y \mapsto \int_{\mathcal{X}} q(x, y){\tilde p}(x)\;\mathrm{d}x.
```

Assuming the detailed balance equation, we have
```math
    \int_{\mathcal{X}} q(x, y){\tilde p}(x)\;\mathrm{d}x = \int_{\mathcal{X}} q(y, x){\tilde p}(y)\;\mathrm{d}x.
```

Since this is a transition kernel, it should satisfy the condition
```math
    \int_{\mathcal{X}} q(x, y) \;\mathrm{d}y = 1,
```
which essentially says that if we start at $x$ we should go *somewhere* with probability one. Switching the variables $x$ and $y$ we find that
```math
    \int_{\mathcal{X}} q(x, y){\tilde p}(x)\;\mathrm{d}x = \int_{\mathcal{X}} q(y, x){\tilde p}(y)\;\mathrm{d}x = {\tilde p}(y)\int_{\mathcal{X}} q(y, x)\;\mathrm{d}x = {\tilde p}(y).
```

This shows that ${\tilde p}$ is a stationary density distribution.

### Detailed balance equation and stationarity of the Metropolis algorithm

We have seen that the transition density of the Metropolis algorithm is
```math
    a(x, y) = q(x, y)\alpha(x, y).
```
where
```math
    \alpha(x, y) = \begin{cases}
        \min\left\{1, \frac{f(y)}{f(x)}\right\}, & f(x) > 0, \\
        1, & f(x) = 0.
    \end{cases}
```
Now we check the detailed balance equation for the distribution density $p(x) = f(x)/Z.$

We only need to check the condition almost surely on $x, y\in\mathcal{X},$ so it suffices to consider $x, y$ such that $f(x), f(y) > 0.$ When $0 < f(x) \leq f(y),$ we have
```math
    \alpha(x, y) = \min\left\{1, \frac{f(y)}{f(x)}\right\} =  \frac{f(y)}{f(x)} = \frac{p(y)}{p(x)}
```
and
```math
    \alpha(y, x) = \min\left\{1, \frac{f(x)}{f(y)}\right\} = 1.
```
Thus,
```math
    a(x, y) p(x) = q(x, y)\alpha(x, y) p(x) = q(x, y)\frac{p(y)}{p(x)}p(x) = q(x, y) p(y).
```
Since the proposal distribution is assumed to be symmetric, we have $q(x, y) = q(y, x),$ and hence
```math
    a(x, y) p(x) = q(y, x)p(y) = q(y, x)p(y).
```
Using that $\alpha(y, x) = 1,$ we obtain
```math
    a(x, y) p(x) = q(y, x)\alpha(y, x)p(y) = a(y, x)p(y).
```

Similarly, when $0 < f(y) \leq f(x),$ we find that
```math
    \alpha(x, y) = 1, \quad \alpha(y, x) = \frac{p(x)}{p(y)},
```
and the result follows in the same way,
```math
    a(x, y)p(x) = q(x, y)p(x) = q(y, x)\frac{p(x)}{p(y)}p(y) = q(y, x)\alpha(y, x)p(y) = a(y, x)p(y).
```

## The Metropolis-Hastings algorithm

[Hastings (1970)](https://doi.org/10.1093/biomet/57.1.97) extended the idea of the Metropolis algorithm to non-symmetric kernels. This is useful, for instance, when the possible events are restricted to a region of the phase space and we don't want to waste computational time generating proposal steps away of this region. For instance, if we want to generate samples of a distribution which we known to only accept non-negative coordinate values, we can use a truncated normal proposal distribution, centered on $X_n = x,$ and truncated to $x' \geq 0,$ which leads to an assymetric density.

The modification is simple. We only need to modify the acceptance ratio to
```math
    r(x, y) = \frac{f(y)q(y, x)}{f(x)q(x, y)},
```
when $f(x)q(x, y) > 0,$ otherwise $r(x, y) = 1.$

### Transition kernel

In the Metropolis-Hastings method, the transition density is still of the form
```math
    a(x, y) = q(x, y)\alpha(x, y),
```
but now
```math
    \alpha(x, y) = \begin{cases}
        \min\left\{1, \frac{f(y)q(y, x)}{f(x)q(x, y)}\right\}, & f(x)q(x, y) > 0, \\
        1, & f(x)q(x, y) = 0.
    \end{cases}
```

### The detailed balance equation

The proof is similar. But we assume that $q(x, y) > 0,$ when $f(x), f(y) > 0 (and consequently $q(y, x) > 0$). 

For the detailed balance equation, we only need to consider  $x, y$ such that $f(x), f(y) > 0.$ Thus, $q(x, y), q(y, x) > 0.$

When $0 < f(y)q(y, x) \geq f(x)q(x, y),$ we have
```math
    \alpha(x, y) = \frac{f(y)q(y, x)}{f(x)q(x, y)}, \quad \alpha(y, x) = 1,
```
so that
```math
    a(x, y) = q(x, y) \frac{f(y)q(y, x)}{f(x)q(x, y)} = \frac{p(y)q(y, x)}{p(x)}, \qquad a(y, x) = q(y, x).
```
Thus,
```math
    a(x, y) p(x) = p(y)q(y, x) = p(y)a(y, x).
```
By symmetry in $x$ and $y,$ the same follows if $0 < f(x)q(x, y) \geq f(y)q(y, x),$ i.e.
```math
    p(y)a(y, x) = a(y, x) p(y).
```
This proves the detailed balance equation, for the density $p=p(x),$ with respect to the transition density of the Metropolis-Hasting Markov chain.

## References

1. [Nicholas Metropolis,  Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller (1953), "Equation of State Calculations by Fast Computing Machines," J. Chem. Phys. 21, 1087–1092](http://dx.doi.org/10.1063/1.1699114)
2. [W. K. Hastings (1970), "Monte Carlo sampling methods using Markov chains and their applications," Biometrika 57 (1), 97-109](https://doi.org/10.1093/biomet/57.1.97)
3. [Jun S. Liu, "Monte Carlo Strategies in Scientific Computing," Springer Series in Statistics, Springer-Verlag New York 2004](https://doi.org/10.1007/978-0-387-76371-2)