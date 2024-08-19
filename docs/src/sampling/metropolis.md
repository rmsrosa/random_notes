# Metropolis and Metropolis-Hastings methods

The Metropolis algorithm was proposed by [N. Metropolis, A. Rosenbluth, M. Rosenbluth, A. Teller, E. Teller (1953)](http://dx.doi.org/10.1063/1.1699114) as a means to compute the equilibrium state of a many-particle system. The actual computations were done in the [MANIAC I (Mathematical Analyzer Numerical Integrator and Automatic Computer Model I)](https://en.wikipedia.org/wiki/MANIAC_I), an early computer developed under the direction of Nicholas Metropolis, at the Los Alamos Scientific Laboratory, and used, in particular, for the development of the hydrogen bomb, which most of the authors of this paper were involved with.

In the application, one aims to compute mean quantities of interest with respect to a statistical equilibrium state of a many-particle system with potential energy $U(x),$ which amounts to compute integrals with the probability density function of the form
```math
    p(x) = \frac{e^{-U(x)}}{Z}.
```
More generally, one considers $f(x)$ such that
```math
    p(x) = \frac{f(x)}{Z},
```
for a possibly unknown normalizing constant $Z.$ In the setting above, $f(x) = e^{-U(x)}.$

## The Metropolis algorithm

The algorithm is built upon the idea of designing a Markov chain $(X_n)_{n=0, 1, \ldots}$ and drawing a (single) sample path $(x_n)_{n=0, 1, \ldots}$ as a representative of the distribution, as follows
1. Choose a *symmetric* (an stationary) transition probability (density) $T(x, y),$ of going from state $X_n = x$ to state $X_{n+1} = y;$
2. Choose a initial state $x_0;$
3. Proceed by induction to define the state $x_{n+1}$ from a given state $x_n$ as follows
    1. Choose a *candidate* $x' \sim T(X_n = x_n, \cdot);$
    2. Compute the acceptance ratio $r(x_n, x'),$ where $r(x, y) = \frac{f(y)}{f(x)}.$
    3. Draw a sample $u$ from a standard uniform distribution, $u \sim \operatorname{Uniform}[0, 1];$
    4. Accept/reject step:
        1. if $u \leq r(x_n, x'),$ accept the state $x'$ and set $x_{n+1} = x;$
        2. if $u > r(x_n, x'),$ reject the state $x'$ and repeat the previous state, $x_{n+1} = x_n.$
4. Discard a initial transient time $N$ called the *burn-in* time and consider the states $x_{N}, x_{N+1}, \ldots,$ as a sample of the desired distribution.

## The transition kernel of the Metropolis algorithm

The kernel $T(x, y)$ is the *proposal distribution* kernel. The actual kernel of the Markov chain $(X_n)_{n=0, 1, \ldots}$  is affected by the acceptance/rejection step. When going from state $x$ to state $y,$ the chances of the proposed state of being accepted has probability
```math
    r(x, y) = \min\{1, \frac{f(y)}{f(x)}\},
```
Thus the probability of going from state $x$ to state $y$ is actually
```math
    A(x, y) = T(x, y)r(x, y).
```
If $y=x,$ then $r(x, y) = 1$ and $A(x, x)$ is simply $T(x, x)$.

Thus, if $p_{X_n}(x)$ is the density of the $X_n$ state, then the density $p_{X_{n+1}}(x)$ of the next state $X_{n+1}$ is given by
```math
    p_{X_{n+1}}(x) = \int_{\mathcal{X}} A(x, y) p_{X_n}(x)\;\mathrm{d}x.
```

## The equilibrium distribution

Any equilibrium distribution $p_0(x)$ must, then, satisfy the equation
```math
    p_0(x) = \int_{\mathcal{X}} A(x, y) p_0(x)\;\mathrm{d}x, \quad x\in\mathcal{X}.
```
In particular, we must check whether the density $p=p(x)$ of the desired distribution is an equilibrium distribution. This is obtained by checking the *detailed balance equation.*

## Detailed balance equation

A distribution, with density $p(x),$ is said to satisfy the **detailed balance equation** of a Markov chain with transition kernel $A(x, y)$ when
```math
    A(x, y)p(x) = A(y, x)p(y), \quad \forall x, y\in\mathcal{X}.
```

## References

1. [Nicholas Metropolis,  Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller (1953), "Equation of State Calculations by Fast Computing Machines," J. Chem. Phys. 21, 1087–1092](http://dx.doi.org/10.1063/1.1699114)