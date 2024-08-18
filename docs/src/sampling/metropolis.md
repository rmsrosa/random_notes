# Metropolis and Metropolis-Hastings methods

The Metropolis algorithm was proposed by [N. Metropolis, A. Rosenbluth, M. Rosenbluth, A. Teller, E. Teller (1953)](http://dx.doi.org/10.1063/1.1699114) as a means to compute the equilibrium state of a many-particle system. The computations were done in the [MANIAC I (Mathematical Analyzer Numerical Integrator and Automatic Computer Model I)](https://en.wikipedia.org/wiki/MANIAC_I), an early computer developed under the direction of Nicholas Metropolis, at the Los Alamos Scientific Laboratory.

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

The algorithm is built upon the idea of designing a Markov chain $(X_n)_{n\in\mathbb{N}}$ as follows
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


## References

1. [Nicholas Metropolis,  Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller (1953), "Equation of State Calculations by Fast Computing Machines," J. Chem. Phys. 21, 1087–1092](http://dx.doi.org/10.1063/1.1699114)