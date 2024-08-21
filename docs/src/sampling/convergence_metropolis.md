# Convergence of the Metropolis-Hastings method

```@meta
Draft = false
```

The convergences of the Metropolis-Hastings and Gibbs MCMC methods were proved in the early 1990's, in a number of articles, under reasonable assumptions. The rate of convergence, however, can be either sub-exponential (a.k.a. sub-geometric) or exponential (geometric), depending on the assumptions. These convergences are based on classical conditions of stability of Markov Chains. We will follow here the paper [Mengersen & Tweedie (1996)](https://doi.org/10.1214/aos/1033066201) and the second edition of the classic book [Meyn & Tweeedie (2009)](https://doi.org/10.1017/CBO9780511626630) (the first edition [Meyn & Tweeedie (1993)](https://doi.org/10.1007/978-1-4471-3267-7) was published a few years before the article).

## Fundamental Markov chain concepts

The fundamental result, for Markov chains, that we use here is the following

!!! note "Theorem (Convergence for a Markov chain)"
    Let $(X_n)_n$ be a Markov chain with transition probability $A_n(x, E) = \mathbb{P}(X_n\in E | X_0 = x),$ where $E\subset \mathcal{X}$ is a Borel measurable set, on the event space $\mathcal{X}=\mathbb{R}^d,$ for some $d\in\mathbb{N}.$ Let $p=p(x)$ be a probability density function, with respect to the Lebesgue measure on $\mathcal{X},$ and suppose the Markov chain is $p$-irreducible and aperiodic. Then, for $p$-almost every initial condition $x\in \mathcal{X},$
    ```math
        \|A_n(x, \cdot) - p\|_{\mathrm{TV}} \rightarrow 0, \quad n\rightarrow \infty,
    ``` 
    where $\|\mu\|_{\mathrm{TV}} = \sup_{A\in\mathcal{B}(\mathcal{X})}|\mu(A)|$ is the total variation norm.

We need to clarify some terminology first. We start with the notion of ${\tilde P}$-irreducibility (see Section 4.2, page 82, of [Meyn & Tweeedie (2009)](https://doi.org/10.1017/CBO9780511626630)).

!!! note "Definition (irreducible chain)"
    A Markov chain $(X_n)_n$ with transition probability $A_n(x, \cdot)$ is called **${\tilde P}$-irreducible,** with respect to a probability distribution ${\tilde P},$ if
    ```math
        {\tilde P}(E) > 0 \Longrightarrow \sum_{n\in \mathcal{N}} A_n(x, E) > 0, \quad \forall x\in \mathcal{X}.
    ```

Since the summation is countable, this is equivalent to assuming that, for any measurable set $E$ with $P$-positive measure $P(E) > 0$ and any $x\in \mathcal{X},$ there exists $n=n(E, x) \in\mathcal{N}$ such that $A_n(x, E) > 0.$ This means that any measurable set with positive measure is eventually reached, with positive measure, starting from any point in $\mathcal{X}.$

For the aperiodicity, we need the concept of *small set.* (see Section 5.2, page 102, of [Meyn & Tweeedie (2009)](https://doi.org/10.1017/CBO9780511626630)).

!!! note "Definition (small set)"
    Let $(X_n)_n$ be a Markov chain with $n$-transition probability $A_n(x, \cdot).$ A set $C$ is called a **small set** if there exist $n\in\mathbb{N},$ $\delta > 0,$ and a probability measure $\nu$ such that
    ```math
        A_n(x, E) \geq \delta\nu(E), \quad \forall x\in E, \;\forall E\in\mathcal{B}(E).
    ```

Equivalently, some authors take $\delta = 1$ and ask $\nu$ to be a nontrivial measure, without necessarily being normalized to a probability measure.

The motivation behind this notion is that we can write
```math
    A_n(x, E) = \delta\nu(E) + (1-\delta)K_n(x, E), \quad \forall E\in\mathcal{B}(\mathcal{X}),
```
where $K_n(x, E) = (A_n(x, E) - \nu(E))/(1-\delta),$ and notice that the $n$-transition probability has a nontrivial portion $\nu$ that does not depend on the initial point $x.$ This allows us to get some uniform bounds.

But that does not give us much intuition to why it is called a small set, or in what sense that would be small. This can be illustrated by a random walk example $X_{n+1} = X_n + W_n,$ where $W_n \sim \mathcal{N}(0, 1)$ (see e.g. pg 11 of [Meyn & Tweeedie (2009)](https://doi.org/10.1017/CBO9780511626630)). If we take a set $C_r=[-r, r],$ where $r > 0,$ then, for each $x\in C,$ the PDF $\mathcal{N}(y; x, 1) = e^{-(x - y)^2/2}/\sqrt{2\pi},$ $y\in\mathbb{R},$ of the normal distribution $\mathcal{N}(x, 1)$ with mean $x$ and variance $1$ is such that
```math
    \mathcal{N}(y; x, 1) \geq \mathcal{N}(2r; 0, 1) = \frac{1}{\sqrt{2\pi}}e^{-2r^2}, \quad \forall x\in C_r = [-r, r].
```
Thus, if we take $\nu_r$ to be the uniform distribution over $C_r,$ and noticing that the Lebesgue measure of $C_r$ is $2r$ and that $\mathcal{A}(x, \cdot) = \mathcal{N}(x, 1)$ is the transition probability of this random walk, we have
```math
    \mathcal{A}(x, \cdot) \geq \delta_r \nu_r(\cdot), \quad \delta_r = \frac{2r}{\sqrt{2\pi}}e^{-r^2}.
```
The value of $\delta_r$ has its maximum at $r = \pm \sqrt{2}{2},$ decreasing to zero either as we increase $r$ towards $\infty$ or decrease it towards zero. In a sense, $\delta_r\nu_r(\cdot)$ is small, regardless of $r > 0.$

```@setup smallset
using Plots

f(x) = exp(-x^2/2) / sqrt(2π)

yy = -4:0.01:4

plts = []

for r in (0.2, 0.5, 1.0)
    δ = exp(-2r^2) / sqrt(2π)
    g(y) = δ * (abs(y) ≤ r)


    plt = plot(xaxis="\$y\$", ylims=(-0.01, 0.5), title="transition densities \$y \\mapsto a(x, y)\$ for \$x\\in C_r=[-r, r]\$, where \$r=$(r)\$\nand plot of \$\\delta_r\\chi_{C_r}(y)\$", titlefont=10, legend=false)

    for x in -r:0.04:r
        plot!(plt, yy, y -> f(y - x), color=1)
    end

    plot!(plt, yy, g, color=2, fill=true)
    push!(plts, plt)
end
```

```@example smallset
plot(plts..., layout=(3, 1), size=(600, 900)) # hide
```

With the notion of small set, we have the definition of aperiodicity.

!!! note "Definition (aperiodic chain)"
    Let $(X_n)_n$ be a Markov chain with transition probability $A_n(x, \cdot).$ Then, the chain is called **aperiodic** when, for some small set $C$ with ${\tilde p}(C) > 0,$ the greatest common divisor of all the integers $n\in\mathbb{N}$ such that
    ```math
        A_n(x, E) \geq \nu(E), \quad \forall x\in E, \;\forall E\in\mathcal{B}(E),
    ```
    is $1.$

## Metropolis-Hastings properties

With these definitions in mind and with the results above, we check that the Metropolis-Hastings chain is $p$-irreducible and aperiodic and, thus, it convergences, in total variation, to the desired distribution $p,$ when $n\rightarrow \infty.$

## References

1. [K. L. Mengersen, R. L. Tweedie (1996), "Rates of convergence of the Hastings and Metropolis algorithms," The Annals of Statistics, 24, no. 1, 101-121](https://doi.org/10.1214/aos/1033066201)
2. [S. P. Meyn, R. L. Tweeedie (1993), "Markov Chains and Stochastic Stability," vol. 1, Springer-Verlag](https://doi.org/10.1007/978-1-4471-3267-7)
3. [S. P. Meyn, R. L. Tweeedie (2009), "Markov Chains and Stochastic Stability," vol. 2, Cambridge University Press](https://doi.org/10.1017/CBO9780511626630)