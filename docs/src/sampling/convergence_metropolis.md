# Convergence of the Metropolis-Hastings method

The convergence of the Metropolis-Hastings and Gibbs MCMC methods were proved in the early 1990's in a number of articles, under reasonable assumptions. The rate of convergence, however, can be either sub-exponential (a.k.a. sub-geometric) or exponential (geometric), depending on the assumptions. These convergences are based on classical conditions of stability of Markov Chains. We will follow here the paper [Mengersen & Tweedie (1996)](https://doi.org/10.1214/aos/1033066201) and the second edition of the classic book [Meyn & Tweeedie (2009)](https://doi.org/10.1017/CBO9780511626630) (the first edition [Meyn & Tweeedie (1993)](https://doi.org/10.1007/978-1-4471-3267-7) was published a few years before the article).

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

Since the summation is countable, this is equivalent to assuming that, for any measurable set $E$ with $P$-positive measure $P(E) > 0$ and any $x\in \mathcal{X},$ there exists $n=n(E, x) \in\mathcal{N}$ such that $A_n(x, E) > 0.$ This means that any measureable set with positive measure is eventually reached, with positive measure, starting from any point in $\mathcal{X}.$

For the aperiodicity, we need the concept of *small set.* (see Section 5.2, page 102, of [Meyn & Tweeedie (2009)](https://doi.org/10.1017/CBO9780511626630)).

!!! note "Definition (small set)"
    Let $(X_n)_n$ be a Markov chain with transition probability $A_n(x, \cdot).$ A set $C$ is called a **small set** if there exist $n\in\mathbb{N}$ and some measure $\nu$ carried by $C,$
    ```math
        A_n(x, E) \geq \delta \nu(E), \quad \forall x\in E, \;\forall E\in\mathcal{B}(E).
    ```

With that, we have the definition of aperiodicity.

!!! note "Definition (aperiodic chain)
    Let $(X_n)_n$ be a Markov chain with transition probability $A_n(x, \cdot).$ Then, the chain is called **aperiodic** when, for some small set $C$ with ${\tilde p}(C) > 0,$ the greatest common divisor of all the $n$ such that
    ```math
        A_n(x, E) \geq \delta \nu(E), \quad \forall x\in E, \;\forall E\in\mathcal{B}(E),
    ```
    is $1.$

With these definitions in mind and with the results above, we check that the Metropolis-Hastings chain is $p$-irreducible and aperiodic and, thus, it convergences, in total variation, to the desired distribution $p,$ when $n\rightarrow \infty.$

## References

1. [K. L. Mengersen, R. L. Tweedie (1996), "Rates of convergence of the Hastings and Metropolis algorithms," The Annals of Statistics, 24, no. 1, 101-121](https://doi.org/10.1214/aos/1033066201)
2. [S. P. Meyn, R. L. Tweeedie (1993), "Markov Chains and Stochastic Stability," vol. 1, Springer-Verlag](https://doi.org/10.1007/978-1-4471-3267-7)
3. [S. P. Meyn, R. L. Tweeedie (2009), "Markov Chains and Stochastic Stability," vol. 2, Cambridge University Press](https://doi.org/10.1017/CBO9780511626630)