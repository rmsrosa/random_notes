# Probability integral transform method

```@meta
Draft = false
```

In some cases, such as for the exponential distributions, it is easy to invert the cumulative distribution function and use that to transform samples of the uniform distributions to samples of the desired distribution. This is based on the following result.

!!! note "Probability Integral Transform (Universality of the Uniform Distribution)"
    Let $X$ be a univariate random variable and let $F=F(x)$ be its cumulative distribution function. Then, $U=F(X)$ is a standard uniform distribution, i.e. a uniform distribution over the interval $[0, 1].$

In other words, we can state this result in the following form:

!!! note "Inverse Probability Integral Transform"
    Let $F=F(x)$ be the cumulative distribution function of a one-dimensional distribution $\mathbb{P}.$ Let $F^{-1}(u) = \inf\{x; \;F(x) \geq u\}$ and let $U \sim \operatorname{Uniform}([0, 1]).$ Then, $X = F^{-1}(U)$ has the same distribution as $\mathbb{P}.$

This results says that, in principle, one can generate any one-dimensional distribution from a transformation of the uniform distribution. In practice, one needs a simple formula for the inverse $x = F^{-1}(u)$ for this to be feasible. One such example is that of the exponential distribution. Before we focus on this example, let us prove the above result.

> **Proof of Probability Integral Transform.**
> 
> Suppose that $F$ is invertible over the range $0< u < 1,$ for simplicity. The CDF of $U$ is then given by
> 
> ```math
> F_U(u) = \mathbb{P}(U \leq u) = \mathbb{P}(F(X) \leq u) = \mathbb{P}(X \leq F^{-1}(u)) = F(F^{-1}(u)) = u,
> ```
> over this range. This means that the $F_U$ is the CDF of the standard uniform distribution.

## Example

Let us now consider the example of the exponential distribution. The exponential distribution with rate $\lambda > 0$ has the PDF
```math
    p(x) = \begin{cases}
        \lambda e^{-\lambda x}, & x \geq 0, \\
        0, & x < 0.
    \end{cases}
```
Its CDF is
```math
    F(x) = \begin{cases}
        \int_0^x \lambda e^{-\lambda x}\;\mathrm{d}x = 1 - e^{-\lambda x}, & x \geq 0, \\
        0, & x < 0.
    \end{cases}
```
The inverse, for $x \geq 0,$ is given by
```math
    \begin{align*}
        u = F(x) & \Leftrightarrow u = 1 - e^{-\lambda x} \\
        & \Leftrightarrow e^{-\lambda x} = 1 - u \\
        & \Leftrightarrow - \lambda x = \ln (1 - u) \\
        & \Leftrightarrow x = - \frac{1}{\lambda} \ln(1 - u).
    \end{align*}
```
Thus,
```math
    F^{-1}(u) = -\frac{1}{\lambda} \ln(1 - u), \quad 0 \leq u < 1.
```

Let us generate a sample using this method.

```@setup invf
using StatsPlots # hide
using Random

rng = Xoshiro(123)

λ = 1.0
U = rand(rng, 10_000)
X = - log.(1 .- U) ./ λ
```

```@example invf
plt1 = histogram(U, title="Histogram of a sample of Unif(0, 1)", titlefont=10, bins=20, normalized=:pdf, label=false) # hide
plt2 = histogram(X, title="Histogram of Exp(λ=$λ) via inverse CDF", titlefont=10, bins=20, normalized=:pdf, label=false) # hide
plot(plt1, plt2, layout = 2, size = (800, 400)) # hide
```