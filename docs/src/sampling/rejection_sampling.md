# Rejection sampling method

The **Rejection Sampling** method, or **Acceptance-rejection method,** was proposed by von Neumann (1951).

## Setting

One considers a univariate random variable $X$ with density $p=p(x)$ under the following assumptions:

1. We may not know the density $p(x)$ but we do know a non-negative function $f(x)$ proportional to the density, with some unknown normalizing constant $Z > 0,$ i.e.
   ```math 
        p(x) = \frac{f(x)}{Z}. 
   ```
2. We know how to sample from another random variable $X'$ with known density $q=q(x)$ which bounds a multiple of the function $f(x)$ with a known bound, i.e. for some known $M>0,$
```math
    f(x) \leq M q(x), \quad \forall x.
```

## The rejection sampling method

Under this setting, we obtain samples of $X$ by sampling from $X'$ and accepting or rejecting the candidate sample according to a specific criteria, and we repeat the process until a candidate is accepted, and for as many samples that we want. More precisely, here are the steps of the method.

1. Draw a sample $x'$ of $X',$ which we call a *candidate sample;*
2. Compute the *acceptance ratio* $r(x'),$ where
```math
    r(x) = \frac{f(x)}{Mq(x)}.
```
3. Draw a sample $u$ from the uniform distribution $\operatorname{Uniform}(0, 1).$
4. Accept/reject step:
    a. If $u \leq r(x'),$ accept the sample $x'$ as a sample $x=x'$ of the desired random variable $X;$
    b. otherwise, if $u > r(x'),$ reject the sample $x'$ and repeat the process drawing a new candidate and so on, until a candidate sample is accepted.
5. Repeat for as many samples as desired.

## Distribution of the accepted samples

How do you check that this really yields samples of $X?$

Let us start with another question: How often is a candidate sample accepted? First, notice that
```math
    \mathbb{P}(X' \textrm{ is accepted}) = \mathbb{P}\left(U \leq \frac{f(X')}{Mq(X')}\right) = \int_{\{q(x) > 0\}} \mathbb{P}\left( U \leq \frac{f(x)}{Mq(x)} \bigg| X' = x\right) q(x)\;\mathrm{d}x,
```
where the integral is restricted to the region $\{q(x) > 0\}$ since $q(x)$ is the PDF of $X'$ and $\{q(x) > 0\}$ is a carrier of the random variable $X',$ i.e. $q(X') > 0$ almost surely.

The probability of $U$ being less than $f(x)/Mq(x)$ is precisely this ratio, i.e.
```math
    \mathbb{P}\left( U \leq \frac{f(x)}{Mq(x)} \bigg| X' = x\right) = \frac{f(x)}{Mq(x)},
```
again in the region $\{q(x) > 0\}.$ Thus,
```math
    \begin{align*}
        \mathbb{P}(X' \textrm{ is accepted}) & = \int_{\{q(x) > 0\}} \frac{f(x)}{Mq(x)} q(x)\;\mathrm{d}x = \int_{\{q(x) > 0\}} \frac{f(x)}{M} \;\mathrm{d}x \\
        & = \frac{Z}{M} \int_{\{q(x) > 0\}} \frac{f(x)}{Z} \;\mathrm{d}x = \int_{\{q(x) > 0\}} p(x) \;\mathrm{d}x = \frac{Z}{M}.
    \end{align*}
```

We may now check the distribution of the accepted samples. This is expressed by the CDF
```math
    F_{X' \textrm{ is accepted}} = \mathbb{P}(X' \leq x | X' \textrm{ is accepted}).
```
This can be computed via
```math
    F_{X' \textrm{ is accepted}} = \mathbb{P}(X' \leq x | X' \textrm{ is accepted}) = \frac{\mathbb{P}(X' \leq x, \; X' \textrm{ is accepted})}{\mathbb{P}(X' \textrm{ is accepted})}.
```
The numerator is given by
```math
    \begin{align*}
        \mathbb{P}(X' \leq x, \; X' \textrm{ is accepted}) & = \mathbb{P}\left( X' \leq x, \; U \leq \frac{f(X')}{Mq(X')}\right) \\
        & = \int_{\{q(x') > 0, \; x' \leq x\}} \mathbb{P}\left( U \leq \frac{f(x')}{Mq(x')} \right) q(x') \;\mathrm{d}x' \\
        & = \int_{\{q(x') > 0, \; x' \leq x\}} \int_{\{0 \leq u \leq f(x')/Mq(x')\}} q(x') \;\mathrm{d}u \;\mathrm{d}x' \\
        & = \int_{\{q(x') > 0, \; x' \leq x\}} \frac{f(x')}{Mq(x')} q(x')\;\mathrm{d}x' \\
        & = \int_{x'\leq x} \frac{f(x')}{M} \;\mathrm{d}x' \\
        & = \frac{Z}{M}\int_{x'\leq x} \frac{f(x')}{Z} \;\mathrm{d}x' \\
        & = \frac{Z}{M} \int_{-\infty}^x p(x')\;\mathrm{d}x' \\
        & = \frac{Z}{M} F_{X}(x).
    \end{align*}
```
Thus,
```math
    F_{X' \textrm{ is accepted}} = \mathbb{P}(X' \leq x | X' \textrm{ is accepted}) = \frac{\mathbb{P}(X' \leq x, \; X' \textrm{ is accepted})}{\mathbb{P}(X' \textrm{ is accepted})} = \frac{\frac{Z}{M} F_{X}(x)}{\frac{Z}{M}} = F_{X}(x).
```

In other words, the probability distribution of the accepted samples of $X'$ is precisely the distribution of $X.$

## Numerical example

## References

1. J. von Neumann, "Various techniques used in connection with random digits. Monte Carlo methods" *Nat. Bureau Standards,* 12 (1951), 36–38.