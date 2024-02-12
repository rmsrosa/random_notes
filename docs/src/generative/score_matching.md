# Score matching of Aapo Hyvärinen

```@meta
Draft = false
```

## Motivation

The motivation is to revisit the original idea of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), as a first step towards building a solid background on score-matching diffusion.

Generative score-matching diffusion methods use Langevin dynamics to draw samples from a modeled score function. It rests on the idea of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) that one can directly model the score function, from the sample data, using a suitable loss function not depending on the unknown score function of the random variable. This is obtained by a simple integration by parts on the MSE loss function between the modeled score function and the actual score function. The integration by parts separates the dependence on the actual score function from the parameters of the model, so the fitting process (minimization over the parameters of the model) does not depend on the unknown score function.

It is worth noticing, in light of the main objective of score-matching diffusion, that the original work of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) has no diffusion. It is a direct modeling of the score function in the original probability space. But this is a fundamental work.

We also mention that the work of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) uses the modified loss function to fit some very specific predefined models. There are three examples. In these examples, the gradient of the model can be computed somewhat more explicitly. There is no artificial neural network involved and no need for automatic differention (AD). We illustrate this approach fitting a Gaussian distribution to samples of a univariate radom variables.

## The score function

For the theoretical discussion, we denote the PDF of a multivariate random variable $\mathbf{X}$, with values in $\mathbb{R}^d$, $d\in\mathbb{N}$, by $p_\mathbf{X}(\mathbf{x})$ and the score function by
```math
    \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x}) = \boldsymbol{\nabla}_{\mathbf{x}}\log(p_\mathbf{X}(\mathbf{x})) = \left( \frac{\partial}{\partial x_i} \log(p_\mathbf{X}(\mathbf{x}))\right)_{i=1, \ldots, d},
```
which is a vector field in $\mathbb{R}^d$.

The parametrized modeled score function is denoted by $\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})$, with parameter values $\boldsymbol{\theta}$.

## Loss functions for score-matching

The score-matching method from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) rests on the following ideas.

1. **Explicit score matching:** Fit the model by minimizing the expected square distance between the model score function $\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})$ and the actual score function $\boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})$, which is sometimes called *explicit score matching,*
```math
    J({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x};
```

2. **Implicit score matching:** Use integration by parts in the expectation to write that $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$, where $C$ is constant with respect to the parameters, so we only need to minimize $\tilde J$, given by
```math
    \tilde J({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
which does not involve the unknown score function of ${\mathbf{X}}$. This is sometimes called *implicit score matching.* Notice the two functions have the same gradient, hence the minimization is, theoretically, the same. This implicit score matching loss function, however, involves the gradient of the modeled score function, which might be expensive to compute.

3. **Empirical implicit score matching:** In practice, the implicit score-matching loss function is estimated via Monte-Carlo, so the unknown $p_\mathbf{X}(\mathbf{x})$ is handled implicitly by the sample data $(\mathbf{x}_n)_n$, and we minimize
```math
    {\tilde J}_{\mathrm{MC}} =  \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right).
```
In a different perspective, this uses the *empirical distribution*
```math
\frac{1}{N} \sum_{n=1}^N \delta_{\mathbf{x}_n},
```
so we call this the *empirical implicit score matching.*

## Concerning the gradient in the loss function

As mentioned before, computing a derivative to form the loss function becomes expensive when combined with the usual optimization methods to fit a neural network, as they require the gradient of the loss function itself, i.e. the optimization process involves the gradient of the gradient of something. Because of that, other methods are developed, such as using kernel density estimation, auto-encoders, finite-differences, and so on. We will explore them in due course. For the moment, we will just sketch the proof of $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$.

## Proof that $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$

### One-dimensional case

We start with the one-dimensional version of the proof from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html). In this case,
```math
    \tilde J({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_X(x) \left( \frac{1}{2}\psi(x; {\boldsymbol{\theta}})^2 + \frac{\partial}{\partial x} \psi(x; {\boldsymbol{\theta}}) \right)\;\mathrm{d}x.
```

Since this is a one-dimensional problem, the score function is a scalar and we have
```math
    \|\psi(x; {\boldsymbol{\theta}}) - \psi_X(x)\|^2 = \psi(x; {\boldsymbol{\theta}})^2 - 2\psi(x; {\boldsymbol{\theta}}) \psi_X(x) + \psi_X(x)^2.
```
Thus
```math
    J({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \left(\psi(x; {\boldsymbol{\theta}})^2 - 2\psi(x; {\boldsymbol{\theta}})\psi_X(x)\right)\;\mathrm{d}x + C,
```
where
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \psi_X(x)^2\;\mathrm{d}x
```
does not depend on ${\boldsymbol{\theta}}$.

For the mixed term, we use that the score function is
```math
    \psi_X(x) = \frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x)).
```
Differentiating the logarithm and using integration by parts, we find
```math
\begin{align*}
    -\int_{\mathbb{R}} p_X(x) \psi(x; {\boldsymbol{\theta}})\psi_X(x)\;\mathrm{d}x & = -\int_{\mathbb{R}} p_X(x) \psi(x; {\boldsymbol{\theta}})\frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x))\;\mathrm{d}x \\
    & = -\int_{\mathbb{R}} p_X(x) \psi(x; {\boldsymbol{\theta}})\frac{1}{p_X(x)}\frac{\mathrm{d}}{\mathrm{d}x}p_X(x)\;\mathrm{d}x \\
    & = -\int_{\mathbb{R}} \psi(x; {\boldsymbol{\theta}})\frac{\mathrm{d}}{\mathrm{d}x}p_X(x)\;\mathrm{d}x \\
    & = \int_{\mathbb{R}} \frac{\partial}{\partial x}\psi(x; {\boldsymbol{\theta}})p_X(x)\;\mathrm{d}x.
\end{align*}
```
Thus, we rewrite $J({\boldsymbol{\theta}})$ as
```math
    J({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_X(x) \left(\frac{1}{2}\psi(x; {\boldsymbol{\theta}})^2 + \frac{\partial}{\partial x}\psi(x; {\boldsymbol{\theta}})\right)\;\mathrm{d}x + C,
```
which is precisely $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$.

For this proof to be justified, we need
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \psi_X(x)^2\;\mathrm{d}x < \infty,
```
and
```math
    \psi(x; {\boldsymbol{\theta}}) p_X(x) \rightarrow 0, \quad |x| \rightarrow \infty,
```
for every ${\boldsymbol{\theta}}$.

### Multi-dimensional case

Here is the multi-dimensional version of the proof, from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html).

We have
```math
    \|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\|^2 = \|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\|^2 - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x}) + \|\boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\|^2.
```

Thus,
```math
    J({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left(\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\|^2 - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\right)\;\mathrm{d}\mathbf{x} + C,
```
where
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})^2\;\mathrm{d}\mathbf{x}
```
does not depend on ${\boldsymbol{\theta}}$.

For the middle term, we use explicitly that the score function is the gradient of the log of the pdf of the distribution,
```math
    \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x}) = \boldsymbol{\nabla}_{\mathbf{x}}\log(p_{\mathbf{X}}(\mathbf{x})).
```
Differentiating the logarithm and using the Divergence Theorem for the integration by parts, we find
```math
\begin{align*}
    -\int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\;\mathrm{d}\mathbf{x} & = -\int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\boldsymbol{\nabla}_{\mathbf{x}}\log(p_{\mathbf{X}}(\mathbf{x}))\;\mathrm{d}\mathbf{x} \\
    & = -\int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\frac{1}{p_{\mathbf{X}}(x)}\boldsymbol{\nabla}_{\mathbf{x}}p_{\mathbf{X}}(\mathbf{x})\;\mathrm{d}\mathbf{x} \\
    & = -\int_{\mathbb{R}} \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\boldsymbol{\nabla}_{\mathbf{x}}p_{\mathbf{X}}(\mathbf{x})\;\mathrm{d}\mathbf{x} \\
    & = \int_{\mathbb{R}} \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})p_{\mathbf{X}}(\mathbf{x})\;\mathrm{d}\mathbf{x}.
\end{align*}
```
Thus, we rewrite $J({\boldsymbol{\theta}})$ as
```math
    J({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left(\frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right)\;\mathrm{d}\mathbf{x} + C,
```
which is precisely $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$.

For this proof to be justified, we need
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})^2\;\mathrm{d}\mathbf{x} < \infty,
```
and
```math
    \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) p_{\mathbf{X}}(\mathbf{x}) \rightarrow \mathbf{0}, \quad |\mathbf{x}| \rightarrow \infty,
```
for every ${\boldsymbol{\theta}}$, which is fine for at most linearly-growing score function and model and an exponentially decreasing Gaussian mixture distribution.

## Example

```@setup aaposcorematching
using Random
using Distributions
using StatsPlots
using Optimisers

rng = Xoshiro(234)
```

We exemplify the score-matching method fitting a Normal distribution to a synthetic univariate random variable $X$. We take as the target model a Gaussian mixture with the corresponding means relatively close to each other so that fitting a single Normal distribution to them is not too far off.

```@setup aaposcorematching
prob = MixtureModel([Normal(2, 1), Normal(4, 1)], [0.4, 0.6])
```

Say we have a sample $\{x_n\}_{n=1}^N$ of $X$, where $N\in\mathbb{N}$.

```@setup aaposcorematching
sample = rand(rng, prob, 100)
xrange = range(minimum(sample) - 0.2, maximum(sample) + 0.2, length=200)
```

```@example aaposcorematching
scatter(sample, one.(sample), xlims=extrema(xrange), ylims=(0, 2), axis=false, legend=false, grid=false, size=(600, 80)) # hide
```

The model is a score function of a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, whose PDF is
```math
    p_{\theta}(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}.
```
The logpdf is
```math
    \log p_{\theta}(x) = -\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2 - \log\left(\sqrt{2\pi}\sigma\right).
```
And the score function is
```math
    \psi(x; \mu, \sigma) = \frac{\partial}{\partial x}\log p_{\theta}(x) = - \frac{x - \mu}{\sigma^2},
```
with parameters $\boldsymbol{\theta} = (\mu, \sigma)$. The derivative of the score function, needed for the loss function, is constant,
```math
    \frac{\partial}{\partial x} \psi(x; \mu, \theta) = -\frac{1}{\sigma^2}.
```

Thus, the implicit score matching loss becomes
```math
    \tilde J({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_X(x) \left( \frac{1}{2}\left(\frac{x - \mu}{\sigma^2}\right)^2 - \frac{1}{\sigma^2} \right)\;\mathrm{d}x.
```
The Monte-Carlo approximation, with the empirical distribution, is
```math
    {\tilde J}_{\mathrm{MC}} = \frac{1}{N} \sum_{n=1}^N \left( \frac{1}{2}\left(\frac{x_n - \mu}{\sigma^2}\right)^2 - \frac{1}{\sigma^2} \right).
```

Implementing this loss for the given model score function yields the following solution, represented graphically in the plot.

```@example aaposcorematching
loss_function(mu, sigma) = mean( (xn - mu)^2 / sigma^4 / 2 - 1 / sigma^2 for xn in sample)
```

```@setup aaposcorematching
murange = copy(xrange)
sigmarange = range(1, 10.0, length=200)
```

```@example aaposcorematching
surface(murange, sigmarange, log ∘ loss_function, color=:vik) # hide
```

```@example aaposcorematching
heatmap(murange, sigmarange, log ∘ loss_function, color=:vik) # hide
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
