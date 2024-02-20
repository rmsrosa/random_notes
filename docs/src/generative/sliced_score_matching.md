# Sliced score matching

## Aim

Detail the sliced score matching method proposed by [Song, Garg, Shi, and Ermon (2020)](https://proceedings.mlr.press/v115/song20a.html) to reduce the computational cost of the score-matching method for high-dimensional problems.

## Background

The score-matching method proposed by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) is based on minimizing the **empirical implicit score matching** objective function $J_{\mathrm{ISM, data}}({\boldsymbol{\theta}})$ given by
```math
    {\tilde J}_{\mathrm{ISM, data}} = \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right),
```
where $\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})$ is the parametrized model fitting the unknown score function $\boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})$ of a random variable $\mathbf{X}$ in $\mathbb{R}^d$, $d\in\mathbb{N}$, and $\{\mathbf{x}_n; \;n=1, \ldots, N\}$ are $N\in NN$ sample points of this random variable.

The objective function $J_{\mathrm{ISM, data}}({\boldsymbol{\theta}})$ is the approximation, using the empirical distribution
```math
    \tilde p_{\mathrm{data}}(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^N \delta(\mathbf{x} - \mathbf{x}_n),
```
of the **implicit score matching** objective
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
where $p_{\mathbf{X}}(\mathbf{x})$ is the (also unknown) probability density function of $X$.

The difficulty of minimizing $J_{\mathrm{ISM, data}}({\boldsymbol{\theta}})$ is the need to compute the divergence of the model function (vector field), which amounts to computing the Hessian of the model function when the gradient of $J_{\mathrm{ISM, data}}({\boldsymbol{\theta}})$ is computed in the minimization process. This is very costly and scales badly with the dimension $d$ of the random variable (and of the neural network).

To reduce this cost, [Song, Garg, Shi, and Ermon (2020)](https://proceedings.mlr.press/v115/song20a.html) proposed the *sliced-score matching* method (see also [Song's blog on sliced score matching](http://yang-song.net/blog/2019/ssm/)). In this method, only some directional derivatives are computed at each sample point, choosing randomly before the start of the minimization process.

## The sliced score matching objective function

In this method, for each sample point $x_n$, $n=1, \ldots, N$, one randomly draws $M$ vectors $\mathbf{v}_{n, m}$, $m=1, \ldots, M$, in the space $\mathbb{R}^d$, forming the **(empirical, implicit) sliced score matching** objective function
```math
    J_{\mathrm{ISSM, data}} = \frac{1}{NM}\sum_{n=1}^N \sum_{m=1}^M \left( \frac{1}{2} \left(\mathbf{v}_{n,m} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right)^2 + \mathrm_{\mathbf{x}}\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\mathbf{v_{n,m} \cdot \mathbf{v_{n,m}}} \right).
```

Under suitable conditions, the minimizer $\boldsymbol{\theta}_{N, M}$ converges to the minimizer $\boldsymbol{\theta}$ of the implicit score matching $J_{\mathrm{ISM}}({\boldsymbol{\theta}})$ (which is the same as that of the explicit score matching), when $N\rightarrow \infty$, with $M$ fixed.

We do not implement numerically the sliced score matching method since it is still computationally intensive in Julia, with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library automatic differentiation library.

## References

1. [Y. Song, S. Garg, J. Shi, S. Ermon (2020), Sliced Score Matching: A Scalable Approach to Density and Score Estimation, Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, PMLR 115:574-584](https://proceedings.mlr.press/v115/song20a.html) -- see also the [arxiv version](https://arxiv.org/abs/1905.07088)
1. [Y. Song's blog on "Sliced Score Matching: A Scalable Approach to Density and Score Estimation"](http://yang-song.net/blog/2019/ssm/)

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)