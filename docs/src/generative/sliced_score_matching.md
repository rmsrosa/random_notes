# Sliced score matching

## Aim

Detail the sliced score matching method proposed by [Song, Garg, Shi, and Ermon (2020)](https://proceedings.mlr.press/v115/song20a.html) to reduce the computational cost of the score-matching method for high-dimensional problems.

## Background

The score-matching method proposed by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) is based on minimizing the **empirical implicit score matching** objective function $J_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}})$ given by
```math
    {\tilde J}_{\mathrm{ISM{\tilde p}_0}} = \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right),
```
where $\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})$ is the score function of a parametrized model probability density function $p(\mathbf{x}; \boldsymbol{\theta})$ fitting the unknown score function $\boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})$ of a random variable $\mathbf{X}$ in $\mathbb{R}^d$, $d\in\mathbb{N}$, and $\{\mathbf{x}_n; \;n=1, \ldots, N\}$ are $N\in \mathbb{N}$ sample points of this random variable.

The objective function $J_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}})$ is the approximation, using the empirical distribution
```math
    {\tilde p}_0(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^N \delta(\mathbf{x} - \mathbf{x}_n),
```
of the **implicit score matching** objective
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
where $p_{\mathbf{X}}(\mathbf{x})$ is the (also unknown) probability density function of $X$.

The difficulty of minimizing $J_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}})$ is the need to compute the divergence of the score function, which amounts to computing the Hessian of the model function, not to mention the gradient of the loss function itself, needed for the parameter optimization. This is very costly and scales badly with the dimension $d$ of the random variable (and of the neural network).

To reduce this cost, [Song, Garg, Shi, and Ermon (2020)](https://proceedings.mlr.press/v115/song20a.html) proposed the *sliced-score matching* method (see also [Song's blog on sliced score matching](http://yang-song.net/blog/2019/ssm/)). In this method, only some directional derivatives are computed at each sample point, choosing randomly before the start of the minimization process.

## The sliced score matching objective function

In this method, for each sample point $x_n$, $n=1, \ldots, N$, one randomly draws $M$ unitary vectors $\mathbf{v}_{n, m}$, $m=1, \ldots, M$, in the space $\mathbb{R}^d$ (say uniformly on the hypersphere, or with respect to a multivariate standard normal or a multivariate Rademacher distribution uniform on $\{\pm 1\}^d$, as discussed in [Song, Garg, Shi, and Ermon (2020)](https://proceedings.mlr.press/v115/song20a.html)), forming the **(empirical, implicit) sliced score matching** objective function
```math
    J_{\mathrm{ISSM{\tilde p}_0}{\tilde q}_0}({\boldsymbol{\theta}}) = \frac{1}{NM}\sum_{n=1}^N \sum_{m=1}^M \left( \frac{1}{2} \left(\mathbf{v}_{n,m} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right)^2 + \boldsymbol{\nabla}_{\mathbf{v}_{n,m}}\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \cdot \mathbf{v}_{n,m} \right),
```
where
```math
\boldsymbol{\nabla}_{\mathbf{v}_{n,m}}\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})
```
is the directional derivative of $\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})$ along $\mathbf{v}_{n,m}$.

Under suitable conditions, the minimizer $\boldsymbol{\theta}_{N, M}$ of $J_{\mathrm{ISSM{\tilde p}_0}}({\boldsymbol{\theta}})$ converges to the minimizer $\boldsymbol{\theta}$ of the implicit score matching $J_{\mathrm{ISM}}({\boldsymbol{\theta}})$ (which is the same as that of the explicit score matching), when $N\rightarrow \infty$, even with $M$ fixed (but notice that the number $NM$ of sample directions grows with $N$, already).

The divergence term in the original implicit score matching objective $J_{\mathrm{ISM}}({\boldsymbol{\theta}})$ is computed as
```math
    \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) = \sum_{i=1}^d \frac{\partial \psi_i}{\partial x_i}(\mathbf{x}; {\boldsymbol{\theta}}) = \sum_{i=1}^d \boldsymbol{\nabla}_{\mathbf{e}_i} \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \mathbf{e}_i
```
and scales with the dimension $d$ of the event space. In contrast, the computation of the collection of $M$ directional derivatives
```math
    \sum_{m=1}^M \boldsymbol{\nabla}_{\mathbf{v}_{n,m}}\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \cdot \mathbf{v}_{n,m}
```
scales with $M$. For high-dimensional problems, as in the applications in mind, $d$ is very large, and $M$ is chosen much smaller than $d$, saving a lot of computational time.

The objective $J_{\mathrm{ISSM{\tilde p}_0}}({\boldsymbol{\theta}})$ can be seen as an approximation, with the empirical distributions ($p_0$ on $\mathbf{x}$ and $q_0$ on $\mathbf{v}$), of the **(implicit) sliced score matching** objective
```math
    J_{\mathrm{ISSM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) q_{\mathbf{V}}(\mathbf{v})\left( \frac{1}{2} \left(\mathbf{v} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)^2 + \boldsymbol{\nabla}_{\mathbf{v}}\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \mathbf{v} \right) \mathrm{d}\mathbf{v}\,\mathrm{d}\mathbf{x},
```
for a probability density function $q_{\mathbf{V}}(\mathbf{v})$, associated with a random vector $\mathbf{V}$ for the directions (a uniform distribution on the hypersphere $S_{d-1}\subset \mathbb{R}^d$ or a multivariate standard normal or a multivariate Rademacher distribution uniform on $\{\pm 1\}^d$, as mentioned above) independent of $X$.

The objective $J_{\mathrm{ISSM}}({\boldsymbol{\theta}})$, in turn, can be obtained via integration by parts, from the **explicit score matching** objective
```math
    J_{\mathrm{ESSM}}({\boldsymbol{\theta}}) = \frac{1}{2} \int_{\mathbb{R}^d} \int_{\mathbb{R}^d}p_{\mathbf{X}}(\mathbf{x}) q_{\mathbf{V}}(\mathbf{v})\left\| \mathbf{v} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) + \boldsymbol{\nabla}_{\mathbf{v}}\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \mathbf{v} \right\|^2 \mathrm{d}\mathbf{v}\,\mathrm{d}\mathbf{x}.
```

We do not implement numerically the sliced score matching method since it is still computationally intensive for automatic differentiation and this approach won't be extended to our (current) final aim of denoising diffusion.

## References

1. [Y. Song, S. Garg, J. Shi, S. Ermon (2020), Sliced Score Matching: A Scalable Approach to Density and Score Estimation, Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, PMLR 115:574-584](https://proceedings.mlr.press/v115/song20a.html) -- see also the [arxiv version](https://arxiv.org/abs/1905.07088)
1. [Y. Song's blog on "Sliced Score Matching: A Scalable Approach to Density and Score Estimation"](http://yang-song.net/blog/2019/ssm/)

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)