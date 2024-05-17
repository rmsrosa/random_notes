# Score-based generative modeling through stochastic differential equations

## Introduction

### Aim

Review the work of [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) that takes a complex data distribution, adds noise to it via a stochastic differential equation and generates new samples by modeling the reverse process. It is a generalization to the continuous case of the previous discrete processes of *denoising diffusion probabilistic models* and *multiple denoising score matching.*

### Background

After [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) proposed the **implicit score matching** to model a distribution by fitting its score function, several works followed it, including the **denosing score matching** of [Paul Vincent (2011)](https://doi.org/10.1162/NECO_a_00142), which perturbed the data so the analytic expression of the score function of the perturbation could be used. Then the **denoising diffusion probabilistic models,** of [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), and the **multiple denoising score matching,** of [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), went one step further by adding several levels of noise, facilitating the generation process. The work of [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) extended that idea to the continuous case, adding noise via a stochastic differential equation.

## Forward SDE

A initial unknown probability distribution with density $p_0=p_0(x),$ associated with a random variable $X_0,$ is embedded into the distribution of an SDE of the form
```math
    \mathrm{d}X_t = f(t)X_t\;\mathrm{d}t + g(t)\;\mathrm{d}W_t,
```
with initial condition $X_0.$ The solution can be obtained with the help of the integrating factor $e^{-\int_0^t f(s)\;\mathrm{d}s}$ associated with the deterministic part of the equation. In this case,
```math
    \begin{aligned}
        \mathrm{d}\left(X_te^{-\int_0^t f(s)\;\mathrm{d}s}\right) & = \mathrm{d}X_t e^{-\int_0^t f(s)\;\mathrm{d}s} - X_t f(t) e^{\int_0^t f(s)\;\mathrm{d}s} \;\mathrm{d}t \\
        & = \left(f(t)X_t\;\mathrm{d}t + g(t)\;\mathrm{d}W_t\right)e^{-\int_0^t f(s)\;\mathrm{d}s} - X_t f(t) e^{-\int_0^t f(s)\;\mathrm{d}s} \;\mathrm{d}t \\
        & = g(t)e^{-\int_0^t f(s)\;\mathrm{d}s}\;\mathrm{d}W_t.
    \end{aligned}
```
Integrating yields
```math
    X_te^{-\int_0^t f(s)\;\mathrm{d}s} - X_0 = \int_0^t g(s)e^{-\int_0^s f(\tau)\;\mathrm{d}\tau}\;\mathrm{d}W_s.
```
Moving the exponential term to the right hand side yields the solution
```math
    X_t = X_0 e^{\int_0^t f(s)\;\mathrm{d}s} + \int_0^t e^{\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)\;\mathrm{d}W_s.
```
The mean value evolves according to
```math
    \mathbb{E}[X_t] = \mathbb{E}[X_0] e^{\int_0^t f(s)\;\mathrm{d}s}.
```
Using the Itô isometry, the second moment evolves with
```math
    \mathbb{E}[X_t^2] = \mathbb{E}[X_0^2]e^{2\int_0^t f(s)\;\mathrm{d}s} + \int_0^t e^{2\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)^2\;\mathrm{d}s.
```
Hence, the variance is given by
```math
    \operatorname{Var}(X_t) = \operatorname{Var}(X_0)e^{2\int_0^t f(s)\;\mathrm{d}s} + \int_0^t e^{2\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)^2\;\mathrm{d}s.
```

Thus, the probability density function $p(t, x)$ can be obtained by conditioning it at each initial point, with
```math
    p(t, x) = \int_{\mathbb{R}} p(t, x | 0, x_0) p_0(x_0)\;\mathrm{d}x_0,
```
and
```math
    p(t, x | 0, x_0) = \mathcal{N}(x; \mu(t)x_0, \zeta(t)^2),
```
where
```math
    \mu(t) = e^{\int_0^t f(s)\;\mathrm{d}s}
```
and
```math
    \zeta(t)^2 = \int_0^t e^{2\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)^2\;\mathrm{d}s.
```

The probability density function $p(t, x)$ can also be obtained with the help of the Fokker-Planck equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(t) p(t, x)) = \frac{1}{2}\Delta_x \left( g(t)^2 p(t, x) \right),
```
whose fundamental solutions are precisely $p(t, x | 0, x_0) = \mathcal{N}(x; \mu(t)x_0, \zeta(t)^2).$

### Examples

#### Variance-exploding SDE

For example, in the variance-exploding case (VE SDE), as discussed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), as the continuous limit of the *Multiple Denoising Score Matching,* we have
```math
    f(t) = 0, \quad g(t) = \sqrt{\frac{\mathrm{d}(\sigma(t)^2)}{\mathrm{d}t}},
```
so that
```math
    \mu(t) = 1
```
and
```math
    \zeta(t)^2 = \int_0^t \frac{\mathrm{d}(\sigma(s)^2)}{\mathrm{d}s}\;\mathrm{d}s = \sigma(t)^2 - \sigma(0)^2.
```

Thus,
```math
    p(t, x | 0, x_0) = \mathcal{N}\left( x; 1, \sigma(t)^2 - \sigma(0)^2\right).
```

#### Variance-preserving SDE

In the variance-preserving case (VP SDE), as discussed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), as the continuous limit of the *Denoising Diffusion Probabilistic Model,*
```math
    f(t) = -\frac{1}{2}\beta(t), \quad g(t) = \sqrt{\beta(t)},
```
so that
```math
    \mu(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}
```
and
```math
    \zeta(t)^2 = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s = \left. -e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau} \right|_{s=0}^{s=t} = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}.
```

Thus,
```math
    p(t, x | 0, x_0) = \mathcal{N}\left( x; e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}, 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}\right).
```

#### Sub-variance-preserving SDE

In the sub-variance-preserving case (VP SDE), proposed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) as an alternative to the previous ones,
```math
    f(t) = -\frac{1}{2}\beta(t), \quad g(t) = \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)\;\mathrm{d}s})},
```
so that
```math
    \mu(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}
```
and
```math
    \begin{align*}
        \zeta(t)^2 & = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)(1 - e^{-2\int_0^s \beta(\tau)\;\mathrm{d}\tau})\;\mathrm{d}s \\
        & = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s - \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}e^{-2\int_0^s \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s \\
        & = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s - \int_0^t e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}e^{-\int_0^s \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s \\
        & = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} \int_0^t e^{-\int_0^s \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s \\
        & = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} + e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} \left.e^{-\int_0^s \beta(\tau)\;\mathrm{d}\tau}\right|_{s=0}^t \\
        & = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} + e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} \left(e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} - 1\right) \\
        & = 1 - 2e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} + e^{-2\int_0^t \beta(\tau)\;\mathrm{d}\tau} \\
        & = \left(1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}\right)^2.
    \end{align*}
```

Thus,
```math
    p(t, x | 0, x_0) = \mathcal{N}\left( x; e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}, \left(1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}\right)^2\right).
```

## Loss function

The loss function for training is a continuous version of the loss for the multiple denoising score-matching. In that case, we had
```math
    J_{\textrm{SMLD}}(\boldsymbol{\theta}) = \frac{1}{2L}\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p(\mathbf{x})p_{\sigma_i}(\tilde{\mathbf{x}}|\mathbf{x})}\left[ \left\| s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma_i^2} \right\|^2 \right],
```
where $\lambda = \lambda(\sigma_i)$ is a weighting factor.

The continuous version becomes
```math
    J_{\textrm{SDE}}(\boldsymbol{\theta}) = \frac{1}{2T}\int_0^T \lambda(t) \mathbb{E}_{p_0(\mathbf{x}_0)p(t, \tilde{\mathbf{x}}|0, \mathbf{x})}\left[ \left\| s_{\boldsymbol{\theta}}(t, \tilde{\mathbf{x}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} p(t, \tilde{\mathbf{x}}|0, \mathbf{x}_0) \right\|^2 \right],
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", ICML'15: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, 2256-2265](https://dl.acm.org/doi/10.5555/3045118.3045358)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)
1. [J. Ho, A. Jain, P. Abbeel (2020), "Denoising diffusion probabilistic models", in Advances in Neural Information Processing Systems 33, NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)