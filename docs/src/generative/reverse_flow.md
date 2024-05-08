# Reverse probability flow

## Aim

Review the reverse probability flow used for sampling, after the Stein score function has been trained, as developed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, and Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)

## Forward SDE

A initial unknown probability distribution with density $p_0=p_0(x),$ associated with a random variable $X_0,$ is embedded into the distribution of an SDE of the form
```math
    \mathrm{d}X_t = f(t)X_t\;\mathrm{d}t + g(t)\;\mathrm{d}W_t,
```
with initial condition $X_0.$ The solution takes the form
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

The Fokker-Planck equation for the probability density function $p(t, x)$ in this case reads
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(t) p(t, x)) = \frac{1}{2}\Delta_x \left( g(t)^2 p(t, x) \right).
```
The fundamental solution can be obtained from the solution of the SDE, with $X_0 = x_0,$ so that
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

### Examples

#### Variance-exploding SDE

For example, in the variance-exploding case (VE SDE), we have
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

#### Variance-preserving SDE

In the variance-preserving case (VP SDE),
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

## References


1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)