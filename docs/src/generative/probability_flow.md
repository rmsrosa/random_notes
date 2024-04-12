# Probability flow ODEs for Itô diffusions

```@meta
Draft = false
```

## Aim

The aim is to review the probability flow sampling method, introduced by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) and generalized by [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

## Probability flow (random) ODEs

The probability flow ODEs are actually ordinary differential equations with random initial conditions, which is a special form of a *Random ODE (RODE).* The mains point is that they have the same probability distributions as their original stochastic versions.

### With a constant scalar diagonal noise amplitude

This is the original result given by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802). The SDE is the Itô diffusion with a constant scalar diagonal noise term
```math
    \mathrm{d}X_t = f(X_t)\;\mathrm{d}t + \sigma\;\mathrm{d}W_t,
```
where the unknown $\{X_t\}_t$ is a vector valued process with values in $\mathbb{R}^d,$ $d\in\mathbb{R};$ $\{W_t\}_t$ is a vector valued process in the same event space $\mathbb{R}^d$ with components made of independent Wiener processes; the drift term is a vector field $f:\mathbb{R}^d \rightarrow \mathbb{R}^d;$ and $\sigma > 0$ is a constant noise amplitude factor.

In this case, given a initial probability distribution $p_0$ for the initial random variable $X_0,$ the probability distribution $p(t, x)$ for $X_t$ is the solution of the Fokker-Planck (Kolmogorov forward) equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \frac{\sigma^2}{2}\Delta_x p(t, x).
```

The diffusion term can also be expressed as a divergence, namely
```math
    \frac{\sigma^2}{2}\Delta_x p(t, x) = \frac{\sigma^2}{2}\nabla_x \cdot \nabla_x p(t, x) = \nabla_x \cdot \left(\frac{\sigma^2}{2}\nabla_x p(t, x)\right).
```
Notice that, at least where $p(t, x) > 0,$ we can write the gradient term in terms of the Stein score $\nabla_x\log p(t, x),$
```math
    \nabla_x p(t, x) = p(t, x) \frac{\nabla_x p(t, x)}{p(t, x)} = p(t, x) \nabla_x \log p(t, x).
```
With the understanding that $s\log s$ vanishes for $s = 0$, which extends this function continuously to the region $s\geq 0,$ we can assume this relation holds everywhere. Thus, the Fokker-Planck equation takes the form
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \nabla_x \cdot (\frac{\sigma^2}{2}p(t, x) \nabla_x \log p(t, x)).
```
We now rewrite this equation as
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left(f(x) p(t, x) - \frac{\sigma^2}{2}p(t, x) \nabla_x \log p(t, x) \right) = 0.
```
Defining
```math
    \tilde f(t, x) = f(x) - \frac{\sigma^2}{2} \nabla_x \log p(t, x),
```
the Fokker-Planck equation becomes a Liouville equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left(\tilde f(x) p(t, x) \right) = 0,
```
associated with the evolution of a distribution governed by an SDE with no diffusion
```math
    \mathrm{d}X_t = \tilde f(t, X_t)\;\mathrm{d}t,
```
i.e. just a Random ODE (more specifically an ODE with random initial data) of the form
```math
    \frac{\mathrm{d}X_t}{\mathrm{d}t} = f(t, X_t) - \frac{\sigma^2}{2} \nabla_x \log p(t, X_t).
```

This equation did not receive any special name in the original work [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802), but got the name **probability flow ODE** in the work [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), which we discuss next.

Before that, we remark that one difficulty with the probability flow ODE is that it requires knownledge of the Stein score function $\nabla_x \log p(t, x)$ of the supposedly unknown distribution. This is circumvented in [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) by using a gradient log density estimator obtained from samples of the forward diffusion process, i.e. from a maximum likelyhood estimation based on the evolution of the empiral distribution of $X_0.$

On the other hand, [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) models the score function directly as a (trained) neural network.

## Nonlinear scalar diagonal noise amplitude

In [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), the authors consider the more general SDE
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
where the diffusion factor is not a scalar diagonal anymore but is a  matrix-valued, time-dependent function $G:I\times \mathbb{R}^d \rightarrow \mathbb{R}^{d\times d},$ and with the drift term also time dependent, $f:I\times \mathbb{R}^d \rightarrow \mathbb{R}^d,$ on an interval of interest $I=[0, T].$

Just for clarity, let us consider first the case in which $G=G(t, x)$ is still diagonal but not a constant anymore, i.e.
```math
    G(t, x) = g(t, x)\mathbf{I},
```
so that
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + g(t, X_t)\;\mathrm{d}W_t,
```
In this case, the Fokker-Planck equation takes the form
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \frac{1}{2}\Delta_x (g(t, x)^2 p(t, x)).
```
As before, the diffusion term can be written as
```math
    \begin{align*}
        \frac{1}{2}\Delta_x (g(t, x)^2 p(t, x)) & = \frac{1}{2}\nabla_x \cdot \nabla_x (g(t, x)^2 p(t, x)) \\
        & = \frac{1}{2}\nabla_x \cdot \left( \nabla_x g(t, x)^2 p(t, x) + g(t, x)^2 \nabla_x p(t, x) \right) \\
        & = \frac{1}{2}\nabla_x \cdot \left( \nabla_x g(t, x)^2 p(t, x) + g(t, x)^2 p(t, x) \nabla_x \log p(t, x) \right)
    \end{align*}
```
Thus, the Fokker-Planck equation becomes
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left( f(x) p(t, x) - \frac{1}{2} \nabla_x g(t, x)^2 p(t, x) - g(t, x)^2 p(t, x) \nabla_x \log p(t, x) \right) = 0,
```
which can also be viewed as the Fokker-Planck equation for the SDE with no diffusion
```math
    \mathrm{d}X_t = \tilde f(t, X_t)\;\mathrm{d}t,
```
with
```math
    \tilde f(t, x) = f(t, x) - \frac{1}{2} \nabla_x g(t, x)^2 - g(t, x)^2\nabla_x \log p(t, x),
```
which is actually a random ODE (more specifically an ODE with random initial condition),
```math
    \frac{\mathrm{d}X_t}{\mathrm{d}t} = f(t, X_t) - \frac{1}{2} \nabla_x g(t, X_t)^2 - g(t, X_t)^2\nabla_x \log p(t, X_t),
```
with the equation for $p(t, x)$ being the associated Liouville equation.

## General Itô diffusion

Now we consider the more general case considered in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), with an SDE of the form
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
where the diffusion factor is not a scalar diagonal anymore but is a  matrix-valued, time-dependent function $G:I\times \mathbb{R}^d \rightarrow \mathbb{R}^{d\times d}.$ In coordinates,
```math
    X_t = (X_t^i)_{i=1}^d, \quad W_t = (W_t^i)_{i=1}^d, \quad f(t, x) = (f_i(t, x))_{i=1}^d, \quad G(t, X_t) = (G_{ij}(t, X_t))_{i, j=1}^d,
``` 
so that the SDE reads
```math
    \mathrm{d}X_t^i = f_i(t, X_t^1, \ldots, X_t^d)\;\mathrm{d}t + \sum_{j=1}^d G_{ij}(t, X_t^1, \ldots, X_t^d)\;\mathrm{d}W_t^j.
```

In this case, the Fokker-Planck equation takes the form
```math
    \frac{\partial p}{\partial t} + \sum_{i=1}^d \frac{\partial}{\partial x_i} (f(x) p(t, x)) = \frac{1}{2}\sum_{i=1}^d \frac{\partial}{\partial x_i} \sum_{j=1}^d \frac{\partial}{\partial x_j} (G_{ik}(t, x)G_{jk}(t, x) p(t, x)).
```
Writing the divergence of a tensor field $A(x) = (A_{ij}(x))_{i,j=1}^d$ as the vector
```math
    \nabla_x \cdot A(x) = \left( \nabla_x \cdot A_{i\cdot}(x)\right)_{i=1}^d = \left( \sum_{j=1}^d \frac{\partial}{\partial x_j} A_{ij}(x)\right)_{i=1}^d,
```
we can write the Fokker-Planck as
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \frac{1}{2}\nabla_x \cdot \left(\nabla_x \cdot (G(t, x)G(t, x)^{\mathrm{tr}} p(t, x))\right).
```
As before, the diffusion term can be written as
```math
    \begin{align*}
        \frac{1}{2}\nabla_x \cdot & \left( \nabla_x \cdot (G(t, x)G(t, x)^{\mathrm{tr}} p(t, x)) \right) \\
        & = \frac{1}{2}\nabla_x \cdot \bigg( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) + G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x p(t, x) \bigg) \\
        & = \frac{1}{2}\nabla_x \cdot \bigg( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) + G(t, x)G(t, x)^{\mathrm{tr}}p(t, x)\nabla_x \log p(t, x) \bigg).
    \end{align*}
```
With that, the Fokker-Planck equation reads
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \frac{1}{2}\nabla_x \cdot \left( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) + G(t, x)G(t, x)^{\mathrm{tr}}p(t, x)\nabla_x \log p(t, x) \right).
```

Rearranging it, we have
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left( \left( f(x) - \frac{1}{2} \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) - \frac{1}{2} G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x) \right) p(t, x) \right) = 0.
```
This is the Liouville equation of the random ODE
```math
    \frac{\mathrm{d}X_t}{\mathrm{d}t} = f(x) - \frac{1}{2} \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) - \frac{1}{2} G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x).
```

## Generalized probability flow SDE for a general Itô diffusion

## References

1. [D. Maoutsa, S. Reich, M. Opper (2020), "Interacting particle solutions of Fokker-Planck equations through gradient-log-density estimation", Entropy, 22(8), 802, DOI: 10.3390/e22080802](https://doi.org/10.3390/e22080802)
1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)