# Denoising score matching of Pascal Vincent

## Introduction

### Aim

Explore the **denoising score matching** method proposed by [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) and illustrate it by fiting a multi-layer perceptron to model the score function of a one-dimensional synthetic Gaussian-mixture distribution.

### Motivation

The motivation is to continue building a solid background on score-matching diffusion.

### Background

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) proposed modeling directly the score of a distribution. This is obtained, in theory, by minimizing an **explicit score matching** objective function. However, this function requires knowing the supposedly unknown target score function. The trick used by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) was then to do an integration by parts and rewrite the optimization problem in terms of an **implicit score matching** objective function, which yields the same minima and does not require further information from the target distribution other than some sample points.

The **implicit score matching** method requires, however, the derivative of the model score function, which is costly to compute in general.

Then, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) explored the idea of using *non-parametric Parzen density estimation* to directly approximate the explicit score matching objective, making a connection with denoising autoenconders, and proposing the **denoising score matching** method.

## Objetive function for denoising score matching

The score-matching method from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) aims to fit the model score function $\psi(\mathbf{x}; {\boldsymbol{\theta}})$ to the score function $\psi_X(\mathbf{x})$ of a random variable $\mathbf{X}$ by minimizing the 
**implicit score matching** objective
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
which is equivalent to minimizing the **explicit score matching** objective
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x},
```
due to the following identity obtained via integration by parts in the expectation
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = {\tilde J}_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C_\sigma,
```
where $C_\sigma$ is constant with respect to the parameters $\boldsymbol{\theta}$. The advantage of ${\tilde J}_{\mathrm{ISM}}({\boldsymbol{\theta}})$ is that it does not involve the unknown score function of $X$. It does, however, involve the gradient of the modeled score function, which is expensive to compute.

In practice, this is further approximated by the **empirical distribution** ${\tilde p}_0$ given by
```math
    {\tilde p}_0(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^N \delta(\mathbf{x} - \mathbf{x}_n),
```
so the implemented implicit score matching objective is
```math
    {\tilde J}_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}}) = \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right).
```

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) briefly mentions that minimizing $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ directly is "basically a non-parametric estimation problem", but dismisses it for the "simple trick of partial integration to compute the objective function very easily". As we have seen, the trick is fine for model functions for which we can compute the gradient without much trouble, but for modeling it with a neural network, for instance, it becomes computationally expensive.

A few years later, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) considered the idea of using a Parzel kernel density estimation
```math
    {\tilde p}_\sigma(\mathbf{x}) = \frac{1}{\sigma^d}\int_{\mathbb{R}^d} K\left(\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma}\right) \;\mathrm{d}{\tilde p}_0(\tilde{\mathbf{x}}) = \frac{1}{\sigma^d N}\sum_{n=1}^N K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right),
```
where $\sigma > 0$ is a kernel window parameter and $K(\mathbf{x})$ is a kernel density properly normalized to have mass one. In this way, the explicit score matching objective function is approximated by
```math
    {\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x}.
```

However, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) did not use this as a final objective function. Pascal further simplified the objective function ${\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}})$ by expanding the gradient of the logpdf of the Parzen estimator, writing a double integral with a conditional probability, and switching the order of integration. In this way, Pascal arrived at the **(Parzen) denoising score matching** objective function
```math
    {\tilde J}_{\mathrm{DSM_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}})\left\| \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) \right\|^2 \mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x},
```
where ${\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x})$ is the conditional density
```math
    {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{1}{\sigma^d}K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right).
```
Notice that the empirical distribution is not a further approximation to this objective function. It comes directly from the Parzen estimator.

### Proof that ${\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}}) = {\tilde J}_{\mathrm{DSM_\sigma}}({\boldsymbol{\theta}}) + C_\sigma$

We start by expanding the integrand of ${\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}})$ and writing
```math
    \begin{align*}
        {\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}}) & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x} \\
        & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left( \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}) + \left\|\boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\right)\mathrm{d}\mathbf{x}
    \end{align*}
```
The last term is constant with respect to the trainable parameters $\boldsymbol{\theta}$, so we just write
```math
    {\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left( \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right)\mathrm{d}\mathbf{x} + C_{\sigma, 1},
```
where
```math
    C_{\sigma, 1} = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left\|\boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\mathrm{d}\mathbf{x}.
```

Now, notice we can write
```math
    {\tilde p}_\sigma(\mathbf{x}) = \frac{1}{\sigma^d}\int_{\mathbb{R}^d} K\left(\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma}\right) \;\mathrm{d}{\tilde p}_0(\tilde{\mathbf{x}}) = \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) \;\mathrm{d}{\tilde p}_0(\tilde{\mathbf{x}}) = \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \;\mathrm{d}\tilde{\mathbf{x}},
```
where ${\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}})$ is the conditional density
```math
    {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) = \frac{1}{\sigma^d}K\left(\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma}\right).
```

Thus, the first term in the objective function becomes
```math
    \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2\;\mathrm{d}\mathbf{x} = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2\;\mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x}.
```

It remains to treat the second term. For that, we use that
```math
    \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}) = \frac{1}{{\tilde p}_\sigma(\mathbf{x})} \boldsymbol{\nabla}_{\mathbf{x}}{\tilde p}_\sigma(\mathbf{x}).
```
Thus,
```math
    \begin{align*}
        \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\mathrm{d}\mathbf{x} & = \int_{\mathbb{R}^d} \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}{\tilde p}_\sigma(\mathbf{x})\mathrm{d}\mathbf{x} \\
    \end{align*}
```

Now we write that
```math
    \begin{align*}
        \boldsymbol{\nabla}_{\mathbf{x}}{\tilde p}_\sigma(\mathbf{x}) & = \boldsymbol{\nabla}_{\mathbf{x}} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \;\mathrm{d}\tilde{\mathbf{x}} \\
        & = \int_{\mathbb{R}^d} \boldsymbol{\nabla}_{\mathbf{x}}{\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \;\mathrm{d}\tilde{\mathbf{x}} \\
        & = \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) \boldsymbol{\nabla}_{\mathbf{x}} \log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \;\mathrm{d}\tilde{\mathbf{x}}.
    \end{align*}
```
Hence,
```math
    \begin{align*}
        \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\mathrm{d}\mathbf{x} & = \int_{\mathbb{R}^d} \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \left(\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) \boldsymbol{\nabla}_{\mathbf{x}} \log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \;\mathrm{d}\tilde{\mathbf{x}}\right)\mathrm{d}\mathbf{x} \\
        & = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}} \log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) \;\mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x}
    \end{align*}
```

Putting the terms together, we find that
```math
    \begin{align*}
        {\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}}) & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left( \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right)\mathrm{d}\mathbf{x} + C_{\sigma, 1} \\
        & = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \left( \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}})\right)\;\mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x} + C_{\sigma, 1}
    \end{align*}
```
Now we add and subtract the constant (with respect to the parameters $\boldsymbol{\theta}$)
```math
    C_{\sigma, 2} = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \left\|\boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}})\right\|^2 \;\mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x}.
```

With that, we obtain
```math
    \begin{align*}
        {\tilde J}_{\mathrm{P_\sigma ESM}}({\boldsymbol{\theta}})
        & = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}}) \Bigg( \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 \\
        & \qquad\qquad - 2\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) + \left\|\boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}})\right\|^2\Bigg)\;\mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x} + C_{\sigma, 1} - C_{\sigma, 2} \\
        & = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) {\tilde p}_0(\tilde{\mathbf{x}})\left\| \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}|\tilde{\mathbf{x}}) \right\|^2 \mathrm{d}\tilde{\mathbf{x}}\,\mathrm{d}\mathbf{x} + C_{\sigma, 1} - C_{\sigma, 2} \\
        & = {\tilde J}_{\mathrm{DSM_\sigma}}({\boldsymbol{\theta}}) + C_\sigma,
    \end{align*}
```
where
```math
    C_\sigma = C_{\sigma, 1} - C_{\sigma, 2}.
```

## Numerical implementation

I won't exhibit any numerical example until I understand how ${\tilde J}_{\mathrm{DSM_\sigma}}({\boldsymbol{\theta}})$ is actually implemented, since it involves a double integral, one of them not discretized over the sample points.

## References

1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
