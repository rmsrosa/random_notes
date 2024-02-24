# Denoising score matching of Pascal Vincent

```@meta
Draft = false
```

## Introduction

### Aim

Explore the **denoising score matching** method proposed by [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) and illustrate it by fiting a multi-layer perceptron to model the score function of a one-dimensional synthetic Gaussian-mixture distribution.

### Motivation

The motivation is to continue building a solid background on score-matching diffusion.

### Background

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) proposed modeling directly the score of a distribution. This is obtained, in theory, by minimizing an **explicit score matching** objective function. However, this function requires knowing the supposedly unknown target score function. The trick used by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) was then to do an integration by parts and rewrite the optimization problem in terms of an **implicit score matching** objective function, which yields the same minima and does not require further information from the target distribution other than some sample points.

The **implicit score matching** method requires, however, the derivative of the model score function, which is costly to compute in general.

Then, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) explored the idea of using *non-parametric Parzen density estimation* to directly approximate the explicit score matching objective, making a connection with denoising autoenconders (proposed earlir by Pascal himself, as a co-author in [Vincent, Larochelle, Lajoie, Bengio, and Manzagol(2010)](https://www.jmlr.org/papers/v11/vincent10a.html)), and proposing the **denoising score matching** method.

## Objetive function for denoising score matching

### The original explicit and implicit score matching

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

### Using Parzen estimation

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) briefly mentions that minimizing $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ directly is "basically a non-parametric estimation problem", but dismisses it for the "simple trick of partial integration to compute the objective function very easily". As we have seen, the trick is fine for model functions for which we can compute the gradient without much trouble, but for modeling it with a neural network, for instance, it becomes computationally expensive.

A few years later, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) considered the idea of using a Parzel kernel density estimation
```math
    {\tilde p}_\sigma(\mathbf{x}) = \frac{1}{\sigma^d}\int_{\mathbb{R}^d} K\left(\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma}\right) \;\mathrm{d}{\tilde p}_0(\tilde{\mathbf{x}}) = \frac{1}{\sigma^d N}\sum_{n=1}^N K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right),
```
where $\sigma > 0$ is a kernel window parameter and $K(\mathbf{x})$ is a kernel density properly normalized to have mass one. In this way, the explicit score matching objective function is approximated by
```math
    {\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x}.
```

### Denoising autoencoder

However, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) did not use this as a final objective function. Pascal further simplified the objective function ${\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}})$ by expanding the gradient of the logpdf of the Parzen estimator, writing a double integral with a conditional probability, simplifying the computation of the gradient of the logarithm of the Parzen estimation, which involves the log of a sum, to the gradient of the logarithm of the conditional probability, which involves the log of a single kernel.

In this way, Pascal arrived at the **(Parzen) denoising score matching** objective function
```math
    {\tilde J}_{\mathrm{DSM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x})\left\| \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \right\|^2 \mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}}
```
where ${\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x})$ is the conditional density
```math
    {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{1}{\sigma^d}K\left(\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma}\right).
```
Notice that the empirical distribution is not a further approximation to this objective function. It comes directly from the Parzen estimator. So, we can write
```math
    {\tilde J}_{\mathrm{DSM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\frac{1}{N}\sum_{n=1}^N \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}_n) \left\| \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}_n) \right\|^2 \mathrm{d}\tilde{\mathbf{x}}.
```

We do need, however, for the sake of implementation, to approximate the (inner) expectation with respect to the conditional distribution. This is achieved by drawing a certain number of sample points from the conditional distribution associated with ${\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}_n)$.

### Denoising autoencoder with the standard Gaussian kernel

At this point, choosing the kernel of the Parzen estimation to be the standard Gaussian kernel
```math
    G(\mathbf{x}) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} \mathbf{x}^2},
```
the conditional distribution is a Normal distribution with mean $\mathbf{x}_n$ and variance $\sigma^2$. Hence, for each $n=1, \ldots, N$, we draw $M$ sample points $\tilde{\mathbf{x}}_{n,m}$, $m=1, \ldots, M$, according to
```math
    \tilde{\mathbf{x}}_{n,m} \sim \mathcal{N}(\mathbf{x}_n, \sigma^2), \quad m=1, \ldots, M.
```
Then, using the associated empirical distribution 
```math
    {\tilde p}_{\sigma, 0}(\tilde{\mathbf{x}}|\mathbf{x}_n) = \frac{1}{M}\sum_{m=1}^M \delta(\tilde{\mathbf{x}} - \tilde{\mathbf{x}}_{n,m}),
```
we arrive at the **empirical denoising score matching** objective
```math
    {\tilde J}_{\mathrm{DSM{\tilde p}_{\sigma, 0}}}({\boldsymbol{\theta}}) = \frac{1}{2}\frac{1}{NM}\sum_{n=1}^N \sum_{m=1}^M \left\| \boldsymbol{\psi}(\mathbf{x}_{n, m}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}_{n,m}|\mathbf{x}_n) \right\|^2 \mathrm{d}\tilde{\mathbf{x}}.
```
With the Gaussian kernel, we have
```math
    \begin{align*}
        \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}_{n,m}|\mathbf{x}_n) & = \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log\left( \frac{1}{\sqrt{2\pi}\sigma^d} e^{-\frac{1}{2} \left(\frac{\tilde{\mathbf{x}} - \mathbf{x}_n}{\sigma}\right)^2} \right) \!\!\Bigg|_{\;\tilde{\mathbf{x}}=\tilde{\mathbf{x}}_{n, m}} \\
        & = \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \left( -\frac{1}{2} \left(\frac{\tilde{\mathbf{x}} - \mathbf{x}_n}{\sigma}\right)^2 - \log(\sqrt{2\pi}\sigma^d) \right)\!\!\Bigg|_{\;\tilde{\mathbf{x}}=\tilde{\mathbf{x}}_{n, m}} \\
        & = \left( - \frac{\tilde{\mathbf{x}} - \mathbf{x}_n}{\sigma^2} \right)\!\!\Bigg|_{\;\tilde{\mathbf{x}}=\tilde{\mathbf{x}}_{n, m}} \\
        & = - \frac{\tilde{\mathbf{x}}_{n, m} - \mathbf{x}_n}{\sigma^2} \\
        & = \frac{\mathbf{x}_n - \tilde{\mathbf{x}}_{n, m}}{\sigma^2}
    \end{align*}
```
Thus, in this case, the **empirical denoising score matching** objective reads
```math
    {\tilde J}_{\mathrm{DSM{\tilde p}_{\sigma, 0}}}({\boldsymbol{\theta}}) = \frac{1}{2}\frac{1}{NM}\sum_{n=1}^N \sum_{m=1}^M \left\| \boldsymbol{\psi}(\mathbf{x}_{n, m}; {\boldsymbol{\theta}}) - \frac{\mathbf{x}_n - \tilde{\mathbf{x}}_{n, m}}{\sigma^2} \right\|^2 \mathrm{d}\tilde{\mathbf{x}}.
```

Often, with $N$ sufficiently large, it suffices to take $M=1$, i.e. a single "corrupted" sample $\tilde{\mathbf{x}}_n$, for each "clean" sample point $\mathbf{x}_n$. In this way, the double summation becomes a single summation and the computation is just as fast as the one with the score of the unconditional Parzen estimation, with the benefit similar to the denosing autoencoders.

### Connection with denoising autoencoder

This brings us to the connection, discussed by [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142),  with denoising autoencoders, which were proposed a bit earlier by [Vincent, Larochelle, Lajoie, Bengio, and Manzagol(2010)](https://www.jmlr.org/papers/v11/vincent10a.html).

In an **autoencoder**, as introduced by [Kramer (1991)](https://doi.org/10.1002/AIC.690370209), one has two models, an *encoder* model $\mathbf{y} = \mathbf{f}_{\boldsymbol{\xi}}(\mathbf{x})$ and a *decoder* $\mathbf{x} = \mathbf{g}_{\boldsymbol{\eta}}(\mathbf{y})$, and the objective is to be able to encode and decode the sample, and recover the original sample as close as possible, i.e.
```math
    J(\boldsymbol{\xi}, \boldsymbol{\eta}) = \frac{1}{N} \sum_{n=1}^N \left\|\mathbf{x}_n - \mathbf{g}_{\boldsymbol{\eta}}(\mathbf{f}_{\boldsymbol{\xi}}(\mathbf{x}_n))\right\|.
```
Of course, this is useful when the *latent* space of the encoded information $\mathbf{y}$ is much smaller than the sample space, otherwise we just need to model the identity operator.

In a **denoising autoencoder**, proposed by [Vincent, Larochelle, Lajoie, Bengio, and Manzagol(2010)](https://www.jmlr.org/papers/v11/vincent10a.html), one first "corrupts" the sample points according to some distribution law, say
```math
    \tilde{\mathbf{x}}_n \sim \mathcal{P}(\tilde{\mathbf{x}}|\mathbf{x}_n),
```
and then use the corrupted sample to train the encoder/decoder pair with the objective function
```math
    J_{\mathcal{P}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \frac{1}{N} \sum_{n=1}^N \left\|\mathbf{x}_n - \mathbf{g}_{\boldsymbol{\eta}}(\mathbf{f}_{\boldsymbol{\xi}}(\tilde{\mathbf{x}}_n)) \right\|.
```

The idea is that the encoder/decoder model learns to better "reconstruct" the information even from "imperfect" information. Think about the case of encoding/decoding a handwritten message, where the letters are not "perfect" according to any font style.

In [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142), by choosing the score model of the form
```math
    \boldsymbol{\psi}(\mathbf{x}, \boldsymbol{\theta}) = \frac{1}{\sigma^2}\left( \mathbf{W}^{\mathrm{tr}} s\left(\mathbf{W}\mathbf{x} + \mathbf{b}\right) + \mathbf{c} - \mathbf{x}\right),
```
where $\boldsymbol{\theta} = (\mathbf{W}, \mathbf{b}, \mathbf{c})$ are the parameters and $s()$ is an activation function such as the sigmoid function, and choosing the noise according to
```math
    \mathcal{P}(\tilde{\mathbf{x}}|\mathbf{x}_n) = \mathcal{N}(\mathbf{x}_n, \sigma^2),
```
one obtains the connection
```math
    {\tilde J}_{\mathrm{DSM{\tilde p}_{\sigma, 0}}}({\boldsymbol{\theta}}) = \frac{1}{2\sigma^4} {\tilde J}_{\mathcal{P}}(\boldsymbol{\theta}),
```
where ${\tilde J}_{\mathcal{P}}(\boldsymbol{\theta})$ is similar to the denoising autoencoder objective $J_{\mathcal{P}}(\boldsymbol{\xi}, \boldsymbol{\eta})$, and is defined by
```math
    {\tilde J}_{\mathcal{P}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^N \left\|\mathbf{x}_n - \mathbf{h}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}_n) \right\|,
```
where $\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x})$ is almost of the form of an encoder/decoder, namely
```math
    \mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}) = \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x})) - \mathbf{x},
```
with
```math
    {f}_{\boldsymbol{\theta}}(\mathbf{x}) = \mathrm{sigmoid}\left(\mathbf{W}\mathbf{x} + \mathbf{b}\right), \quad \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{y}) = \mathbf{W}^{\mathrm{tr}}\mathbf{y} + \mathbf{c}.
```

Remember that here we are not trying to encode/decode the variate $\mathbf{x}$ itself, but its score function, so the above structure is compatible with that.

[Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) does not mention explicitly that what we denoted above by $\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x})$ is not exactly of the form $\mathbf{g}_{\boldsymbol{\theta}}(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}))$ and actually seems to suggest they are of the same form, for the sake of the connection with a denoising autoenconder. But we see here that it is not. Let us not freak out about that, though. This is good enough to draw some connection between denoising score matching and denoising autoencoder and to have this as an inspiration.

## Proof that ${\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) = {\tilde J}_{\mathrm{DSM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) + C_\sigma$

We start by renaming the dummy variable in the expression for ${\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}})$, writing
```math
    {\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\right\|^2\;\mathrm{d}\tilde{\mathbf{x}}.
```

Then, we expand the integrand of ${\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}})$ and write
```math
    \begin{align*}
        {\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\right\|^2\;\mathrm{d}\tilde{\mathbf{x}} \\
        & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left( \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}) + \left\|\boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\right\|^2\right)\mathrm{d}\tilde{\mathbf{x}}.
    \end{align*}
```
The last term is constant with respect to the trainable parameters $\boldsymbol{\theta}$, so we just write
```math
    {\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left( \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\right)\mathrm{d}\tilde{\mathbf{x}} + C_{\sigma, 1},
```
where
```math
    C_{\sigma, 1} = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left\|\boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\right\|^2\mathrm{d}\tilde{\mathbf{x}}.
```

Now, notice we can write
```math
    {\tilde p}_\sigma(\tilde{\mathbf{x}}) = \frac{1}{\sigma^d}\int_{\mathbb{R}^d} K\left(\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma}\right) \;\mathrm{d}{\tilde p}_0(\mathbf{x}) = \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \;\mathrm{d}{\tilde p}_0(\mathbf{x}) = \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \;\mathrm{d}\mathbf{x},
```
where ${\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x})$ is the conditional density
```math
    {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{1}{\sigma^d}K\left(\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma}\right).
```

Thus, the first term in the objective function becomes
```math
    \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2\;\mathrm{d}\tilde{\mathbf{x}} = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2\;\mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}}.
```

It remains to treat the second term. For that, we use that
```math
    \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}) = \frac{1}{{\tilde p}_\sigma(\tilde{\mathbf{x}})} \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}{\tilde p}_\sigma(\tilde{\mathbf{x}}).
```
Thus,
```math
    \begin{align*}
        \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\mathrm{d}\tilde{\mathbf{x}} & = \int_{\mathbb{R}^d} \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}{\tilde p}_\sigma(\tilde{\mathbf{x}})\mathrm{d}\tilde{\mathbf{x}} \\
    \end{align*}
```

Now we write that
```math
    \begin{align*}
        \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}{\tilde p}_\sigma(\tilde{\mathbf{x}}) & = \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \;\mathrm{d}\mathbf{x} \\
        & = \int_{\mathbb{R}^d} \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}{\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \;\mathrm{d}\mathbf{x} \\
        & = \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \;\mathrm{d}\mathbf{x}.
    \end{align*}
```
Hence,
```math
    \begin{align*}
        \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\mathrm{d}\tilde{\mathbf{x}} & = \int_{\mathbb{R}^d} \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \left(\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \;\mathrm{d}\mathbf{x}\right)\mathrm{d}\tilde{\mathbf{x}} \\
        & = \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \;\mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}}
    \end{align*}
```

Putting the terms together, we find that
```math
    \begin{align*}
        {\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}) \left( \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}})\right)\mathrm{d}\tilde{\mathbf{x}} + C_{\sigma, 1} \\
        & = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \left( \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2 - 2\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\right)\;\mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}} + C_{\sigma, 1}
    \end{align*}
```
Now we add and subtract the constant (with respect to the parameters $\boldsymbol{\theta}$)
```math
    C_{\sigma, 2} = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \left\|\boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\right\|^2 \;\mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}}.
```

With that, we finally obtain the desired relation
```math
    \begin{align*}
        {\tilde J}_{\mathrm{ESM{\tilde p}_\sigma}}({\boldsymbol{\theta}})
        & = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x}) \Bigg( \left\|\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}})\right\|^2 \\
        & \qquad\qquad - 2\boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) \cdot \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) + \left\|\boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\right\|^2\Bigg)\;\mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}} + C_{\sigma, 1} - C_{\sigma, 2} \\
        & = \frac{1}{2}\int_{\mathbb{R}^d} \int_{\mathbb{R}^d} {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) {\tilde p}_0(\mathbf{x})\left\| \boldsymbol{\psi}(\tilde{\mathbf{x}}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}}\log {\tilde p}_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \right\|^2 \mathrm{d}\mathbf{x}\,\mathrm{d}\tilde{\mathbf{x}} + C_{\sigma, 1} - C_{\sigma, 2} \\
        & = {\tilde J}_{\mathrm{DSM{\tilde p}_\sigma}}({\boldsymbol{\theta}}) + C_\sigma,
    \end{align*}
```
where
```math
    C_\sigma = C_{\sigma, 1} - C_{\sigma, 2}.
```

## Numerical example

We illustrate, numerically, the use of the **denoising (explicit) score matching** objective ${\tilde J}_{\mathrm{DSM{\tilde p}_\sigma}}$ to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

We use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages.

#### Packages

```@example denoisingscorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation
using Markdown

nothing # hide
```

#### Reproducibility

We set the random seed for reproducibility purposes.

```@example denoisingscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup denoisingscorematching
function Distributions.gradlogpdf(d::UnivariateMixture, x::Real)
    ps = probs(d)
    cs = components(d)
    ps1 = first(ps)
    cs1 = first(cs)
    pdfx1 = pdf(cs1, x)
    pdfx = ps1 * pdfx1
    glp = pdfx * gradlogpdf(cs1, x)
    if iszero(ps1)
        glp = zero(glp)
    end
    @inbounds for (psi, csi) in Iterators.drop(zip(ps, cs), 1)
        if !iszero(psi)
            pdfxi = pdf(csi, x)
            if !iszero(pdfxi)
                pipdfxi = psi * pdfxi
                pdfx += pipdfxi
                glp += pipdfxi * gradlogpdf(csi, x)
            end
        end
    end
    if !iszero(pdfx) # else glp is already zero
        glp /= pdfx
    end 
    return glp
end
```

### Data

We build the usual target model and draw samples from it.

```@setup denoisingscorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sigma = 0.5
sample_points = permutedims(rand(rng, target_prob, 1024))
noised_sample_points = sample_points .+ sigma .* randn(size(sample_points))
dsm_target = ( sample_points .- noised_sample_points ) ./ sigma ^ 2
data = (noised_sample_points, dsm_target)
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup denoisingscorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example denoisingscorematching
plt # hide
```

Visualizing the score function.
```@setup denoisingscorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example denoisingscorematching
plt # hide
```

### The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example denoisingscorematching
model = Chain(Dense(1 => 8, relu), Dense(8 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example denoisingscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

Here it is how we implement the **empirical denoising score matching** objective
```math
    {\tilde J}_{\mathrm{DSM{\tilde p}_{\sigma, 0}}}({\boldsymbol{\theta}}) = \frac{1}{2}\frac{1}{NM}\sum_{n=1}^N \sum_{m=1}^M \left\| \boldsymbol{\psi}(\mathbf{x}_{n, m}; {\boldsymbol{\theta}}) - \frac{\mathbf{x}_n - \tilde{\mathbf{x}}_{n, m}}{\sigma^2} \right\|^2 \mathrm{d}\tilde{\mathbf{x}}.
```
First we precompute the matrix $(\mathbf{a}_{n,m})_{n,m}$ given by
```math
    \mathbf{a}_{n,m} = \frac{\mathbf{x}_n - \tilde{\mathbf{x}}_{n, m}}{\sigma^2}.
```
Then, at each iteration of the optimization process, we take the current parameters $\boldsymbol{\theta}$ and apply the model to the perturbed points $\tilde{\mathbf{x}}_{n, m}$ to obtain the predicted scores $\{\boldsymbol{\psi}_{n,m}^{\boldsymbol{\theta}}\}$ with values
```math
    \boldsymbol{\psi}_{n,m}^{\boldsymbol{\theta}} = \boldsymbol{\psi}(\tilde{\mathbf{x}}_{n,m}, \boldsymbol{\theta}),
```
and then compute half the mean square distance between the two matrices:
```math
    \frac{1}{2}\sum_{m=1}^M \sum_{n=1}^N \left\| \boldsymbol{\psi}_{n,m}^{\boldsymbol{\theta}} - \mathbf{a}_{n,m}\right|^2.
```

In the implementation below, we just use $M=1$, so the matrices $(\boldsymbol{\psi}_{n,m}^{\boldsymbol{\theta}})_{n,m}$ and $(\mathbf{a}_{n,m})_{n,m}$ are actually just vectors. Besides, this is a scalar example, i.e. with $d=1$, so they are indeed plain real-valued vectors $(\psi_{n,1})_n$ and $(a_{n,1})_n$.

In general, though, these objects $(\boldsymbol{\psi}_{n,m}^{\boldsymbol{\theta}})_{n,m}$ and $(\mathbf{a}_{n,m})_{n,m}$ are $\mathbb{R}^d$-vector-valued matrices.

```@example denoisingscorematching
function loss_function_dsm(model, ps, st, data)
    noised_sample_points, dsm_target = data
    y_score_pred, st = Lux.apply(model, noised_sample_points, ps, st)
    loss = mean(abs2, y_score_pred .- dsm_target) / 2
    return loss, st, ()
end
```

### Optimization setup

#### Optimization method

We use the Adam optimiser.

```@example denoisingscorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example denoisingscorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example denoisingscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example denoisingscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_dsm, data, tstate_org)
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example denoisingscorematching
function train(tstate::Lux.Experimental.TrainState, vjp, data, loss_function, epochs, numshowepochs=20, numsavestates=0)
    losses = zeros(epochs)
    tstates = [(0, tstate)]
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
            loss_function, data, tstate)
        if ( epochs ≥ numshowepochs > 0 ) && rem(epoch, div(epochs, numshowepochs)) == 0
            println("Epoch: $(epoch) || Loss: $(loss)")
        end
        if ( epochs ≥ numsavestates > 0 ) && rem(epoch, div(epochs, numsavestates)) == 0
            push!(tstates, (epoch, tstate))
        end
        losses[epoch] = loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate, losses, tstates
end
```

### Training

Now we train the model with the objective function ${\tilde J}_{\mathrm{ESM{\tilde p}_\sigma{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example denoisingscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_dsm, 500, 20, 125)
nothing # hide
```

### Results

Testing out the trained model.
```@example denoisingscorematching
y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example denoisingscorematching
plot(title="Fitting", titlefont=10)

plot!(xrange, target_score', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(xx', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup denoisingscorematching
ymin, ymax = extrema(target_score)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
    plot(title="Fitting evolution", titlefont=10)

    plot!(xrange, target_score', linewidth=4, label="score function")

    scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, y_pred', linewidth=2, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example denoisingscorematching
gif(anim, fps = 20) # hide
```

Recovering the PDF of the distribution from the trained score function.
```@example denoisingscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(xrange, target_pdf', label="original")
plot!(xrange, pdf_pred', label="recoverd")
```

And the animation of the evolution of the PDF.
```@setup denoisingscorematching
ymin, ymax = extrema(target_pdf)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_pred) * dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot(title="Fitting evolution", titlefont=10, legend=:topleft)

    plot!(xrange, target_pdf', linewidth=4, fill=true, alpha=0.3, label="PDF")

    scatter!(sample_points', s -> pdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, pdf_pred', linewidth=2, fill=true, alpha=0.3, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example denoisingscorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example denoisingscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol (2010), "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion". Journal of Machine Learning Research. 11 (110), 3371-3408](https://www.jmlr.org/papers/v11/vincent10a.html)
1. [M. A. Kramer (1991), "Nonlinear principal component analysis using autoassociative neural networks", AIChE Journal. 37 (2), 233--243. doi:10.1002/aic.690370209.](https://doi.org/10.1002/AIC.690370209)
