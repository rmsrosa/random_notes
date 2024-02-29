# Denoising diffusion probabilistic models

```@meta
Draft = true
```

## Introduction

### Aim

Review the **denoising diffusion probabilistic models** introduced in [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and further improved in [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html).

### Motivation

Build a solid foundation on generative diffusion models, for which DDPM is an integral part as a discretized analog of the SDE model.

### Background

The main idea in [Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) is to embed the random variable we want to model into a Markov chain and model the whole Markov chain. This is a much more complex task and greatly enlarge the dimension of the problem, but which yields more stability to the training and the generative processes. The desired random variable, for which we only have access to a sample, is considered as an initial condition to a Markov chain converging to a simple and tractable distribution, usually a normal distribution. The training process fits a model to the whole Markov chain. Then, the model is used to reverse the process and generate (aproximate) samples of our target distribution from samples of the tractable distribution. The tractable final distribution becomes the initial distribution of the reverse process, and the initial desired target distribution becomes the final distribution.

## Details of the method

We consider the unknown target distribution to be an initial distribution for a Markov process, so we denote it by $\mathbb{P}_0$, with the associated random variable denoted by $\mathbf{X}_0$ in $\mathbb{R}^d$, $d\in\mathbb{N}$. The probability density function is denoted by $p_0(\mathbf{x})$.

The idea is to define a forward Markov process to add noise to the initial random variable and drive the distribution close to a standard Gaussian distribution. This forward process is pretty well defined and straightforward to implement. From the initial sample points we obtain samples of the Markov process. The idea then is to use those sample trajectories to learn the reverse Markov process. With this model of the reverse process, we can build new sample points out of (new) samples of the standard Gaussian distribution.

Since the approximate reverse process is made of a initial standard Gaussian distribution and the process just adds Gaussian noises, the approximate reverse process is a Gaussian process. Thus we can just parametrized it by its (time-dependent) mean and variance. The problem is that ... 

### The forward Markov chain

The initial random variable $\mathbf{X}_0$ is evolved in time as a Markov chain $\{\mathbf{X}_k\}_{k\in \mathbb{Z}_0^+}$ according to
```math
    \mathbf{X}_k = \sqrt{1 - \beta_k}\mathbf{X}_{k-1} + \sqrt{\beta_k}\mathbf{Z}_k, \quad \mathbf{Z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
```
where $\boldsymbol{\beta}=\{\beta_k\}_{k\in \mathbb{N}}$ is given, with $0 < \beta_k < 1$, for every $k$.

The marginal probability density function of the step $k$ conditioned on the step $k-1$ satisfies
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_k|\mathbf{x}_{k-1}) \sim \mathcal{N}\left(\sqrt{1 - \beta_k}\mathbf{x}_{k-1}, \beta_k\right).
```

By taking the expectation of the recurrence relation of the Markov chain, we see that the means ${\bar{\mathbf{X}}}_k = \mathbb{E}[\mathbf{X}_k]$ evolve according to
```math
    {\bar{\mathbf{X}}}_k = \sqrt{1 - \beta_k}{\bar{\mathbf{X}}}_{k-1},
```
With $0 < \beta_k < 1$, we see that the mean value decays to zero exponentially,
```math
    {\bar{\mathbf{X}}}_k \rightarrow 0, \qquad k \rightarrow \infty.
```

Notice that
```math
    {\bar{\mathbf{X}}}_k^2 = (1 - \beta_k){\bar{\mathbf{X}}}_{k-1}^2,
```
and
```math
    \mathbf{X}_k - {\bar{\mathbf{X}}}_k = \sqrt{1 - \beta_k}\left(\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right) + \sqrt{\beta_k}\mathbf{Z}_k.
```
Thus,
```math
    \left|\mathbf{X}_k - {\bar{\mathbf{X}}}_k\right|^2 = (1 - \beta_k)\left|\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right|^2 + 2\sqrt{1 - \beta_k}\left(\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right)\sqrt{\beta_k}\mathbf{Z}_k + \beta_k \mathbf{Z}_k^2.
```
Taking the expectation, we find that
```math
    \mathbb{E}\left[ \left\|\mathbf{X}_k - {\bar{\mathbf{X}}}_k\right\|^2 \right] = (1 - \beta_k) \mathbb{E}\left[ \left|\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right\|^2 \right] + \beta_k.
```
This means that the variance satisfies
```math
    \operatorname{Var}(\mathbf{X}_k) = (1 - \beta_k)\operatorname{Var}(\mathbf{X}_{k-1}) + \beta_k.
```
The parameters $\boldsymbol{\beta}=\{\beta_k\}_k$ are said to be a **variance schedule**.

At the limit $k\rightarrow \infty$, with $0 < \beta_k < 1$, we see that the variance converges exponentially to one,
```math
    \operatorname{Var}(\mathbf{X}_k) \rightarrow 1, \qquad k \rightarrow \infty.
```
Thus, $\{\mathbf{X}_k\}_k$ converges to the standard Gaussian distribution.

We can also write
```math
    \operatorname{Var}(\mathbf{X}_k) - \operatorname{Var}(\mathbf{X}_{k-1}) = - \beta_k\operatorname{Var}(\mathbf{X}_{k-1}) + \beta_k
```
and
```math
    \frac{\operatorname{Var}(\mathbf{X}_k) - \operatorname{Var}(\mathbf{X}_{k-1})}{\beta_k} = -\operatorname{Var}(\mathbf{X}_{k-1}) + 1,
```
so that the variance schedule $\boldsymbol{\beta}=\{\beta_k\}_k$ is also interpreted as **step sizes**.

The probability density functions $p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}) = p_{\boldsymbol{\beta}}(\mathbf{x}_0, \ldots, \mathbf{x}_K)$ of the Markov chain, where $\mathbf{x}_{0:K} = (\mathbf{x}_0, \dots, \mathbf{x}_K)$ is a portion of a trajectory up to some sufficiently large step $K\in\mathbb{N}$, satisfies the conditional marginal relation
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_k|\mathbf{x}_{k-1}) \sim \mathcal{N}(\sqrt{1 - \beta_k}\mathbf{x}_{k-1}, \beta_k),
```
Then,
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}|\mathbf{x}_0) = p_{\boldsymbol{\beta}}(\mathbf{x}_{K}|\mathbf{x}_{K-1})\cdots p_{\boldsymbol{\beta}}(\mathbf{x}_1|\mathbf{x}_0) = \prod_{k=1}^{K}p_{\boldsymbol{\beta}}(\mathbf{x}_k|\mathbf{x}_{k-1}).
```

Thus,
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}) = \int_{\mathbb{R}^d} \prod_{k=1}^K p_{\boldsymbol{\beta}}(\mathbf{x}_k|\mathbf{x}_{k-1})p_0(\mathbf{x_0})\;\mathrm{d}\mathbf{x_0}.
```

An approximate distribution is obtained with the empirical distribution based on samples of the initial random variable $\mathbf{X}_0$, which we denote by
```math
    {\tilde p}_{\delta_0}(\mathbf{x}_0) = \frac{1}{N}\sum_{n=1}^N \delta(\mathbf{x}_0 - \mathbf{x}_0^n),
```
where the samples are denoted now by $\{\mathbf{x}_0^n\}_{n=1}^N$.

We can iterate the Markov transition formula and write
```math
    \begin{align*}
        \mathbf{X}_k & = \sqrt{1 - \beta_{k}}\mathbf{X}_{k-1} + \sqrt{\beta_{k}}\mathbf{Z}_{k} \\
        & = \sqrt{1 - \beta_{k}}\left( \sqrt{1 - \beta_{k-1}}\mathbf{X}_{k-2} + \sqrt{\beta_{k-1}}\mathbf{Z}_{k-1} \right) + \sqrt{\beta_{k}}\mathbf{Z}_{k} \\
        & = \sqrt{1 - \beta_{k}}\sqrt{1 - \beta_{k-1}}\mathbf{X}_{k-2} + \sqrt{1 - \beta_{k}}\sqrt{\beta_{k-1}}\mathbf{Z}_{k-1} + \sqrt{\beta_{k}}\mathbf{Z}_{k} \\
        & = \sqrt{1 - \beta_{k}}\sqrt{1 - \beta_{k-1}}\left( \sqrt{1 - \beta_{k-2}}\mathbf{X}_{k-3} + \sqrt{\beta_{k-2}}\mathbf{Z}_{k-2} \right) + \sqrt{1 - \beta_{k}}\sqrt{\beta_{k-1}}\mathbf{Z}_{k-1} + \sqrt{\beta_{k}}\mathbf{Z}_{k} \\
        & = \cdots \\
        & = \sqrt{1 - \beta_{k}} \cdots \sqrt{1 - \beta_1}\mathbf{X}_0 + \sqrt{1 - \beta_{k}}\cdots \sqrt{1 - \beta_1}\mathbf{Z}_1 + \cdots + \sqrt{1 - \beta_{k}}\sqrt{\beta_{k-1}}\mathbf{Z}_{k-1} + \sqrt{\beta_{k}}\mathbf{Z}_{k}.
    \end{align*}
```
By defining
```math
    \alpha_k = 1 - \beta_k,
```
we rewrite this as
```math
    \mathbf{X}_k = \sqrt{\alpha_{k}\cdots \alpha_1}\mathbf{X}_0 + \sqrt{\alpha_{k}\cdots\alpha_2}\sqrt{1 - \alpha_1}\mathbf{Z}_1 + \cdots + \sqrt{\alpha_{k}}\sqrt{1 - \alpha_{k-1}}\mathbf{Z}_{k-1} + \sqrt{1 - \alpha_{k}}\mathbf{Z}_{k}.
```

Since the $\mathbf{Z}_k$ are standard Gaussian random variables, thus with zero mean and variance one, their linear combination is also a Gaussian with zero mean, while the variance is given by the sum of the variances which end up simplifying to
```math
\alpha_{k}\cdots\alpha_2 (1 - \alpha_1) + \cdots + \alpha_{k}(1 - \alpha_{k-1}) + (1 - \alpha_{k}) = 1 - \alpha_{k}\cdots \alpha_1.
```

Defining now
```math
    \bar\alpha_k = \alpha_k\cdots\alpha_1,
```
we obtain
```math
    \mathbf{X}_k = \sqrt{\bar{\alpha}_{k}}\mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\mathbf{Z}}_k, \qquad \bar{\mathbf{Z}}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
```

### The backward Markov chain

Now we want to be able to revert the Markov chain. But what would be $\mathbf{X}_{k-1}$ given $\mathbf{X}_k = \mathbf{x}_k?$

When also conditioned on the initial sample, we can use Bayes' rule and write
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0\right) = \frac{p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)p_{\boldsymbol{\beta}}\left(\mathbf{x}_{k-1}|\mathbf{x}_0\right)}{p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_0\right)}
```

Using the Markovian property on the first term of the nominator and ignoring the normalization constant, we know that
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) = p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}\right) \propto \exp\left(-\frac{1}{2}\frac{\left(\mathbf{x}_k - \sqrt{\alpha_k}\mathbf{x}_{k-1}\right)^2}{\beta_k}\right),
```
while
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_{k-1}|\mathbf{x}_0\right) \propto \exp\left(-\frac{1}{2}\frac{\left(\mathbf{x}_{k-1} - \sqrt{\bar{\alpha}_{k-1}}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_{k-1}}\right),
```
and
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_0\right) \propto \exp\left(-\frac{1}{2}\frac{\left(\mathbf{x}_k - \sqrt{\bar{\alpha}_k}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_k}\right).
```
Thus,
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) \propto \exp\left( - \frac{1}{2}\left(\frac{\left(\mathbf{x}_k - \sqrt{\alpha_k}\mathbf{x}_{k-1}\right)^2}{\beta_k} + \frac{\left(\mathbf{x}_{k-1} - \sqrt{\bar{\alpha}_{k-1}}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_{k-1}} - \frac{\left(\mathbf{x}_k - \sqrt{\bar{\alpha}_k}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_k} \right)\right).
```
We separate the dependence on the variable $\mathbf{x}_{k-1}$ from that on the conditioned variables $\mathbf{x}_k$ and $\mathbf{x}_0$.
```math
    \begin{align*}
        p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) & \propto \exp\bigg( - \frac{1}{2}\bigg(\frac{\mathbf{x}_k^2 - 2\mathbf{x}_k \sqrt{\alpha_k}\mathbf{x}_{k-1} + \alpha_k\mathbf{x}_{k-1}^2}{\beta_k} \\
        & \qquad \qquad \qquad + \frac{\mathbf{x}_{k-1}^2 - 2\mathbf{x}_{k-1}\sqrt{\bar{\alpha}_{k-1}}\mathbf{x}_0 + \bar{\alpha}_{k-1}\mathbf{x}_0^2}{1 - \bar{\alpha}_{k-1}} \\
        & \qquad \qquad \qquad \qquad \qquad - \frac{\mathbf{x}_k^2 - \mathbf{x}_k\sqrt{\bar{\alpha}_k}\mathbf{x}_0 + \bar{\alpha}_k\mathbf{x}_0^2}{1 - \bar{\alpha}_k} \bigg)\bigg) \\
        & = \exp\bigg( -\frac{1}{2}\bigg( \left( \frac{\alpha_k}{\beta_k} + \frac{1}{1 - \bar{\alpha}_{k-1}}\right)\mathbf{x}_{k-1}^2 \\
        & \qquad \qquad \qquad - \left(\frac{2\sqrt{\alpha_k}}{\beta_k}\mathbf{x}_k + \frac{2\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_{k-1}}\mathbf{x}_0\right)\mathbf{x}_{k-1} \\
        & \qquad \qquad \qquad + \left( \frac{1}{\beta_k} - \frac{1}{1 - \bar{\alpha}_k}\right)\mathbf{x}_{k}^2 + \frac{\sqrt{\bar{\alpha}_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k\mathbf{x}_0 \\
        & \qquad \qquad \qquad \qquad \qquad + \left( \frac{\bar{\alpha}_{k-1}}{1 - \bar{\alpha}_{k-1}} - \frac{\bar{\alpha}_k}{1 - \bar{\alpha}_k} \right)\mathbf{x}_0^2\bigg)\bigg).
    \end{align*}
```
Completing the squares, we write
```math
    \left( \frac{\alpha_k}{\beta_k} + \frac{1}{1 - \bar{\alpha}_{k-1}}\right)\mathbf{x}_{k-1}^2 - \left(\frac{2\sqrt{\alpha_k}}{\beta_k}\mathbf{x}_k + \frac{2\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_{k-1}}\mathbf{x}_0\right)\mathbf{x}_{k-1} = \frac{\left(\mathbf{x}_{k-1} - \tilde{\boldsymbol{\mu}}_k\right)^2}{\tilde{\beta}_k} - \frac{{\tilde{\boldsymbol{\mu}}}_k^2}{\tilde \beta_k},
```
with
```math
    \begin{align*}
        \tilde\beta_k & = \frac{1}{\left( \frac{\alpha_k}{\beta_k} + \frac{1}{1 - \bar{\alpha}_{k-1}}\right)}, \\
        \frac{\tilde{\boldsymbol{\mu}}_k}{\tilde\beta_k} & = \frac{\sqrt{\alpha_k}}{\beta_k}\mathbf{x}_k + \frac{\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_{k-1}}\mathbf{x}_0.
    \end{align*}
```
Using that $\beta_k = 1 - \alpha_k$, we find the variance of $p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)$ to be
```math
    \tilde\beta_k = \frac{1}{\left( \frac{\alpha_k}{\beta_k} + \frac{1}{1 - \bar{\alpha}_{k-1}}\right)} = \frac{\beta_k(1 - \bar{\alpha}_{k-1})}{\alpha_k(1 - \bar{\alpha}_{k-1}) + \beta_k} = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\beta_k,
```
With that, we rewrite the mean of $p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)$ as
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k & = \tilde\beta_k\left(\frac{\sqrt{\alpha_k}}{\beta_k}\mathbf{x}_k + \frac{\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_{k-1}}\mathbf{x}_0\right) = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\beta_k\left(\frac{\sqrt{\alpha_k}}{\beta_k}\mathbf{x}_k + \frac{\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_{k-1}}\mathbf{x}_0\right) \\
        & = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0.
    \end{align*}
```
Then, we obtain
```math
    \begin{align*}
        \frac{{\tilde{\boldsymbol{\mu}}}_k^2}{\tilde \beta_k} & = \frac{1 - \bar{\alpha}_k}{(1 - \bar{\alpha}_{k-1})\beta_k}\left(\frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0\right) \\
        & = \frac{\sqrt{\alpha_k}}{\beta_k}\mathbf{x}_k + \frac{\sqrt{\bar{\alpha}_{k-1}}}{(1 - \bar{\alpha}_{k-1})}\mathbf{x}_0.
    \end{align*}
```

Thus, we write
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) \propto \exp\bigg( -\frac{1}{2}\bigg( \frac{\left(\mathbf{x}_{k-1} - \tilde{\boldsymbol{\mu}}_k\right)^2}{\tilde{\beta}_k} + \tilde\gamma_k \bigg)\bigg),
```
where
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k & = \tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0) = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0, \\
        \tilde\beta_k & = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\beta_k, \\
        \tilde\gamma_k & = \tilde\gamma_k(\mathbf{x}_k, \mathbf{x}_0) = \left( \frac{1}{\beta_k} - \frac{1}{1 - \bar{\alpha}_k}\right)\mathbf{x}_{k}^2 + \frac{\sqrt{\bar{\alpha}_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k\mathbf{x}_0 \\
        & \qquad \qquad \qquad \qquad + \left( \frac{\bar{\alpha}_{k-1}}{1 - \bar{\alpha}_{k-1}} - \frac{\bar{\alpha}_k}{1 - \bar{\alpha}_k} \right)\mathbf{x}_0^2 - \frac{{\tilde{\boldsymbol{\mu}}}_k^2}{\tilde \beta_k}.
    \end{align*}
```
Hence, we find that
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) \sim \mathcal{N}\left(\tilde{\boldsymbol{\mu}}_k, \tilde \beta_k\mathbf{I}\right).
```

With that, we can write
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}\right) = \int_{\mathbb{R}^d} p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1},\mathbf{x}_0\right)p_{\boldsymbol{\beta}}(\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0,
```
which we eventually approximate with the empirical distribution to train our model.

Notice we can write the (initial) target distribution as
```math
    p_0(\mathbf{x}_0) = p_{\boldsymbol{\beta}}(\mathbf{x}_0) = \int_{(\mathbf{R}^d)^{K}} p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}) \;\mathrm{d}\mathbf{x}_{1:K},
```
and then
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}) = \int_{\mathbb{R}^d} p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}|\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0,
```
with
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_{0:K}|\mathbf{x}_0) = p_{\boldsymbol{\beta}}(\mathbf{x}_0|\mathbf{x}_1, \mathbf{x}_0)p_{\boldsymbol{\beta}}(\mathbf{x}_1|\mathbf{x}_2, \mathbf{x}_0)\cdots p_{\boldsymbol{\beta}}(\mathbf{x}_{K-1}|\mathbf{x}_K, \mathbf{x}_0),
```
and
```math
    p_{\boldsymbol{\beta}}(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0) = G(\mathbf{x}_{k-1}; \tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0), \tilde \beta_k(\mathbf{x}_k, \mathbf{x}_0)),
```
where $G(\mathbf{x}; \boldsymbol{\mu}, \beta)$ is the Gaussian kernel
```math
    G(\mathbf{x}; \boldsymbol{\mu}, \beta) = \frac{1}{\beta^{d/2}}G\left(\frac{\mathbf{x} - \boldsymbol{\mu}}{\beta^{1/2}}\right) = \frac{1}{\sqrt{2\pi\beta^d}}e^{-\frac{1}{2}\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{\beta}},
```
with $G(\mathbf{x})$ being the standard Gaussian kernel.

### Reparametrization trick

Now, from the relation $\mathbf{X}_k = \sqrt{\bar{\alpha}_{k}}\mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\mathbf{Z}}_k$, where $\bar{\mathbf{Z}}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we find that
```math
    \mathbf{x}_k = \sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k,
```
for a sample $\boldsymbol{\epsilon}_k$ of the standard normal distribution. We use that to rewrite $\mathbf{x}_0$ in terms of $\mathbf{x}_k$, i.e.
```math
    \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_{k}}}\mathbf{x}_k - \frac{\sqrt{1 - \bar{\alpha}_{k}}}{\sqrt{\bar{\alpha}_{k}}}\bar{\boldsymbol{\epsilon}}_k.
```
Plugging this into the formula for the mean $\tilde{\boldsymbol{\mu}}_k$, we obtain
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k & = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\left(\frac{1}{\sqrt{\bar{\alpha}_{k}}}\mathbf{x}_k - \frac{\sqrt{1 - \bar{\alpha}_{k}}}{\sqrt{\bar{\alpha}_{k}}}\bar{\boldsymbol{\epsilon}}_k\right) \\
        & = \left(\frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k} + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{(1 - \bar{\alpha}_k)\sqrt{\bar\alpha_k}}\right)\mathbf{x}_k - \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\frac{\sqrt{1 - \bar{\alpha}_{k}}}{\sqrt{\bar{\alpha}_{k}}}\bar{\boldsymbol{\epsilon}}_k \\
        & = \left(\frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k} + \frac{\beta_k}{(1 - \bar{\alpha}_k)\sqrt{\alpha_k}}\right)\mathbf{x}_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\bar{\boldsymbol{\epsilon}}_k \\
        & = \left(\frac{(1 - \bar{\alpha}_{k-1})\alpha_k + \beta_k}{(1 - \bar{\alpha}_k)\sqrt{\alpha_k}}\right)\mathbf{x}_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\bar{\boldsymbol{\epsilon}}_k \\
        & = \left(\frac{\alpha_k - \bar{\alpha}_k + \beta_k}{(1 - \bar{\alpha}_k)\sqrt{\alpha_k}}\right)\mathbf{x}_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\bar{\boldsymbol{\epsilon}}_k
    \end{align*}
```
Using, again, that $\beta_k = 1 - \alpha_k$, we find
```math
    \tilde{\boldsymbol{\mu}}_k = \frac{1}{\sqrt{\alpha_k}}\mathbf{x}_k - \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\bar{\boldsymbol{\epsilon}}_k.
```

### The model

We want to approximate the distribution of the Markov process with some model pdf $p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})$, which yields an approximation of the target pdf $p_0(\mathbf{x}_0)$ via
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0}) = \int_{(\mathbf{R}^d)^{K}} p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}) \;\mathrm{d}\mathbf{x}_{1:K}.
```
where
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}) = \int_{\mathbb{R}^d} p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}|\mathbf{x}_K)p_{\boldsymbol{\theta}}(\mathbf{x}_K)\;\mathrm{d}\mathbf{x}_K,
```
and with
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_K) \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
```

We have
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}|\mathbf{x}_K) = p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)p_{\boldsymbol{\theta}}(\mathbf{x}_1|\mathbf{x}_2)\cdots p_{\boldsymbol{\theta}}(\mathbf{x}_{K-1}|\mathbf{x}_K).
```


... at some point we use the empirical distribution
```math
    p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1}\right) \approx \frac{1}{N}\sum_{n=1}^N p_{\boldsymbol{\beta}}\left(\mathbf{x}_k|\mathbf{x}_{k-1},\mathbf{x}_0^n\right).
```
## Numerical example

We illustrate the method, numerically, to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

As usual, we use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages and set the seed for reproducibility purposes.

```@setup ddpmscorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation
using Markdown

nothing # hide
```

There are several Julia libraries for artificial neural networks and for automatic differentiation (AD). The most established package for artificial neural networks is the [FluxML/Flux.jl](https://github.com/FluxML/Flux.jl) library, which handles the parameters implicitly, but it is moving to explicit parameters. A newer library that handles the parameters explicitly is the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library, which is taylored to the differential equations [SciML](https://sciml.ai) ecosystem.

Since we aim to combine score-matching with neural networks and, eventually, with stochastic differential equations, we thought it was a reasonable idea to experiment with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library.

As we mentioned, the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library is a newer package and not as well developed. In particular, it seems the only AD that works with it is the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library. Unfortunately, the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library is not so much fit to do AD on top of AD, as one can see from e.g. [Zygote: Design limitations](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1). Thus we only illustrate this with a small network on a simple univariate problem.

#### Reproducibility

We set the random seed for reproducibility purposes.

```@setup ddpmscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup ddpmscorematching
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

We build the target model and draw samples from it.

The target model is a univariate random variable denoted by $X$ and defined by a probability distribution. Associated with that we consider its PDF and its score-function.

```@setup ddpmscorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sample_points = permutedims(rand(rng, target_prob, 1024))
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup ddpmscorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example ddpmscorematching
plt # hide
```

Visualizing the score function.
```@setup ddpmscorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example ddpmscorematching
plt # hide
```

```@example ddpmscorematching
G(x) = exp(-x^2 / 2) / √(2π)

psigma(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) for xn in sample_points) / sigma

score_parzen(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) * (xn - x) / sigma^2 for xn in sample_points) / psigma(x, sigma, sample_points) / sigma
```

```@example ddpmscorematching
sigma = 0.5
score_parzen_points = map(x -> score_parzen(x, sigma, sample_points), sample_points)
data = (sample_points, score_parzen_points)
```

### Markov chain

Now we evolve the sample as the initial state of a Markov chain $\{\mathbf{X}_k\}_{k=0, 1, \ldots, k_f}$, with

```math
    \mathbf{X}_{k+1} \sim \mathcal{N}(\sqrt{1 - \beta_k} \mathbf{X}_k, \beta_k \mathbf{I}),
```
where $\{\beta_k\}_{k=0}^{k_f}$ is a given schedule, which we take to be...

```@example ddpmscorematching

function ddpm_chain!(rng, xt, beta_schedule)
    @assert axes(xt, 1) == only(axes(beta_schedule))
    i1 = firstindex(axes(xt, 1))
    randn!(rng, xt)
    xt[i1, :] .= x0
    for i in Iterators.drop(axes(xt, 1), 1)
        xt[i, :] .= sqrt(1 - beta_schedule[i1]) .* view(xt, i1, :) .+ beta_schedule[i1] .* view(xt, i, :)
        i1 = i
    end
    return xt
end

function ddpm_chain(rng, x0, beta_schedule)
    xt = beta_schedule .* x0'
    ddpm_chain!(rng, xt, beta_schedule)
    return xt
end
```

```@example ddpmscorematching
beta_init = 0.1
beta_final = 0.5
beta_len = 40
beta_schedule = range(beta_init, beta_final, beta_len)
```

```@example ddpmscorematching
x0 = vec(sample_points)
```

```@example ddpmscorematching
xt = ddpm_chain(rng, x0, beta_schedule)
```

```@example ddpmscorematching
plot(xt)
```

### The neural network model

The neural network we consider is a again a feed-forward neural network made, but now it is a two-dimensional model, since it takes both the variate $x$ and the discrete time $n$, to account for the evolution of the Markov chain.

```@example ddpmscorematching
model = Chain(Dense(1 => 8, sigmoid), Dense(8 => 1))
```

We initialize the *parameters* and the *state* of the model.
```@example ddpmscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

Here it is how we implement the objective ${\tilde J}_{\mathrm{P_\sigma ESM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example ddpmscorematching
function loss_function_parzen(model, ps, st, data)
    sample_points, score_parzen_points = data
    y_score_pred, st = Lux.apply(model, sample_points, ps, st)
    loss = mean(abs2, y_score_pred .- score_parzen_points)
    return loss, st, ()
end
```

### Optimization setup

#### Optimization method

We use the Adam optimiser.

```@example ddpmscorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example ddpmscorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example ddpmscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example ddpmscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_parzen, data, tstate_org)
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example ddpmscorematching
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

Now we train the model with the objective function ${\tilde J}_{\mathrm{P_\sigma ESM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example ddpmscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_parzen, 500, 20, 125)
nothing # hide
```

### Results

Testing out the trained model.
```@example ddpmscorematching
y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example ddpmscorematching
plot(title="Fitting", titlefont=10)

plot!(xrange, target_score', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(xx', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup ddpmscorematching
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

```@example ddpmscorematching
gif(anim, fps = 20) # hide
```

Recovering the PDF of the distribution from the trained score function.
```@example ddpmscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(xrange, target_pdf', label="original")
plot!(xrange, pdf_pred', label="recoverd")
```

And the animation of the evolution of the PDF.
```@setup ddpmscorematching
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

```@example ddpmscorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example ddpmscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", ICML'15: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, 2256-2265](https://dl.acm.org/doi/10.5555/3045118.3045358)
1. [J. Ho, A. Jain, P. Abbeel (2020), "Denoising diffusion probabilistic models", in Advances in Neural Information Processing Systems 33, NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)