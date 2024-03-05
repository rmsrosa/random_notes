# Denoising diffusion probabilistic models

```@meta
Draft = false
```

## Introduction

### Aim

Review the **denoising diffusion probabilistic models (DDPM)** introduced in [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and further improved in [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html).

### Motivation

Build a solid foundation on score-based generative diffusion models, for which DDPM, although not initially seen as score-based, is related to, as a discretized analog of the SDE model.

### Background

The main idea in [Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and improved in [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) is to embed the random variable we want to model into a Markov chain and model the whole Markov chain. This is a much more complex task which greatly increases the dimension of the problem, but which yields more stability to both training and generative processes.

The desired random variable, for which we only have access to a sample, is considered as an initial condition to a Markov chain converging to a simple and tractable distribution, usually a normal distribution. The training process fits a model to the Markov chain up to a relatively large time step. Then, the model is used to reverse the process and generate (aproximate) samples of our target distribution from samples of the tractable distribution. The tractable asymptotic distribution of the forward process becomes the initial distribution of the model reverse process, and the (initial) desired target distribution is approximated by the final distribution of the reverse model process.

Besides the original articles [Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html),
another source, which in the beginning helped me understand the main ideas of the foundational articles, was the blog post [What are diffusion models? Lil’Log by Lilian Weng (2021)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

## Details of the method

We consider the unknown target distribution to be an initial distribution for a Markov process, so we denote it by $\mathbb{P}_0$, with the associated random variable denoted by $\mathbf{X}_0$ in $\mathbb{R}^d$, $d\in\mathbb{N}$. The probability density function is denoted by $p_0(\mathbf{x})$.

The idea is to define a forward Markov process to add noise to the initial random variable and drive the distribution close to a standard Gaussian distribution. This forward process is pretty well defined and straightforward to implement. From the initial sample points we obtain samples of the Markov process. The idea then is to use those sample trajectories to learn the reverse Markov process. With this model of the reverse process, we can build new sample points out of (new) samples of the standard Gaussian distribution.

Since the approximate reverse process is made of a initial standard Gaussian distribution and the process just adds Gaussian noises, the approximate reverse process is a Gaussian process. Thus we can just parametrized it by its (time-dependent) mean and variance. The problem is that ... 

### The forward Markov chain

The initial random variable $\mathbf{X}_0$ is evolved in time as a Markov chain $\{\mathbf{X}_k\}_{k\in \mathbb{Z}_0^+}$ according to
```math
    \mathbf{X}_k = \sqrt{1 - \beta_k}\mathbf{X}_{k-1} + \sqrt{\beta_k}\mathbf{Z}_k, \quad \mathbf{Z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
```
where $\boldsymbol{\beta}=\{\beta_k\}_{k\in \mathbb{N}}$ is given, with $0 < \beta_k < 1$, for every $k$. In practice, we will stop at some $K\in\mathbb{N}$, so that $k=1, 2, \ldots, K$.

The marginal probability density function of the step $k$ conditioned on the step $k-1$ satisfies
```math
    p(\mathbf{x}_k|\mathbf{x}_{k-1}) \sim \mathcal{N}\left(\sqrt{1 - \beta_k}\mathbf{x}_{k-1}, \beta_k\right).
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
    \left\|\mathbf{X}_k - {\bar{\mathbf{X}}}_k\right\|^2 = (1 - \beta_k)\left\|\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right\|^2 + 2\sqrt{1 - \beta_k}\left(\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right)\sqrt{\beta_k}\mathbf{Z}_k + \beta_k \mathbf{Z}_k^2.
```
Taking the expectation, we find that
```math
    \mathbb{E}\left[ \left\|\mathbf{X}_k - {\bar{\mathbf{X}}}_k\right\|^2 \right] = (1 - \beta_k) \mathbb{E}\left[ \left\|\mathbf{X}_{k-1} - {\bar{\mathbf{X}}}_{k-1}\right\|^2 \right] + \beta_k.
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

The probability density functions $p(\mathbf{x}_{0:K}) = p(\mathbf{x}_0, \ldots, \mathbf{x}_K)$ of the Markov chain, where $\mathbf{x}_{0:K} = (\mathbf{x}_0, \dots, \mathbf{x}_K)$ is a portion of a trajectory up to some sufficiently large step $K\in\mathbb{N}$, satisfies the conditional marginal relation
```math
    p(\mathbf{x}_k|\mathbf{x}_{k-1}) \sim \mathcal{N}(\sqrt{1 - \beta_k}\mathbf{x}_{k-1}, \beta_k),
```
Then,
```math
    p(\mathbf{x}_{0:K}|\mathbf{x}_0) = p(\mathbf{x}_{K}|\mathbf{x}_{K-1})\cdots p(\mathbf{x}_1|\mathbf{x}_0) = \prod_{k=1}^{K}p(\mathbf{x}_k|\mathbf{x}_{k-1}).
```

Thus,
```math
    p(\mathbf{x}_{0:K}) = \int_{\mathbb{R}^d} \prod_{k=1}^K p(\mathbf{x}_k|\mathbf{x}_{k-1})p_0(\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0.
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
    p\left(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0\right) = \frac{p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)p\left(\mathbf{x}_{k-1}|\mathbf{x}_0\right)}{p\left(\mathbf{x}_k|\mathbf{x}_0\right)}
```

Using the Markovian property on the first term of the nominator and ignoring the normalization constant, we know that
```math
    p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) = p\left(\mathbf{x}_k|\mathbf{x}_{k-1}\right) \propto \exp\left(-\frac{1}{2}\frac{\left(\mathbf{x}_k - \sqrt{\alpha_k}\mathbf{x}_{k-1}\right)^2}{\beta_k}\right),
```
while
```math
    p\left(\mathbf{x}_{k-1}|\mathbf{x}_0\right) \propto \exp\left(-\frac{1}{2}\frac{\left(\mathbf{x}_{k-1} - \sqrt{\bar{\alpha}_{k-1}}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_{k-1}}\right),
```
and
```math
    p\left(\mathbf{x}_k|\mathbf{x}_0\right) \propto \exp\left(-\frac{1}{2}\frac{\left(\mathbf{x}_k - \sqrt{\bar{\alpha}_k}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_k}\right).
```
Thus,
```math
    p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) \propto \exp\left( - \frac{1}{2}\left(\frac{\left(\mathbf{x}_k - \sqrt{\alpha_k}\mathbf{x}_{k-1}\right)^2}{\beta_k} + \frac{\left(\mathbf{x}_{k-1} - \sqrt{\bar{\alpha}_{k-1}}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_{k-1}} - \frac{\left(\mathbf{x}_k - \sqrt{\bar{\alpha}_k}\mathbf{x}_0\right)^2}{1 - \bar{\alpha}_k} \right)\right).
```
We separate the dependence on the variable $\mathbf{x}_{k-1}$ from that on the conditioned variables $\mathbf{x}_k$ and $\mathbf{x}_0$.
```math
    \begin{align*}
        p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) & \propto \exp\bigg( - \frac{1}{2}\bigg(\frac{\mathbf{x}_k^2 - 2\mathbf{x}_k \sqrt{\alpha_k}\mathbf{x}_{k-1} + \alpha_k\mathbf{x}_{k-1}^2}{\beta_k} \\
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
Using that $\beta_k = 1 - \alpha_k$, we find the variance of $p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)$ to be
```math
    \tilde\beta_k = \frac{1}{\left( \frac{\alpha_k}{\beta_k} + \frac{1}{1 - \bar{\alpha}_{k-1}}\right)} = \frac{\beta_k(1 - \bar{\alpha}_{k-1})}{\alpha_k(1 - \bar{\alpha}_{k-1}) + \beta_k} = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\beta_k,
```
With that, we rewrite the mean of $p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)$ as
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
    p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) \propto \exp\bigg( -\frac{1}{2}\bigg( \frac{\left(\mathbf{x}_{k-1} - \tilde{\boldsymbol{\mu}}_k\right)^2}{\tilde{\beta}_k} + \tilde\gamma_k \bigg)\bigg),
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
    p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right) = \mathcal{N}\left(\mathbf{x}_k; \tilde{\boldsymbol{\mu}}_k, \tilde \beta_k\mathbf{I}\right),
```
where $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \beta\mathbf{I})$ is the Gaussian kernel
```math
    \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \beta\mathbf{I}) = \frac{1}{\sqrt{2\pi\beta^d}}e^{-\frac{1}{2}\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{\beta}},
```
with $\mathcal{N}(\mathbf{x}; \mathbf{0}, \mathbf{I})$ being the standard Gaussian kernel.

With that, we can write
```math
    p\left(\mathbf{x}_k|\mathbf{x}_{k-1}\right) = \int_{\mathbb{R}^d} p\left(\mathbf{x}_k|\mathbf{x}_{k-1},\mathbf{x}_0\right)p(\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0.
```

Notice we can write the (initial) target distribution as
```math
    p_0(\mathbf{x}_0) = p(\mathbf{x}_0) = \int_{(\mathbf{R}^d)^{K}} p(\mathbf{x}_{0:K}) \;\mathrm{d}\mathbf{x}_{1:K},
```
and then
```math
    p(\mathbf{x}_{0:K}) = \int_{\mathbb{R}^d} p(\mathbf{x}_{0:K}|\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0,
```
with
```math
    p(\mathbf{x}_{0:K}|\mathbf{x}_0) = p(\mathbf{x}_0|\mathbf{x}_1, \mathbf{x}_0)p(\mathbf{x}_1|\mathbf{x}_2, \mathbf{x}_0)\cdots p(\mathbf{x}_{K-1}|\mathbf{x}_K, \mathbf{x}_0),
```
and
```math
    p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{k-1}; \tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0), \tilde \beta_k(\mathbf{x}_k, \mathbf{x}_0)\mathbf{I}).
```

### Reparametrization trick

Now, from the relation $\mathbf{X}_k = \sqrt{\bar{\alpha}_{k}}\mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\mathbf{Z}}_k$, where $\bar{\mathbf{Z}}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we find that
```math
    \mathbf{x}_k = \sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k,
```
for a sample $\boldsymbol{\epsilon}_k$ of the standard normal distribution. We use that to rewrite $\tilde{\boldsymbol{\mu}}_k$ in terms of $\mathbf{x}_0$ and $\boldsymbol{\epsilon}_k$,
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k & = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0 \\
        & = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\left(\sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k\right) + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0 \\
        & = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}\sqrt{\bar{\alpha}_{k}} + \beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0 + \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k \\
        & = \frac{((1 - \bar{\alpha}_{k-1})\alpha_k + \beta_k)\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0 + \frac{(1 - \bar{\alpha}_{k-1})\alpha_k}{\sqrt{1 - \bar{\alpha}_k}\sqrt{\alpha_k}}\bar{\boldsymbol{\epsilon}}_k \\
        & = \frac{(\alpha_k - \bar{\alpha}_k + \beta_k)\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0 + \frac{(1 - \bar{\alpha}_{k-1})\alpha_k}{\sqrt{1 - \bar{\alpha}_k}\sqrt{\alpha_k}}\bar{\boldsymbol{\epsilon}}_k \\
    \end{align*}
```
Using that $\beta_k = 1 - \alpha_k$, we find
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k & = \frac{(1 - \bar{\alpha}_k)\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0 + \frac{\alpha_k - \bar{\alpha}_k}{(1 - \bar{\alpha}_k)\sqrt{\alpha_k}}\sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k \\
        & = \sqrt{\bar{\alpha}_{k-1}}\mathbf{x}_0 + \frac{\alpha_k - \bar{\alpha}_k}{(1 - \bar{\alpha}_k)\sqrt{\alpha_k}}\sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k \\
    \end{align*}
```

We can also rewrite $\mathbf{x}_0$ in terms of $\mathbf{x}_k$, i.e.
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
In this way, we *reparametrize* the mean in terms of $\mathbf{x}_k$ and $\bar{\boldsymbol{\epsilon}}_k$, instead of $\mathbf{x}_k$ and $\mathbf{x}_0$.

### The model

We want to approximate the distribution of the Markov process with some model pdf $p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})$, which yields an approximation of the target pdf $p_0(\mathbf{x}_0)$ via
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0}) = \int_{(\mathbf{R}^d)^{K}} p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}) \;\mathrm{d}\mathbf{x}_{1:K}.
```

One of the key points in the forward Markov chain is  that the limit distribution of $\mathbf{X}_k$ as $k\rightarrow \infty$ is a standard normal distribution. Thus, for our model, we assume that the distribution at step $K$, with $K$ taken relatively large, is precisely a standard normal distribution. With that, the model is written as
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}) = \int_{\mathbb{R}^d} p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}|\mathbf{x}_K)p_{\boldsymbol{\theta}}(\mathbf{x}_K)\;\mathrm{d}\mathbf{x}_K,
```
with
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_K) = \mathcal{N}(\mathbf{x}_K; \mathbf{0}, \mathbf{I}).
```

We also have
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}|\mathbf{x}_K) = p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)p_{\boldsymbol{\theta}}(\mathbf{x}_1|\mathbf{x}_2)\cdots p_{\boldsymbol{\theta}}(\mathbf{x}_{K-1}|\mathbf{x}_K).
```

As in the reverse Markov process, we assume each $p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)$ is a Gaussian distribution. Hence, each conditional distribution is parametrized by its mean and its variance,
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k) = \mathcal{N}(\mathbf{x}_{k-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k), \beta_{\boldsymbol{\theta}}(\mathbf{x}_k, k)).
```

Due to the reparametrization trick used in the target Markov chain, we also reparametrize $\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k)$ in a similar way:
```math
    \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) = \frac{1}{\sqrt{\alpha_k}}\mathbf{x}_k - \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_k, k).
```

Although $\beta_{\boldsymbol{\theta}}(\mathbf{x}_k, k)$ are also learnable, they are set to constants, for the sake of simplicity of the loss function:
```math
    \beta_{\boldsymbol{\theta}}(\mathbf{x}_k, k) = \sigma_k,
```
for pre-determined constants $\sigma_k$, $k=1, \ldots, K$.

Actually, for the final step $p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)$ of the reverse process,  [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) bases it on the first step of the forward trajectory, in order to "remove the edge effect" (see Appendix B.2):
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1) = p(\mathbf{x}_1, \mathbf{x}_0)\frac{\mathcal{N}(\mathbf{x}_0; \mathbf{0}, \mathbf{I})}{\mathcal{N}(\mathbf{x}_1; \mathbf{0}, \mathbf{I})}.
```
In [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), however, this is setup differently, being truncated to the support of the original distribution, which is assumed to represent an image, with data in $\{0, 1, \ldots, 255\}$ scaled to $[-1, 1]$, i.e. each coordinate $x_i$, $i=1, \ldots, d$, in $\{(a - 127.5) / 127.5; \;a=0, \ldots, 255\}$, so that
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1) = \prod_{i=1}^d \int_{\delta_-(x_{0i})}^{\delta_+(x_{0i})} \mathcal{N}(x_i; \mu_{\boldsymbol{\theta}, i}(\mathbf{x}_1, 1), \sigma_1^2)\; \mathbf{x}_i,
```
with
```math
    \delta_-(x_{0i}) = \begin{cases}
        -\infty, & x = -1, \\
        x - 1/255, & x > -11,
    \end{cases}
    \qquad 
    \delta_+(x_{0i}) = \begin{cases}
        \infty, & x = 1, \\
        x + 1/255, & x < 1.
    \end{cases}
```

In any case, our model is completely defined by $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_k, k)$, $k=1, \ldots, K$, the parameters $\sigma_1, \ldots, \sigma_K$, and the (final) conditional distribution $p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)$.

### The loss function

Now we need a loss function to train the parametrizations $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_k, k)$ and $\beta_{\boldsymbol{\theta}}(\mathbf{x}_k, k)$ of our model.

#### The cross-entropy loss

Ideally, one would maximize the (log-)likelyhood of the model, by minimizing the **cross-entropy loss** function
```math
    L_{\mathrm{CE}}(\boldsymbol{\theta}) = H(p_0, p_{\boldsymbol{\theta}}) = \mathbb{E}_{p_0}\left[-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0)\right] = -\int_{\mathbb{R}^d} p_0(\mathbf{x}_0)\log p_{\boldsymbol{\theta}}(\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0 \approx -\frac{1}{N}\sum_{n=1}^N \log p_{\boldsymbol{\theta}}(\mathbf{x}_0^n).
```
But $p_{\boldsymbol{\theta}}(\mathbf{x}_{0})$, given as
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{0}) = \int_{(\mathbf{R}^d)^{K}} p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}) \;\mathrm{d}\mathbf{x}_{1:K},
```
is *intractable*.

#### The variational lower bound loss

We substitute for $p_{\boldsymbol{\theta}}(\mathbf{x}_{0})$ and multiply and divide by $p(\mathbf{x}_{1:K}|\mathbf{x}_0)$ to find 
```math
    \begin{align*}
        L_{\mathrm{CE}}(\boldsymbol{\theta}) & = \mathbb{E}_{p_0}\left[-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0)\right] \\
        & = -\int_{\mathbb{R}^d} p_0(\mathbf{x}_0)\log p_{\boldsymbol{\theta}}(\mathbf{x}_0)\;\mathrm{d}\mathbf{x}_0 \\
        & = -\int_{\mathbb{R}^d} p_0(\mathbf{x}_0)\log \left(\int_{(\mathbf{R}^d)^{K}} p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K}) \;\mathrm{d}\mathbf{x}_{1:K}\right)\mathrm{d}\mathbf{x}_0 \\
        & = -\int_{\mathbb{R}^d} p_0(\mathbf{x}_0)\log \left(\int_{(\mathbf{R}^d)^{K}} p(\mathbf{x}_{1:K}|\mathbf{x}_0) \frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})}{p(\mathbf{x}_{1:K}|\mathbf{x}_0)} \;\mathrm{d}\mathbf{x}_{1:K}\right)\mathrm{d}\mathbf{x}_0.
    \end{align*}
```
Now we use Jensen's inequality to obtain the following upper bound for the cross-entropy loss,
```math
    \begin{align*}
        L_{\mathrm{CE}}(\boldsymbol{\theta}) & \leq -\int_{\mathbb{R}^d} p_0(\mathbf{x}_0)\int_{(\mathbf{R}^d)^{K}} p(\mathbf{x}_{1:K}|\mathbf{x}_0) \log \left(\frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})}{p(\mathbf{x}_{1:K}|\mathbf{x}_0)} \right)\mathrm{d}\mathbf{x}_{1:K}\,\mathrm{d}\mathbf{x}_0 \\
        & = -\int_{(\mathbf{R}^d)^{K+1}} p(\mathbf{x}_{0:K}) \log \left(\frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})}{p(\mathbf{x}_{1:K}|\mathbf{x}_0)} \right)\mathrm{d}\mathbf{x}_{0:K} \\
        & = \int_{(\mathbf{R}^d)^{K+1}} p(\mathbf{x}_{0:K}) \log \left(\frac{p(\mathbf{x}_{1:K}|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})} \right)\mathrm{d}\mathbf{x}_{0:K}
    \end{align*}
```
This expression defines what is called the **variational lower bound** loss
```math
    L_{\mathrm{VLB}}(\boldsymbol{\theta}) = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{p(\mathbf{x}_{1:K}|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})} \right] = \int_{(\mathbf{R}^d)^{K+1}} p(\mathbf{x}_{0:K}) \log \left(\frac{p(\mathbf{x}_{1:K}|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})} \right)\mathrm{d}\mathbf{x}_{0:K}.
```

[Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) manipulated this loss to a more tractable form as follows
```math
    \begin{align*}
        L_{\mathrm{VLB}}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{p(\mathbf{x}_{1:K}|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:K})} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{\prod_{k=1}^K p(\mathbf{x}_k|\mathbf{x}_{k-1})}{p_{\boldsymbol{\theta}}(\mathbf{x}_K)\prod_{k=1}^K p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \prod_{k=1}^K \frac{p(\mathbf{x}_k|\mathbf{x}_{k-1})}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \sum_{k=1}^K \log \frac{p(\mathbf{x}_k|\mathbf{x}_{k-1})}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_k|\mathbf{x}_{k-1})}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right].
    \end{align*}
```
From Bayes' rule and the Markovian property of $\{X_k\}_k$ (as we derived earlier for $p\left(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0\right)$), we have
```math
    p\left(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0\right) = \frac{p\left(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{x}_0\right)p\left(\mathbf{x}_{k-1}|\mathbf{x}_0\right)}{p\left(\mathbf{x}_k|\mathbf{x}_0\right)} = \frac{p\left(\mathbf{x}_k|\mathbf{x}_{k-1}\right)p\left(\mathbf{x}_{k-1}|\mathbf{x}_0\right)}{p\left(\mathbf{x}_k|\mathbf{x}_0\right)},
```
which we can write as
```math
    p(\mathbf{x}_k|\mathbf{x}_{k-1}) = \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)p(\mathbf{x}_k|\mathbf{x}_0)}{p(\mathbf{x}_{k-1}|\mathbf{x}_0)}.
```
Hence,
```math
    \begin{align*}
        L_{\mathrm{VLB}}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_k|\mathbf{x}_{k-1})}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \frac{p(\mathbf{x}_k|\mathbf{x}_0)}{p(\mathbf{x}_{k-1}|\mathbf{x}_0)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_k|\mathbf{x}_0)}{p(\mathbf{x}_{k-1}|\mathbf{x}_0)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} + \log \prod_{k=2}^K  \frac{p(\mathbf{x}_k|\mathbf{x}_0)}{p(\mathbf{x}_{k-1}|\mathbf{x}_0)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \sum_{k=2}^K \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} + \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{p(\mathbf{x}_1|\mathbf{x}_0)} \right].
    \end{align*}
```
The first, second and fourth terms combine to yield
```math
    \begin{align*}
        -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_1|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} + \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{p(\mathbf{x}_1|\mathbf{x}_0)} & = -\log p_{\boldsymbol{\theta}}(\mathbf{x}_K) + \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)} \\
        & = \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_K)} - \log p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)
    \end{align*}
```
Thus,
```math
    \begin{align*}
        L_{\mathrm{VLB}}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ - \log p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1) + \sum_{k=2}^K \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} + \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_K)} \right].
    \end{align*}
```
This can be written as
```math
    L_{\mathrm{VLB}}(\boldsymbol{\theta}) = L_{\mathrm{VLB}, 0}(\boldsymbol{\theta}) + L_{\mathrm{VLB}, 1}(\boldsymbol{\theta}) + \cdots + L_{\mathrm{VLB}, K}(\boldsymbol{\theta}),
```
where
```math
    \begin{align*}
        L_{\mathrm{VLB, 0}}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ - \log p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1) \right], \\
        L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right], \quad k = 2, \ldots, K, \\
        L_{\mathrm{VLB}, K}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_K)} \right].
    \end{align*}
```
Notice the terms with $k>0$ involve Kullback-Leibler divergences.

In the model, the last marginal is taken to be a standard normal distribution, and hence this term is constant and has no parameter to learn:
```math
    L_{\mathrm{VLB}, K}(\boldsymbol{\theta}) = L_{\mathrm{VLB}, K} = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{p(\mathbf{x}_K|\mathbf{x}_0)}{\mathcal{N}(\mathbf{x}_K; \mathbf{0}, \mathbf{I})} \right].
```

Thus, the variational lower bound becomes
```math
    L_{\mathrm{VLB}}(\boldsymbol{\theta}) = L_{\mathrm{VLB}, 0}(\boldsymbol{\theta}) + L_{\mathrm{VLB}, 1}(\boldsymbol{\theta}) + \cdots + L_{\mathrm{VLB}, K}.
```

#### Simplifications

Since the last term in $L_{\mathrm{VLB}}(\boldsymbol{\theta})$ is constant, we only need to minimize 
```math
    L_{\mathrm{VLB}}^*(\boldsymbol{\theta}) = L_{\mathrm{VLB, 0}}(\boldsymbol{\theta}) + L_{\mathrm{VLB}, 1}(\boldsymbol{\theta}) + \cdots + L_{\mathrm{VLB}, K-1}(\boldsymbol{\theta}).
```

For $L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta})$, with $k=2, \ldots, K$, we use that
```math
    p\left(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0\right) = \mathcal{N}\left(\mathbf{x}_{k-1}; \tilde{\boldsymbol{\mu}}_k, \tilde \beta_k\mathbf{I}\right).
```
where
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k & = \tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0) = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0, \\
        \tilde\beta_k & = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\beta_k.
    \end{align*}
```

We have also modeled $p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)$ with
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k) = \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k), \sigma_k\mathbf{I}).
```

Moreover,
```math
    \begin{align*}
        L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta}) & = \mathbb{E}_{p(\mathbf{x}_{0:K})}\left[ \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_0, \mathbf{x}_k)p(\mathbf{x}_{k-1}|\mathbf{x}_0, \mathbf{x}_k)}\left[ \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right] \\
        & = \mathbb{E}_{p(\mathbf{x}_0, \mathbf{x}_k)}\left[\mathbb{E}_{p(\mathbf{x}_{k-1}|\mathbf{x}_0, \mathbf{x}_k)}\left[ \log \frac{p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0)}{p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)} \right]\right] \\
        & = \mathbb{E}_{p(\mathbf{x}_0, \mathbf{x}_k)} \left[D_{\mathrm{KL}}\left(p(\mathbf{x}_{k-1}|\mathbf{x}_k, \mathbf{x}_0) \| p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1}|\mathbf{x}_k)\right)\right].
    \end{align*}
```

The Kullback-Leibler divergence between two multivariate normals can be computed explicitly. In general, we have
```math
    D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \| \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)) = \frac{1}{2}\left( (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) \cdot \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)  + \operatorname{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) - \log \frac{\det\boldsymbol{\Sigma}_1}{\det\boldsymbol{\Sigma}_2} - d\right).
```
Thus,
```math
    \begin{align*}
        L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta}) & = \frac{1}{2}\mathbb{E}_{p(\mathbf{x}_0, \mathbf{x}_k)} \left[\frac{\|\tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) \|^2}{\sigma_k^2} + \frac{\tilde \beta_k}{\sigma_k^2} - \log\frac{{\tilde\beta_k}^d}{\sigma_k^{2d}} - d\right] \\
        & = \frac{1}{2}\mathbb{E}_{p(\mathbf{x}_0, \mathbf{x}_k)} \left[\frac{\|\tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) \|^2}{\sigma_k^2}\right] + C_k,
    \end{align*}
```
for a constant $C_k$ (with respect to the trainable parameters $\boldsymbol{\theta}$), where
```math
    \begin{align*}
        \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) & = \frac{1}{\sqrt{\alpha_k}}\mathbf{x}_k - \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) \\
        \tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0) & = \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k}\mathbf{x}_k + \frac{\beta_k\sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k}\mathbf{x}_0.
    \end{align*}
```
Thanks to the reparametrization,
```math
    \tilde{\boldsymbol{\mu}}_k = \frac{1}{\sqrt{\alpha_k}}\mathbf{x}_k - \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\bar{\boldsymbol{\epsilon}}_k.
```
Thus,
```math
    \begin{align*}
        \tilde{\boldsymbol{\mu}}_k(\mathbf{x}_k, \mathbf{x}_0) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) & = \frac{1}{\sqrt{\alpha_k}}\mathbf{x}_k - \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\bar{\boldsymbol{\epsilon}}_k - \frac{1}{\sqrt{\alpha_k}}\mathbf{x}_k + \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_k, k) \\
        & = \frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\left(\bar{\boldsymbol{\epsilon}}_k - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_k, k)\right).
    \end{align*}
```
Now we reparametrize the loss in terms of $\mathbf{x}_0$ and $\bar{\boldsymbol{\epsilon}}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, by writing $\mathbf{x}_k$ as
```math
    \mathbf{x}_k = \sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k.
```
With this reparametrization, the expectation also becomes in terms of $\mathbf{x}_0$ and $\bar{\boldsymbol{\epsilon}}_k$, so the loss becomes
```math
    L_{\mathrm{VLB}}^*(\boldsymbol{\theta}) = L_{\mathrm{VLB, 0}}(\boldsymbol{\theta}) + L_{\mathrm{VLB}, 1}(\boldsymbol{\theta}) + \cdots + L_{\mathrm{VLB}, K-1}(\boldsymbol{\theta})
```
with
```math
    L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta}) = \frac{1}{2\sigma_k^2}\frac{1-\alpha_k}{\sqrt{1 - \bar{\alpha}_{k}}\sqrt{\alpha_{k}}}\mathbb{E}_{p_0(\mathbf{x}_0)\mathcal{N}(\bar{\boldsymbol{\epsilon}}_k; \mathbf{0}, \mathbf{I})} \left[\left\|\bar{\boldsymbol{\epsilon}}_k - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k, k\right) \right\|^2\right],
```
for $k=1, \ldots, K$.

At this point, a stochastic gradient descent approach is taken, and instead of descend along the gradient of the sum of all $L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta})$, only one of them is randomly selected at each step, i.e. one considers
```math
    L_{\mathrm{VLB},\mathrm{unif}}^*(\boldsymbol{\theta}) = \mathbb{E}_{k \sim \operatorname{Uniform}(1, \ldots, K)}\left[ L_{\mathrm{VLB}, k-1}(\boldsymbol{\theta}) \right].
```

A further simplification proposed by [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), which was found to perform better in practice, is to simply drop the weighting term and minimize
```math
    L_{\mathrm{VLB}, k-1}^{\mathrm{simple}}(\boldsymbol{\theta}) = \mathbb{E}_{p_0(\mathbf{x}_0)\mathcal{N}(\bar{\boldsymbol{\epsilon}}_k; \mathbf{0}, \mathbf{I})} \left[\left\|\bar{\boldsymbol{\epsilon}}_k - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k, k\right) \right\|^2\right],
```
yielding the loss
```math
    L_{\mathrm{VLB,unif}}^{\mathrm{simple}, *}(\boldsymbol{\theta}) = \mathbb{E}_{k \sim \operatorname{Uniform}(1, \ldots, K),p_0(\mathbf{x}_0)\mathcal{N}(\bar{\boldsymbol{\epsilon}}_k; \mathbf{0}, \mathbf{I})} \left[\left\|\bar{\boldsymbol{\epsilon}}_k - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\bar{\alpha}_{k}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{k}}\bar{\boldsymbol{\epsilon}}_k, k\right) \right\|^2\right].
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

We build the target model, draw samples from it, and prepare all the parameters for training.

```@example ddpmscorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sample_points = permutedims(rand(rng, target_prob, 1024))
```

```@example ddpmscorematching
beta_init = 0.02
beta_final = 0.4
beta_len = 40
beta_schedule = range(beta_init, beta_final, beta_len)
alpha_schedule = 1 .- beta_schedule
alpha_tilde = cumprod(alpha_schedule)
coeffs = (
    krange=1:beta_len,
    sqrtx = map(√, alpha_tilde),
    sqrt1mx = map(x -> √(1 - x), alpha_tilde)
)
data = (sample_points, coeffs)
```

```@setup ddpmscorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example ddpmscorematching
plt # hide
```

```@setup ddpmscorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example ddpmscorematching
plt # hide
```

### Markov chain

Now we evolve the sample as the initial state of a Markov chain $\{\mathbf{X}_k\}_{k=1, \ldots, K}$, with

```math
    \mathbf{X}_k \sim \mathcal{N}(\sqrt{1 - \beta_k} \mathbf{X}_1, \beta_k \mathbf{I}),
```
where $\{\beta_k\}_{k=1}^{K}$ is a given schedule.

```@example ddpmscorematching
Markdown.parse("""We choose the schedule to be a linear schedule from ``\\beta_1 = $beta_init`` to ``\\beta_K = $beta_final`` in ``K = $beta_len`` steps.""") # hide
```

```@setup ddpmscorematching
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

```@setup ddpmscorematching
x0 = vec(sample_points)
```

```@setup ddpmscorematching
xt = ddpm_chain(rng, x0, beta_schedule)
```

```@example ddpmscorematching
plot(xt, label=nothing, title="Sample paths of the Markov diffusion", titlefont=10) # hide
```

The final histogram and the asymptotic standard normal distribution.

```@setup ddpmscorematching
plt = plot(title="PDF and histogram of the chain state at \$K=$beta_len", titlefont=10)
histogram!(plt, xt[end, :], normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, x -> pdf(Normal(), x), linewidth=4, label="pdf")
```

### The neural network model

The neural network we consider is a again a feed-forward neural network made, but now it is a two-dimensional model, since it takes both the variate $x$ and the discrete time $k$, to account for the evolution of the Markov chain.

```@example ddpmscorematching
model = Chain(Dense(2 => 32, relu), Dense(32 => 32, relu), Dense(32 => 1))
```

We initialize the *parameters* and the *state* of the model.
```@example ddpmscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

Here it is how we implement the objective $L_{\mathrm{VLB,unif}}^{\mathrm{simple}, *}(\boldsymbol{\theta})$.
```@example ddpmscorematching
function loss_function_uniform_simple(model, ps, st, data)
    sample_points, coeffs = data
    epsilons = randn(size(sample_points))
    ks = rand(coeffs.krange, size(sample_points))
    model_input = [coeffs.sqrtx[ks] .* sample_points .+ coeffs.sqrt1mx[ks] .* epsilons; ks]
    epsilons_pred, st = Lux.apply(model, model_input, ps, st)
    loss = mean(abs2, epsilons_pred .- epsilons)
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
Lux.Training.compute_gradients(vjp_rule, loss_function_uniform_simple, data, tstate_org)
```

#### Training loop

We repeat the usual training loop considered in the previous notes.

```@setup ddpmscorematching
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
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_uniform_simple, 5000, 20, 125)
nothing # hide
```

### Results

## References

1. [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", ICML'15: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, 2256-2265](https://dl.acm.org/doi/10.5555/3045118.3045358)
1. [J. Ho, A. Jain, P. Abbeel (2020), "Denoising diffusion probabilistic models", in Advances in Neural Information Processing Systems 33, NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
1. [L. Weng (2021), "What are diffusion models?" Lil’Log. lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)