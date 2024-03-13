# Multiple denoising score matching with annealed Langevin dynamics

## Introduction

### Aim

Review the **multiple denoising score matching (MDSM),** or **denosing score matching with Langevin dynamics (SMLD),** which fits a **noise conditional score network (NCSN),** as introduced by [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), which together with DDPM was one step closer to the score-based SDE model.

### Background

After [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) suggested fitting the score function of a distribution, several directions were undertaken to improve the quality of the method and make it more practical.

One of the approaches was the *denoising score matching* of [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142), in which the data is corrupted by a Gaussian noise and the model was trained to correctly denoise the corrupted data. The model itself would either be of the pdf itself or of an energy potential for the pdf. In any case, one would have a model for the pdf and could draw samples directly using that.

[Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) came with two ideas tied together. The first idea was to model directly the score function and use the Langevin equation to draw samples from it. One difficulty with Langevin sampling, however, is in correctly estimating the weights of multimodal distributions, either superestimating or subestimating some modal regions, depending on where the initial distribution of points is located relative to the model regions. It may take a long time to reach the desired distribution.

In order to overcome that, [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) also proposed using an annealed version of Langevin dynamics, based on a scale of *denoising score matching* models, with different levels of noise, instead of a single denoising. Lower noises are closer to the target distribution but are challenging to the Langevin sampling, while higher noises are better for Langevin sampling but depart from the target distributions. Combining different levels of noise and gradually sampling between different denoising models improve the modeling and sampling of a distribution. That is the idea of their proposed **noise conditional score network (NCSN)** framework, in a method that was later denominated **denosing score matching with Langevin dynamics (SMLD),** and for which a more precise description would be **multiple denosing score matching with annealed Langevin dynamics,** or simply **multiple denoising score matching (MDSM).**

## Multiple denoising score matching 

The idea is to consider a sequence of denoising score matching models, starting with a relatively large noise level $\sigma_1$, to avoid the difficulties with Langevin sampling described earlier, and end up with a relatively small noise level $\sigma_L$, to minimize the noisy effect on the data.

For training, one trains directly a score model according to a weighted loss involving all noise levels.

Then, for sampling, a corresponding sequence of Langevin dynamics, with decreasing levels of noise, driving new samples closer and closer to the target distribution.

### The model

More precisely, one starts with a positive geometric sequence of noise levels $\sigma_1, \ldots, \sigma_L$ satisfying
```math
    \frac{\sigma_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1,
```
which is the same as
```math
    \sigma_i = \theta^i \sigma_1, \quad i = 1, \ldots, L,
```
for a starting $\sigma_1 > 0$ and a rate $0 < \theta < 1$ given by $\theta = \sigma_2/\sigma_1 = \ldots = \sigma_L/\sigma_{L-1}$.

For each $\sigma=\sigma_i$, $i=1, \ldots, L$, one considers the perturbed distributions
```math
    p_{\sigma}(\tilde{\mathbf{x}}) = \int_{\mathbb{R}^d} p(\mathbf{x})p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})\;\mathrm{d}\mathbf{x},
```
with a perturbation kernel
```math
    p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathbf{N}\left(\tilde\mathbf{x}; \mathbf{x}, \sigma^2 \mathbf{I}).
```
This yields a sequence of perturbed distributions
```math
    \{p_{\sigma_i}\}_{i=1}^L.
```

We model the corresponding family of score functions $\{s_{\boldsymbol{\theta}}(\tilde\mathbf{x}, \sigma)\}$, i.e. such that $s_{\boldsymbol{\theta}}(\tilde\mathbf{x}, \sigma_i)$ approximates the score function of $p_{\sigma_i}$, i.e.
```math
    s_{\boldsymbol{\theta}}(\tilde\mathbf{x}, \sigma_i) \approx \boldsymbol{\nabla}_{\tilde\mathbf{x}} \log p_{\sigma_i}(\tilde\mathbf{x}).
```

The **noise conditional score network (NCSN)** is precisely $s_{\boldsymbol{\theta}}(\tilde\mathbf{x}, \sigma).$

### The loss function

One wants to train the noise conditional score network (NCSN) $s_{\boldsymbol{\theta}}(\tilde\mathbf{x}, \sigma)$ by weighting together the denosing loss function of each perturbation, i.e.
```math
    J_{MDSM}(\boldsymbol{\theta}) = \frac{1}{2L}\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p(\mathbf{x})p_{\sigma_i}(\tilde\mathbf{x}|\mathbf{x})}\left[\left\| s_{\boldsymbol{\theta}}(\tilde\mathbf{x}, \sigma_i) - \frac{\mathbf{x} - \tilde\mathbf{x}}{\sigma_i^2} \right\|\right],
```
where $\lambda = \lambda(\sigma_i)$ is a weighting factor.


## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)