# Noise conditional score network

## Introduction

### Aim

Review the **noise conditional score network (NCSN)** introduced by [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), which together with DDPM was one step closer to the score-based SDE model.

### Background

After [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) suggested fitting the score function, several directions were undertaken to improve the method or make it more practical. One of the approaches was the *denoising score matching* of [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142), in which the data is corrupted by a Gaussian noise and the model was trained to correctly denoise the corrupted data. The model itself would either be of the pdf itself or of an energy potential for the pdf. In any case, one would have a model for the pdf and could draw samples directly using that.

[Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) proposed modeling directly the score function and use the Langevin equation to draw samples from it. One difficulty with Langevin sampling is in correctly estimating the weights of multimodal distributions, either superestimating or subestimating some modal regions, depending on where the initial distribution of points is located relative to the model regions.

In order to overcome that, [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) proposed using an annealed version of Langevin dynamics, based on a scale of *denoising score matching* models, with different levels of noise. Lower noises are closer to the target distributions but are challenging to the Langevin sampling, while higher noises depart from the target distributions but are better for Langeving sampling. Combining different levels of noise and gradually sampling between different denoising models improve the modeling and sampling of a distribution. That is the idea of their proposed **noise conditional score network (NCSN)** method.

## Modeling, training and sampling NCSN

### The model

One starts with a positive geometric sequence of noise levels $\sigma_1, \ldots, \sigma_L$ satisfying
```math
    \frac{\sigma_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1,
```
which is the same as
```math
    \sigma_i = \theta^i \sigma_1, \quad i = 1, \ldots, L,
```
for a starting $\sigma_1 > 0$ and a rate $0 < \theta < 1$ given by $\theta = \sigma_2/\sigma_1 = \ldots = \sigma_L/\sigma_{L-1}$.

The idea is to start with a relatively large $\sigma_1$ to avoid the difficulties with Langevin sampling described earlier and end up with a relatively small $\sigma_L$ to minimize the noisy effect on the data.

Then one considers a corresponding sequence of score-matching denoising models
```math
    p_\sigma
```



## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)