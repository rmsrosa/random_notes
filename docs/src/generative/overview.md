# Overview of score-based methods

Here we review the main results associated with score-based generative methods.

The use of Langevin equations to draw samples of a *known* distribution from its score function was proposed by [Roberts and Tweedie (1996)](https://doi.org/10.2307/3318418).

The idea of *modeling* the score function from samples of a distribution was then proposed by [Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), via **implicit score matching**. The implicit score matching is obtained via integration by parts of the **explicit score matching** objective, which requires the unknown target score function. The implicit score function, however, requires computing the gradient of the modeled score function, which works fine for some specific models for which the derivatives can be readily computed, as considered by [Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), but otherwise might be computationally intense.

[Kingma and LeCun (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html) used automatic differentiation of certain neural network topologies to compute the gradient of the modeled score function and proposed regularizing the **implicit score matching** objective for stability purposes. But optimizing this requires computing the gradient of the loss function, which then requires computing the hessian of the modeled score function. Although doable, this is computationally expensive.

[Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) proposed working with the explicit score matching and using Parzen kernel density estimation to approximate the gradient of the desired score function, avoiding the need to differentiate the modeled score function.

[Pang, Xu, Li, Song, Ermon, and Zhu (2020)](https://openreview.net/forum?id=LVRoKppWczk) proposed using finite differences to approximate the gradient of the modeled score function of the **implicit score matching** objective, greatly expediting the optimization process.

## References

1. [G. O. Roberts, R. L. Tweedie (1996), "Exponential Convergence of Langevin Distributions and Their Discrete Approximations", Bernoulli, Vol. 2, No. 4, 341-363, doi:10.2307/3318418](https://doi.org/10.2307/3318418)
1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Durk P. Kingma, Yann Cun (2010), "Regularized estimation of image statistics by Score Matching", Advances in Neural Information Processing Systems 23 (NIPS 2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [T. Pang, K. Xu, C. Li, Y. Song, S. Ermon, J. Zhu (2020), Efficient Learning of Generative Models via Finite-Difference Score Matching, NeurIPS](https://openreview.net/forum?id=LVRoKppWczk) - see also the [arxiv version](https://arxiv.org/abs/2007.03317)