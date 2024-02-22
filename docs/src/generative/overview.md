# Overview of the main results pertaining to score-based generative methods

Here we review the main results associated with score-based generative methods.

## Main results

The Langevin equation dates back to the beginning of the 20th Century, as proposed by [Langevin (1908)](https://gallica.bnf.fr/ark:/12148/bpt6k3100t/f530.item), in relation to the Brownian motion.

The use of the Langevin equation (and especially of the overdamped Langevin equation) to draw samples of a *known* distribution from its score function was proposed by [Roberts and Tweedie (1996)](https://doi.org/10.2307/3318418). Rates of convergence were also given and are still an important area of research.

The idea of *modeling* the score function from samples of a distribution was then proposed by [Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), via **implicit score matching**. The implicit score matching is obtained via integration by parts of the **explicit score matching** objective, which requires the unknown target score function. The implicit score function, however, requires computing the gradient of the modeled score function, which works fine for some specific models for which the derivatives can be readily computed, as considered by [Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), but otherwise might be computationally intensive.

[Kingma and LeCun (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html) used automatic differentiation of certain neural network topologies to compute the gradient of the modeled score function and proposed a **regularized implicit score matching** objective for stability purposes, by adding a penalization on the gradient of the model score function. This penalization does not add much to the complexity of the method, but it is nevertheless a computationally demanding method, since it is based on the implicit score matching and needs the gradient of the score function. This computational cost also increases dramatically with the dimension of the problem.

[Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) proposed the **denoising score matching** method, by working with the explicit score matching and using Parzen kernel density estimation to approximate the gradient of the desired score function, avoiding the need to differentiate the modeled score function. This is also interpreted in connection with the denoising auto-encoder methods.

[Song, Garg, Shi, and Ermon (2020)](https://proceedings.mlr.press/v115/song20a.html) addressed again the implicit score matching approach and proposed a **sliced (implicit) score matching** method to reduce the computational cost of the implicit score matching for high-dimensional problems. The trick is to take derivatives only at certain random directions, at each sample point.

A few months later, [Pang, Xu, Li, Song, Ermon, and Zhu (2020)](https://openreview.net/forum?id=LVRoKppWczk) proposed the **finite-difference (implicit) score matching** method, using finite differences to approximate the gradient of the modeled score function of the implicit score matching objective, greatly reducing the computational cost of the optimization process.

## References

1. [P. Langevin (1908), "Sur la théorie du mouvement brownien [On the Theory of Brownian Motion]". C. R. Acad. Sci. Paris. 146: 530–533](https://gallica.bnf.fr/ark:/12148/bpt6k3100t/f530.item)
1. [G. O. Roberts, R. L. Tweedie (1996), "Exponential Convergence of Langevin Distributions and Their Discrete Approximations", Bernoulli, Vol. 2, No. 4, 341-363, doi:10.2307/3318418](https://doi.org/10.2307/3318418)
1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Durk P. Kingma, Yann Cun (2010), "Regularized estimation of image statistics by Score Matching", Advances in Neural Information Processing Systems 23 (NIPS 2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Y. Song, S. Garg, J. Shi, S. Ermon (2020), Sliced Score Matching: A Scalable Approach to Density and Score Estimation, Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, PMLR 115:574-584](https://proceedings.mlr.press/v115/song20a.html) -- see also the [arxiv version](https://arxiv.org/abs/1905.07088)
1. [T. Pang, K. Xu, C. Li, Y. Song, S. Ermon, J. Zhu (2020), Efficient Learning of Generative Models via Finite-Difference Score Matching, NeurIPS](https://openreview.net/forum?id=LVRoKppWczk) - see also the [arxiv version](https://arxiv.org/abs/2007.03317)