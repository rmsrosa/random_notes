# Probability flow

## Aim

The aim is to review the probability flow sampling method, introduced by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) and generalized by [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

## Probability flow ODE for an Itô diffusion with scalar noise

This is the original result given by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802). The original SDE is the Itô diffusion with scalar noise term
$$
\mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + g(t, X_t)\;\mathrm{d}W_t,
$$
where the unknown $\{X_t\}_t$ is a scalar or vector valued process, $X_t\in \mathbb{R}^d,$ $d\in\mathbb{R},$ and $\{W_t\}_t$ is a Wiener process with the same dimension as the unknown, with independent components. The functions take the form $f:I\times \mathbb{R}^d \rightarrow \mathbb{R}^d$ and $g:I\times\mathbb{R}^d\rightarrow \mathbb{R},$ where $I\subset\mathbb{R}$ is an interval. In this case,  

## Probability flow ODE for a general Itô diffusion

## Generalized probability flow SDE for a general Itô diffusion

## References

1. [D. Maoutsa, S. Reich, M. Opper (2020), "Interacting particle solutions of Fokker-Planck equations through gradient-log-density estimation", Entropy, 22(8), 802, DOI: 10.3390/e22080802](https://doi.org/10.3390/e22080802)
1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)