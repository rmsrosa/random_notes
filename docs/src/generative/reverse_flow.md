# Reverse probability flow

## Aim

Review the reverse probability flow used for sampling, after the Stein score function has been trained, as developed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, and Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html), based on the probability flow ODE developed in these articles and on the reverse time diffusion equation model previously worked out by [Anderson (1982)](https://doi.org/10.1016/0304-4149(82)90051-5).

## Reverse ODE

For an ODE of the form
```math
    \frac{\mathrm{d}x}{\mathrm{d}t} = f(t, x),
```
reverting time, on a time interval $[0, T],$ is just a matter of decreasing $t,$ from $T$ to $0.$ One way to think of it is via the integral formula
```math
    x(T) = x(t) + \int_t^T f(s, x(s))\;\mathrm{d}s,
```
so that
```math
    x(t) = x(T) - \int_t^T f(s, x(s))\;\mathrm{d}s.
```

Another way is to write ${\tilde x}(\tilde t\,) = x(T - \tilde t\,)$ and use the chain rule
```math
    \frac{\mathrm{d}{\tilde x}(\tilde t\,)}{\mathrm{d}\tilde t} = -\frac{\mathrm{d}x}{\mathrm{d}t}(T - \tilde t\,) = - f(T-\tilde t, x(T-\tilde t\,)) = -f(T-\tilde t, {\tilde x}(\tilde t\,)).
```
Integrating from $0$ to $T$ yields an integral relation equivalent to the previous one. In fact,
```math
    {\tilde x}(\tilde t\,) = {\tilde x}(0) - \int_0^{\tilde t} f(T-\tau, {\tilde x}(\tau)) \;\mathrm{d}\tau.
```
Going back to $x(\cdot)$ and making the change of variables $s = T - \tau,$ 
```math
    x(T - \tilde t\,) = x(T) - \int_0^T f(T-\tau, x(T-\tau))\;\mathrm{d}\tau = x(T) + \int_T^{T-\tilde t} f(s, x(s))\;\mathrm{d}s.
```
Back to $t = T - \tilde t$ yields
```math
    x(t) = x(T) - \int_t^T f(s, x(s))\;\mathrm{d}s.
```

The Euler method for the reverse flow is simply stepping backward from $t$ to $t - \Delta t,$ with the Taylor approximation reading
```math
    x(t_j) = x(t_{j+1}) - f(t_{j+1}, x(t_{j+1}))\Delta t,
```
with
```math
    t_j = T - j\Delta t,
```
so that $t_0 = T$ and $t_n = 0,$ for
```math
    \Delta t = T / n,
```
and $n\in\mathbb{N}$ given.

If the initial condition is a random variable $X_0,$ and the flow evolves to $X_T,$ then the reverse flow evolves back to $X_0.$ By approximating $X_T \sim Y_T$ by another random variable $Y_T,$ say a standard normal distribution, then the reverse flow evolves back towards an approximation $Y_0$ of the initial distribution $X_0.$

## Reverse Itô diffusion

Consider now a forward evolution given by an Itô diffusion SDE
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + g(t, X_t)\;\mathrm{d}W_t,
```
with initial distribution $X_0$ at $t = 0.$


## References

1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)
3. [B. D. O. Anderson (1982). Reverse-time diffusion equation models, Stochastic Process. Appl., vol. 12, no. 3, 313–326, DOI: 10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)