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

We should remark that this is a pathwise reversion, i.e. the same path can be traced back and forth with these ordinary differential equations. This is in contrast with what happens with in the stochastic case.

## Reverse Itô diffusion

Consider now a forward evolution given by an Itô diffusion SDE
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + g(t, X_t)\;\mathrm{d}W_t,
```
with initial distribution $X_0$ at $t = 0.$ In this case, we will not find a reverse equation tracing back a given sample path $X_t(\omega).$ Instead, we obtain a reverse SDE generating the same probability distribution. This will require knowledge of the Stein score function, which is not a problem in the use case we have in mind, where the Stein score is properly modeled.

The original way of obtaining the reverse SDE, derived in [Anderson (1982)](https://doi.org/10.1016/0304-4149(82)90051-5), and seen in other works, is by looking at the joint distribution $p(t, x_t, s, x_s)$ at two different times $t$ and $s$ and by working with conditional distributions. We do it differently here, though. We look at the connection between the SDE and the probability flow, introduced by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) and generalized by [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

For the stochastic differential equation above, with diagonal noise, the probability flow ODE takes the form
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{1}{2} \nabla_y g(t, Y_t)^2 - g(t, Y_t)^2\nabla_y \log p(t, Y_t).
```
Both $\{X_t\}_t$ and $\{Y_t\}_t$ have the same probability distribution $p(t, \cdot).$

We first write the reverse ODE in terms of $\tilde Y_{\tilde t} = Y_{T - \tilde t},$ in the reverse time variable $\tilde t = T - t.$ We have
```math
    \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} = - f(T - \tilde t, Y_{T - \tilde t}) + \frac{1}{2} \nabla_y g(T - \tilde t, Y_{T - \tilde t})^2 + g(T - \tilde t, Y_{T - \tilde t})^2\nabla_y \log p(T - \tilde t, Y_{T - \tilde t}),
```
i.e.
```math
    \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} = - f(T - \tilde t, {\tilde Y}_{\tilde t}) + \frac{1}{2} \nabla_y g(T - \tilde t, {\tilde Y}_{\tilde t})^2 + g(T - \tilde t, {\tilde Y}_{\tilde t})^2\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}).
```
Now we add and subtract $\nabla_y g(T - \tilde t, {\tilde Y}_{\tilde t})^2,$ to find
```math
    \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} = - f(T - \tilde t, {\tilde Y}_{\tilde t}) - \frac{1}{2} \nabla_y g(T - \tilde t, {\tilde Y}_{\tilde t})^2 + \nabla_y g(T - \tilde t, {\tilde Y}_{\tilde t})^2 + g(T - \tilde t, {\tilde Y}_{\tilde t})^2\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}).
```

With the proper sign for the second term on the right hand side, we see this is the probability flow equation for the associated SDE
```math
    \mathrm{d}{\tilde X}_{\tilde t} = \left(- f(T - \tilde t, {\tilde X}_{\tilde t}) + \nabla_x g(T - \tilde t, {\tilde X}_{\tilde t})^2 + g(T - \tilde t, {\tilde X}_{\tilde t})^2\nabla_x \log p(T - \tilde t, {\tilde X}_{\tilde t})\right)\;\mathrm{d}{\tilde t} + g(T - \tilde t, {\tilde X}_{\tilde t})\;\mathrm{d}{\tilde W}_{\tilde t},
```
where
```math
    {\tilde W}_{\tilde t} = W_{T - \tilde t}
```
is a reverse Wiener process. Back to the original variables $t = T - \tilde t$ and $X_t = {\tilde X}_{\tilde t},$
```math
    \mathrm{d}X_t = \left(f(t, X_t) - \nabla_x g(t, X_t)^2 - g(t, X_t)^2\nabla_x \log p(t, X_t)\right)\;\mathrm{d}t + g(t, X_t)\;\mathrm{d}W_{T-t}.
```




## References

1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)
3. [B. D. O. Anderson (1982). Reverse-time diffusion equation models, Stochastic Process. Appl., vol. 12, no. 3, 313–326, DOI: 10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)