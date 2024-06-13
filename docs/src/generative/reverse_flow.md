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

We remark that this is a *pathwise reversion,* meaning that each forward path $x(t)$ with initial condition $x(0)$ is traced back by the reverse equation starting at the final point $x(T).$ This is in contrast with the result for SDEs, for which, in general, only the probability distribution is recovered with the backward flow, not necessarily the individual samples paths. In order to trace back the exact forward paths, a specific Wiener process must be used.

## Reverse Itô diffusion

Consider now a forward evolution given by an Itô diffusion SDE
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
where the drift factor is a vector-valued function $f:I\times \mathbb{R}^d \rightarrow \mathbb{R}^d$, and the diffusion factor is a matrix-valued, time-dependent function $G:I\times \mathbb{R}^d \rightarrow \mathbb{R}^{d\times d}.$

In the following proof, we cannot deduce that this reverse equation traces back a given sample path $X_t(\omega),$ as in the ODE case. Instead, we only obtain that the reverse SDE generates the same probability distribution as the forward SDE.

Notice the reverse diffusion equation requires knowledge of the Stein score function, which fortunately is not a problem in the use case we have in mind, where the Stein score is properly modeled.

The original way of obtaining the reverse SDE, derived in [Anderson (1982)](https://doi.org/10.1016/0304-4149(82)90051-5), and seen in other works, is by looking at the joint distribution $p(t, x_t, s, x_s)$ at two different times $t$ and $s$ and by working with conditional distributions. We do it differently here, though. We look at the connection between the SDE and the probability flow, introduced by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) and generalized by [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

For the stochastic differential equation above, the probability flow ODE obtained by [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html) reads (except for the symbol $\{Y_t\}_t$ instead of $\{X_t\}_t$) 
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{1}{2} \nabla_x \cdot ( G(t, Y_t)G(t, Y_t)^{\mathrm{tr}} ) - \frac{1}{2} G(t, Y_t)G(t, Y_t)^{\mathrm{tr}}\nabla_x \log p(t, Y_t).
```
Both $\{X_t\}_t$ and $\{Y_t\}_t$ have the same probability distribution $p(t, \cdot).$

We now write the reverse ODE by making the change of variables $\tilde Y_{\tilde t} = Y_{T - \tilde t},$ with the reverse time variable $\tilde t = T - t.$ It is just an ODE (pathwise), so the reverse equation is straightforward chain rule
```math
    \begin{align*}
        \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} = - \frac{\mathrm{d}Y_{T - \tilde t}}{\mathrm{d}\tilde t} & = - f(T - \tilde t, Y_{T - \tilde t}) + \frac{1}{2} \nabla_y \cdot ( G(T - \tilde t, Y_{T - \tilde t})G(T - \tilde t, Y_{T - \tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + \frac{1}{2} G(T - \tilde t, Y_{T - \tilde t})G(T - \tilde t, Y_{T - \tilde t})^{\mathrm{tr}}\nabla_y \log p(T - \tilde t, Y_{T - \tilde t}),
    \end{align*}
```
i.e.
```math
    \begin{align*}
        \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} & = - f(T - \tilde t, {\tilde Y}_{\tilde t}) + \frac{1}{2} \nabla_y \cdot ( G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + \frac{1}{2} G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}}\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}).
    \end{align*}
```
The terms with $GG^{\mathrm{tr}}$ don't come with the right sign, so we just rewrite it as (like adding and subtracting the same terms)
```math
    \begin{align*}
        \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} & = - f(T - \tilde t, {\tilde Y}_{\tilde t}) + \nabla_y \cdot ( G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}}\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}) \\
        & \qquad \qquad - \frac{1}{2} \nabla_y \cdot ( G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad - \frac{1}{2} G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}}\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}).
    \end{align*}
```

With the proper sign, the last two terms on the right hand side become the diffusion term in the associated SDE for which this is the probability flow equation, namely
```math
    \begin{align*}
        \frac{\mathrm{d}{\tilde X}_{\tilde t}}{\mathrm{d}\tilde t} & = \bigg( - f(T - \tilde t, {\tilde X}_{\tilde t}) + \nabla_x \cdot ( G(T - \tilde t, {\tilde X}_{\tilde t})G(T - \tilde t, {\tilde X}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde t, {\tilde X}_{\tilde t})G(T - \tilde t, {\tilde X}_{\tilde t})^{\mathrm{tr}}\nabla_x \log p(T - \tilde t, {\tilde X}_{\tilde t}) \bigg) \;\mathrm{d}\tilde t\\
        & \qquad \qquad \qquad + G(T - \tilde t, {\tilde X}_{\tilde t})\;\mathrm{d}{\tilde W}_{\tilde t},
    \end{align*}
```
where $\{{\tilde W}_{\tilde t}\}_{\tilde t}$ is a (possibly different) Wiener process. In integral form, the equation for ${\tilde X}_{\tilde t},$ integrating from $\tilde \tau = 0$ to $\tilde \tau = T - \tilde t,$ reads
```math
    \begin{align*}
        {\tilde X}_{\tilde t} - {\tilde X}_0 & = \int_0^{T - \tilde t} \bigg( - f(T - \tilde \tau, {\tilde X}_{\tilde \tau}) + \nabla_x \cdot ( G(T - \tilde \tau, {\tilde X}_{\tilde \tau})G(T - \tilde \tau, {\tilde X}_{\tilde \tau})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde \tau, {\tilde X}_{\tilde \tau})G(T - \tilde \tau, {\tilde X}_{\tilde \tau})^{\mathrm{tr}}\nabla_x \log p(T - \tilde \tau, {\tilde X}_{\tilde \tau}) \bigg) \;\mathrm{d}\tilde \tau\\
        & \qquad \qquad \qquad + \int_0^{T - \tilde t} G(T - \tilde \tau, {\tilde X}_{\tilde \tau})\;\mathrm{d}{\tilde W}_{\tilde \tau},
    \end{align*}
```

Back to the original variables $t = T - \tilde t$ and $X_t = {\tilde X}_{\tilde t},$ this becomes
```math
    \begin{align*}
        X_t - X_T & = -\int_T^{t} \bigg( - f(\tau, X_\tau) + \nabla_x \cdot ( G(\tau, X_\tau)G(\tau, X_\tau)^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(\tau, X_\tau)G(\tau, X_\tau)^{\mathrm{tr}}\nabla_x \log p(\tau, X_\tau) \bigg) \;\mathrm{d}\tau\\
        & \qquad \qquad \qquad - \int_T^{t} G(\tau, X_\tau)\mathrm{d}{\tilde W}_{T-\tau},
    \end{align*}
```
which can be written as
```math
    \begin{align*}
        X_t - X_T & = \int_{t}^{T} \bigg( - f(\tau, X_\tau) + \nabla_x \cdot ( G(\tau, X_\tau)G(\tau, X_\tau)^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(\tau, X_\tau)G(\tau, X_\tau)^{\mathrm{tr}}\nabla_x \log p(\tau, X_\tau) \bigg) \;\mathrm{d}\tilde \tau\\
        & \qquad \qquad \qquad + \int_{t}^T G(\tau, X_\tau)\mathrm{d}{\tilde W}_{T - \tau},
    \end{align*}
```
where
```math
    \int_{T-t}^T H_\tau \mathrm{d}{\tilde W}_{T - \tau}
```
is a *reverse Itô integral,* with the integrand, in the approximating summations, computed at the rightmost point of each mesh interval. Since $\{{\tilde W}_{T - \tau}\}_{\tau}$ is a reverse Wiener process, this integral is well defined and is essentially the Itô integral rephrased backwards. Let us examine this more carefully. We start with the Itô integral
```math
    \int_0^{T - \tilde t} H_{T-\tilde \tau}\;\mathrm{d}{\tilde W}_{\tilde \tau},
```
defined for any given (non-antecipative) process $\{H_{T-\tilde\tau}\}_{\tilde \tau \geq 0}.$ It is given as the limit, as the mesh $0 = \tilde \tau_0 < \tilde \tau_1 < \ldots < \tilde \tau_n = T - \tilde t$ is refined, of the sums
```math
    \sum_{j=1}^n H_{T-\tilde \tau_{j-1}}({\tilde W}_{{\tilde \tau}_j} - {\tilde W}_{{\tilde \tau}_{j-1}}),
```
we see that the points $\tau_j = T - \tilde \tau_j$ form a mesh $T - t = \tau_n = T - \tilde \tau_n < \ldots < T - \tilde \tau_1 < T - \tilde \tau_0 = \tau_0 = T,$ with points $\tau_j = T - \tilde \tau_j$ decreasing in $j,$ and the summation can be written as
```math
    -\sum_{j=1}^n H_{T - \tau_{j-1}}({\tilde W}_{T - \tau_j} - {\tilde W}_{T - \tau_{j-1}}),
```
and with $T - \tau_{j-1} < T - \tau_j,$ which means at the "front" of the *decreasing* steps!

## Tracing back the same forward paths with a specific Wiener process

In order to trace back the same sample paths, one must use a specific Wiener process $\{\bar W_t\}_{t\geq 0}$ define as the weak solution (i.e. with the specific original Wiener process $\{W_t\}_{t\geq 0}$ of the forward path)
```math
    \mathrm{d}\bar W_t = \mathrm{d}W_t + \frac{1}{p(t, X_t)}\nabla_x \cdot (p(t, X_t) G(t, X_t)) \;\mathrm{d}t,
```
i.e.
```math
    \bar W_t = W_t + \int_0^t \frac{1}{p(s, X_s)}\nabla_x \cdot (p(s, X_s) G(s, X_s)) \;\mathrm{d}s.
```
The proof that $\{\bar W_t\}_{t\geq 0}$ is actually a Wiener process is not trivial. We will be content in considering a specific illustrative one-dimensional case, given by
```math
    \mathrm{d}X_t = \sigma \;\mathrm{d}W_t,
```
with
```math
    X_0 = 0.
```

This is a somewhat trivial example, with solution
```math
    X_t = \sigma W_t.
```
The probability distribution function is
```math
    p(t, x) = \frac{1}{\sqrt{2\pi \sigma^2 t}}e^{-\frac{1}{2}\frac{x^2}{\sigma^2t}},
```
for $t > 0,$ $x\in\mathbb{R}.$

## References

1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)
3. [B. D. O. Anderson (1982). Reverse-time diffusion equation models, Stochastic Process. Appl., vol. 12, no. 3, 313–326, DOI: 10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)