# Reverse probability flow

## Aim

Review the reverse probability flow used for sampling, after the Stein score function has been trained, as developed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, and Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html), based on the probability flow ODE developed in these articles and on the reverse time diffusion equation model previously worked out by [Anderson (1982)](https://doi.org/10.1016/0304-4149(82)90051-5) (see also [Haussmann and Pardoux (1986)](https://doi.org/10.1214/aop/1176992362))

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

If the initial condition is a random variable $X_0,$ and the flow evolves to $X_T,$ then the reverse flow evolves back to $X_0.$ By approximating $X_T \sim Y_T$ by another random variable $Y_T,$ say a standard normal distribution as in the generative diffusion processes, then the reverse flow evolves back towards an approximation $Y_0$ of the initial distribution $X_0.$

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

We now write the reverse ODE by making the change of variables $\tilde Y_{\tilde t} = Y_{T - \tilde t},$ with the reverse time variable $\tilde t = T - t.$ It is just an ODE (pathwise), so the reverse equation follows from a straightforward chain rule, upon the change $\tilde t \mapsto T - \tilde t,$
```math
    \begin{align*}
        \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} = \frac{\mathrm{d}}{\mathrm{d}\tilde t}Y_{T - \tilde t} = - \frac{\mathrm{d}Y_{T - \tilde t}}{\mathrm{d}\tilde t} & = - f(T - \tilde t, Y_{T - \tilde t}) + \frac{1}{2} \nabla_y \cdot ( G(T - \tilde t, Y_{T - \tilde t})G(T - \tilde t, Y_{T - \tilde t})^{\mathrm{tr}} ) \\
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
The terms with $GG^{\mathrm{tr}}$ don't come with the right sign (for the conversion from probability flow ODE to the associated SDE), so we just rewrite it as (like adding and subtracting the same terms)
```math
    \begin{align*}
        \frac{\mathrm{d}{\tilde Y}_{\tilde t}}{\mathrm{d}\tilde t} & = - f(T - \tilde t, {\tilde Y}_{\tilde t}) + \nabla_y \cdot ( G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}}\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}) \\
        & \qquad \qquad - \frac{1}{2} \nabla_y \cdot ( G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad - \frac{1}{2} G(T - \tilde t, {\tilde Y}_{\tilde t})G(T - \tilde t, {\tilde Y}_{\tilde t})^{\mathrm{tr}}\nabla_y \log p(T - \tilde t, {\tilde Y}_{\tilde t}).
    \end{align*}
```

Now, with the proper sign, the last two terms on the right hand side become the diffusion term in the associated SDE for which this is the probability flow equation, namely
```math
    \begin{align*}
        \mathrm{d}{\tilde X}_{\tilde t} & = \bigg( - f(T - \tilde t, {\tilde X}_{\tilde t}) + \nabla_x \cdot ( G(T - \tilde t, {\tilde X}_{\tilde t})G(T - \tilde t, {\tilde X}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde t, {\tilde X}_{\tilde t})G(T - \tilde t, {\tilde X}_{\tilde t})^{\mathrm{tr}}\nabla_x \log p(T - \tilde t, {\tilde X}_{\tilde t}) \bigg) \;\mathrm{d}\tilde t\\
        & \qquad \qquad \qquad + G(T - \tilde t, {\tilde X}_{\tilde t})\;\mathrm{d}{\tilde W}_{\tilde t},
    \end{align*}
```
where $\{{\tilde W}_{\tilde t}\}_{\tilde t}$ is a (possibly different) Wiener process. In integral form, the equation for ${\tilde X}_{\tilde t},$ integrating from $\tilde t = 0$ to $\tilde t = T - t,$ reads
```math
    \begin{align*}
        {\tilde X}_{\tilde t} - {\tilde X}_0 & = \int_0^{\tilde t} \bigg( - f(T - \tilde \tau, {\tilde X}_{\tilde \tau}) + \nabla_x \cdot ( G(T - \tilde \tau, {\tilde X}_{\tilde \tau})G(T - \tilde \tau, {\tilde X}_{\tilde \tau})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde \tau, {\tilde X}_{\tilde \tau})G(T - \tilde \tau, {\tilde X}_{\tilde \tau})^{\mathrm{tr}}\nabla_x \log p(T - \tilde \tau, {\tilde X}_{\tilde \tau}) \bigg) \;\mathrm{d}\tilde \tau\\
        & \qquad \qquad \qquad + \int_0^{\tilde t} G(T - \tilde \tau, {\tilde X}_{\tilde \tau})\;\mathrm{d}{\tilde W}_{\tilde \tau},
    \end{align*}
```

Back to the original time $t = T - \tilde t,$ setting ${\hat X}_t = {\tilde X}_{T - t} = {\tilde X}_{\tilde t},$ and making the change of variable $\tau = T - \tilde \tau$ in the integral term, this becomes
```math
    \begin{align*}
        {\hat X}_t - {\hat X}_T & = \int_t^T \bigg( - f(\tau, {\hat X}_\tau) + \nabla_x \cdot ( G(\tau, {\hat X}_\tau)G(\tau, {\hat X}_\tau)^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(\tau, {\hat X}_\tau)G(\tau, {\hat X}_\tau)^{\mathrm{tr}}\nabla_x \log p(\tau, {\hat X}_\tau) \bigg) \;\mathrm{d}\tau\\
        & \qquad \qquad \qquad - \int_t^{T} G(\tau, {\hat X}_\tau)\mathrm{d}{\tilde W}_{T-\tau},
    \end{align*}
```
which can be written as
```math
    \begin{align*}
        {\hat X}_T - {\hat X}_t & = \int_{t}^{T} \bigg( f(\tau, {\hat X}_\tau) - \nabla_x \cdot ( G(\tau, {\hat X}_\tau)G(\tau, {\hat X}_\tau)^{\mathrm{tr}} ) \\
        & \qquad \qquad - G(\tau, {\hat X}_\tau)G(\tau, {\hat X}_\tau)^{\mathrm{tr}}\nabla_x \log p(\tau, {\hat X}_\tau) \bigg) \;\mathrm{d} \tau\\
        & \qquad \qquad \qquad + \int_{t}^T G(\tau, {\hat X}_\tau)\mathrm{d}{\hat W}_\tau,
    \end{align*}
```
with shorthand
```math
    \begin{align*}
        \mathrm{d}{\hat X}_t & = \bigg( f(t, {\hat X}_t) - \nabla_x \cdot ( G(t, {\hat X}_t)G(t, {\hat X}_t)^{\mathrm{tr}} ) \\
        & \qquad \qquad - G(t, {\hat X}_t)G(t, {\hat X}_t)^{\mathrm{tr}}\nabla_x \log p(t, {\hat X}_t) \bigg) \;\mathrm{d}t + G(\tau, {\hat X}_t)\mathrm{d}{\hat W}_t,
    \end{align*} 
```
where
```math
    {\hat W}_t = {\tilde W}_{T - t},
```
with the understanding that $\{{\hat W}_t\}_{0 \leq t \leq T}$ is a *backward Wiener process,* for which ${\hat W}_T = 0;$ the term ${\hat X}_t = {\tilde X}_{T - t}$ is independent of *previous* steps of the backward Wiener process, such as ${\hat W}_{t - \tau} - {\hat W}_t = {\tilde W}_{T - t + \tau} - {\tilde W}_{T - t},$ $\tau > 0;$ and the stochastic integral above is a *backward Itô integral,* with
```math
    \int_t^T {\hat H}_\tau \;\mathrm{d}{\hat W}_\tau = \lim \sum_{i=0}^{n-1} {\hat H}_{t_{i+1}} ( {\hat W}_{t_{i+1}} - {\hat W}_{t_{i}} ),
```
where $t = \tau_0 < \tau_1 < \tau_n = T,$ and the limit is taken as $\max_{i=0, n-1}|\tau_{i+1} - \tau_i| \rightarrow 0.$ This is essentially the Itô integral rephrased backwards. Let us examine this more carefully.

We start with the Itô integral
```math
    \int_0^{T - t} {\tilde H}_{\tilde \tau}\;\mathrm{d}{\tilde W}_{\tilde \tau},
```
defined for any given (non-antecipative) process $\{H_{\tilde t}\}_{\tilde t \geq 0},$ with respect to a (forward) Wiener process $\{{\tilde W}_{\tilde t}\}_{\tilde t \geq 0}.$ This can be thought as the limit, as the mesh $0 = \tilde \tau_0 < \tilde \tau_1 < \ldots < \tilde \tau_n = T - \tilde t$ is refined, of the sums
```math
    \sum_{j=1}^n {\tilde H}_{\tilde \tau_{j-1}}({\tilde W}_{{\tilde \tau}_j} - {\tilde W}_{{\tilde \tau}_{j-1}}).
```
Now we define the points $\tau_j = T - \tilde \tau_j,$ which form a mesh $T = \tau_0 = T - {\tilde \tau}_0 > \ldots T - {\tilde \tau}_n = T - t = \tau_n.$ The summation can be written as
```math
    \sum_{j=1}^n {\tilde H}_{\tilde \tau_{j-1}}({\tilde W}_{{\tilde \tau}_j} - {\tilde W}_{{\tilde \tau}_{j-1}}) = \sum_{j=1}^n {\tilde H}_{T - \tau_{j-1}} ( {\tilde W}_{T - \tau_j} - {\tilde W}_{T - \tau_{j-1}} ).
```
Defining ${\hat H}_t = {\tilde H}_{T - t}$ and ${\hat W}_t = {\tilde W}_{T - t},$ we write the above as
```math
    \sum_{j=1}^n {\hat H}_{\tau_{j-1}} ( {\hat W}_{\tau_j} - {\hat W}_{\tau_{j-1}} ).
```
But notice that, now, $\tau_j < \tau_{j-1}.$ In order to make this fact look more natural, we reindex the summation with $i = N - j,$ and define the mesh with ${\hat \tau}_i = \tau_{N-i},$ so that
```math
    \begin{align*}
        \sum_{j=1}^n {\hat H}_{\tau_{j-1}} ( {\hat W}_{\tau_j} - {\hat W}_{\tau_{j-1}} ) & = \sum_{i=0}^{n-1} {\hat H}_{\tau_{N-i-1}} ( {\hat W}_{\tau_{N-i}} - {\hat W}_{\tau_{N-i-1}} ) \\
        & = \sum_{i=0}^{n-1} {\hat H}_{\tau_{N-(i+1)}} ( {\hat W}_{\tau_{N-i}} - {\hat W}_{\tau_{N-(i+1)}} ) \\
        & = \sum_{i=0}^{n-1} {\hat H}_{{\hat \tau}_{i+1}} ( {\hat W}_{{\hat \tau}_{i}} - {\hat W}_{{\hat \tau}_{i+1}} ) \\
        & = - \sum_{i=0}^{n-1} {\hat H}_{{\hat \tau}_{i+1}} ( {\hat W}_{{\hat \tau}_{i+1}} - {\hat W}_{{\hat \tau}_{i}} ).
    \end{align*}
```
The mesh runs from ${\hat \tau}_0 = \tau_N = T-t$ to ${\hat \tau}_N = \tau_0 = T.$ As the mesh is refined, this becomes the backward Itô integral
```math
    -\int_t^T {\hat H}_{\hat \tau}\;\mathrm{d}{\hat W}_{\hat \tau}.
```
Thus, we have obtained the following identity between the forward and backward Itô integrals,
```math
    \int_0^{T - t} {\tilde H}_{\tilde \tau}\;\mathrm{d}{\tilde W}_{\tilde \tau} = -\int_t^T {\hat H}_{\hat \tau}\;\mathrm{d}{\hat W}_{\hat \tau},
```
with the relevant changes of variables
```math
    {\hat H}_t = {\tilde H}_{T - t}, \qquad {\hat W}_t = {\tilde W}_{T - t}.
```
The process ${\tilde H}_{\tilde t}$ is independent of future increments of the Wiener process $\{{\tilde W}_{\tilde t}\}_{\tilde t \geq 0}$ if, and only if, ${\hat H}_t$ is independent of previous increments of the backward Wiener process $\{{\hat W}_t\}_{0\leq t \leq T}.$

## Tracing back the same forward paths with a specific Wiener process

Notice we wrote, above, ${\hat X}_t$ instead of $X_t,$ because the paths might not be the same, although the distributions are. In order to trace back the same sample paths, one must use a specific Wiener process $\{\bar W_t\}_{t\geq 0}$ defined as the weak solution (i.e. with the specific original Wiener process $\{W_t\}_{t\geq 0}$ of the forward path)
```math
    \mathrm{d}\bar W_t = \mathrm{d}W_t + \frac{1}{p(t, X_t)}\nabla_x \cdot (p(t, X_t) G(t, X_t)) \;\mathrm{d}t,
```
i.e.
```math
    \bar W_t = W_t + \int_0^t \frac{1}{p(s, X_s)}\nabla_x \cdot (p(s, X_s) G(s, X_s)) \;\mathrm{d}s.
```
With this noise, if $\{X_t\}_{t\geq 0}$ is the solution of the forward diffusion equation
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
then the (pathwise) reverse flow ${\tilde X}_{\tilde t} = X_{T - \tilde t}$ is a (weak) solution (because it solves a diffusion equation with a specific Wiener process) of
```math
    \mathrm{d}{\tilde X}_{\tilde t} = {\tilde f}(\tilde t, {\tilde X}_{\tilde t})\;\mathrm{d}t + {\tilde G}(\tilde t, {\tilde X}_{\tilde t})\;\mathrm{d}{\bar W}_{\tilde t},
```
with
```math
    {\tilde G}(\tilde t, {\tilde X}_{\tilde t}) = G(T - \tilde t, {\tilde X}_{\tilde t})
```
and
```math
    \begin{align*}
        {\tilde f}(\tilde t, {\tilde X}_{\tilde t}) & = \bigg( - f(T - \tilde t, {\tilde X}_{\tilde t}) + \nabla_x \cdot ( G(T - \tilde t, {\tilde X}_{\tilde t})G(T - \tilde t, {\tilde X}_{\tilde t})^{\mathrm{tr}} ) \\
        & \qquad \qquad + G(T - \tilde t, {\tilde X}_{\tilde t})G(T - \tilde t, {\tilde X}_{\tilde t})^{\mathrm{tr}}\nabla_x \log p(T - \tilde t, {\tilde X}_{\tilde t}) \bigg).
    \end{align*}
```

The proof that $\{\bar W_t\}_{t\geq 0}$ is actually a Wiener process is not trivial, thought. We will be content in considering a specific illustrative one-dimensional case.

## A simple scalar example

Consider the trivial diffusion equation
```math
    \mathrm{d}X_t = \sigma \;\mathrm{d}W_t,
```
with
```math
    X_0 = 0.
```

The solution is simply
```math
    X_t = \sigma W_t.
```

The marginal probability distribution functions of this stochastic process are
```math
    p(t, x) = \frac{1}{\sqrt{2\pi \sigma^2 t}}e^{-\frac{1}{2}\frac{x^2}{\sigma^2t}}, \quad x\in\mathbb{R},
```
for $t > 0.$ In this case,
```math
    \frac{1}{p(s, x)}\nabla_x \cdot (p(s, x) G(s, x)) = \sigma \frac{1}{p(s, x)}\nabla_x \cdot (p(s, x)) = \sigma \nabla_x \log(p(s, x)),
```
with
```math
    \sigma\nabla_x \log(p(s, x)) = \sigma \nabla_x \left( -\frac{1}{2}\frac{x^2}{\sigma^2t} - \log(\sqrt{2\pi \sigma^2 t}) \right) = - \frac{x}{\sigma t}.
```

The reverse Wiener process takes the form
```math
    {\bar W}_t = W_t - \int_0^t \frac{X_s}{\sigma s} \;\mathrm{d}s = W_t - \int_0^t \frac{W_s}{s}\;\mathrm{d}s.
```

The reverse equation reads
```math
    \mathrm{d}{\tilde X}_{\tilde t} = \sigma \;\mathrm{d}{\bar W}_{\tilde t},
```
i.e.
```math
    {\tilde X}_{\tilde t} = \sigma {\bar W}_{\tilde t} = \sigma W_{\tilde t} - \sigma \int_0^{\tilde t} \frac{W_s}{s}\;\mathrm{d}s.
```
i.e.
```math
    X_{T - \tilde t} = \sigma W_{\tilde t} - \sigma \int_0^{\tilde t} \frac{W_s}{s}\;\mathrm{d}s.
```
Replacing $t = T - \tilde t,$
```math
    X_t = \sigma W_{T - t} - \sigma \int_0^{T - t} \frac{W_s}{s}\;\mathrm{d}s.
```
Changing the integration variable according to $s \mapsto T - s,$ we find
```math
    X_t = \sigma W_{T - t} + \sigma \int_t^T \frac{W_{T-s}}{T-s}\;\mathrm{d}s.
```


## References

1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)
3. [B. D. O. Anderson (1982). Reverse-time diffusion equation models, Stochastic Process. Appl., vol. 12, no. 3, 313–326, DOI: 10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)
4. [U. G. Haussmann, E. Pardoux (1986). Time reversal of diffusions, Ann. Probab. 14, no. 4, 1188-1205](https://doi.org/10.1214/aop/1176992362)