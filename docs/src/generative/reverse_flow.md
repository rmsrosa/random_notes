# Reverse probability flow

```@meta
Draft = false
```

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

The original way of obtaining the reverse SDE, derived in [Anderson (1982)](https://doi.org/10.1016/0304-4149(82)90051-5), and seen in other works (e.g. [Haussmann and Pardoux (1986)](https://doi.org/10.1214/aop/1176992362)), is by looking at the joint distribution $p(t, x_t, s, x_s)$ at two different times $t$ and $s$ and by working with conditional distributions. We do it differently here, though. We look at the connection between the SDE and the probability flow, introduced by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) and generalized by [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

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

The fact that $\{\bar W_t\}_{t\geq 0}$ is actually a Wiener process is based on the characterization of the Wiener process as an almost surely continuous martingale with $W_0 = 0$ and with quadratic variation $[W_t, W_t] = t,$ for all $t\geq 0.$ Since the second term in the definition of $\bar W_t$ is a Riemann integral, its quadtratic variation is zero, and thus
```math
    [\bar W_t, \bar W_t]_t = [W_t, W_t]_t = t.
```

The fact that $X_t$ is independent of *previous* steps of $\{\bar W_t\}_{t \geq 0},$ say $\bar W_{t_2} - \bar W_{t_1},$ $0 \leq t_1 < t_2 \leq t,$ follows from the facts that $X_t$ is adapted to $\{W_t\}_{t\geq 0}$ and that $W_t$ itself is independent of previous steps of $\{\bar W_t\}_{t \geq 0}.$ The proof of that is a bit involved, though, and can be found in [Anderson (1982)](https://doi.org/10.1016/0304-4149(82)90051-5). Here, we content ourselves in proving that in a specific simple case below.

## A simple scalar example

For illustrative purposes, consider the trivial diffusion equation
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

The marginal probability densities of this stochastic process are
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
Thus, the reverse Wiener process takes the form
```math
    {\bar W}_t = W_t - \int_0^t \frac{X_s}{\sigma s} \;\mathrm{d}s = W_t - \int_0^t \frac{W_s}{s}\;\mathrm{d}s.
```

Write
```math
    X_T - X_t = \sigma W_T - \sigma W_t = \sigma {\bar W}_T - \sigma {\bar W}_t + \sigma\int_t^T \frac{W_s}{s}\;\mathrm{d}s = \sigma ({\bar W}_T - {\bar W}_t) + \int_t^T \frac{X_s}{s}\;\mathrm{d}s.
```
This becomes the reverse diffusion equation
```math
    \mathrm{d}X_t = \frac{X_t}{t}\;\mathrm{d}t + \sigma\;\mathrm{d}{\bar W}_t.
```
The diffusion term is a trivial constant term, but, nevertheless, we still have the remarkable property that $X_t=\sigma W_t$ is independent of previous increments of ${\bar W}_t.$ Indeed, both are Gaussian processes with zero mean, so the covariance, with $0 \leq t - \tau < t,$ is given by
```math
    \begin{align*}
        \mathbb{E}[X_t ({\bar W}_t - {\bar W}_{t - \tau})] & = \sigma\mathbb{E}\left[ W_t ({\bar W}_t - {\bar W}_{t - \tau})\right] \\
        & = \sigma\mathbb{E}\left[ W_t \left(W_t - W_{t - \tau} - \int_{t-\tau}^t \frac{W_s}{s}\;\mathrm{d}s\right)\right] \\
        & = \sigma\mathbb{E}\left[ W_t^2\right] - \sigma\mathbb{E}\left[W_t W_{t - \tau}\right] - \int_{t-\tau}^t \frac{\mathbb{E}[W_t W_s]}{s}\;\mathrm{d}s \\
        & = \sigma t - \sigma \min\{t, t - \tau\} - \sigma \int_{t-\tau}^t \frac{\min\{t, s\}}{s}\;\mathrm{d}s \\
        & = \sigma t - \sigma (t - \tau) - \sigma \int_{t-\tau}^t \;\mathrm{d}s \\
        & = \sigma t - \sigma (t - \tau) - \sigma \tau \\
        & = 0,
    \end{align*}
```
showing that they are uncorrelated. Similarly with
```math
    \mathbb{E}[X_{t_2} ({\bar W}_{t_1} - {\bar W}_{t_0})] = \sigma t_1 - \sigma t_0 - \sigma (t_1 - t_0) = 0,
```
for $0 \leq t_0 < t_1 \leq t_2.$

In order to see that $\{\bar W_t\}_{t \geq 0}$ is, in fact, a Wiener process, we first notice that the formula in the definition implies that it is a Gaussian process with zero expectation at each time. Now we compute the covariance, at times $t, s \geq 0,$ 
```math
    \begin{align*}
        \mathbb{E}[\bar W_t \bar W_s] & = \mathbb{E}\left[ \left(W_t - \int_0^t \frac{W_\tau}{\tau}\;\mathrm{d}\tau\right)\left(W_s - \int_0^s \frac{W_\xi}{\xi}\;\mathrm{d}\xi\right)\right] \\
        & = \mathbb{E}\left[ W_t W_s - \int_0^s \frac{W_t W_\xi}{\xi}\;\mathrm{d}\xi - \int_0^t \frac{W_\tau W_s}{\tau}\;\mathrm{d}\tau + \int_0^t \int_0^s \frac{W_\tau W_\xi}{\tau\xi}\;\mathrm{d}\xi\;\mathrm{d}\tau\right] \\
        & = \mathbb{E}[W_t W_s] - \int_0^s \frac{\mathbb{E}[W_t W_\xi]}{\xi}\;\mathrm{d}\xi - \int_0^t \frac{\mathbb{E}[W_\tau W_s]}{\tau}\;\mathrm{d}\tau + \int_0^t \int_0^s \frac{\mathbb{E}[W_\tau W_\xi]}{\tau\xi}\;\mathrm{d}\xi\;\mathrm{d}\tau \\
        & = \min\{t, s\} - \int_0^s \frac{\min\{t, \xi\}}{\xi}\;\mathrm{d}\xi - \int_0^t \frac{\min\{\tau, s\}}{\tau}\;\mathrm{d}\tau + \int_0^t \int_0^s \frac{\min\{\tau, \xi\}}{\tau\xi}\;\mathrm{d}\xi\;\mathrm{d}\tau
    \end{align*}
```
Assuming $0 \leq s \leq t,$ we obtain
```math
    \begin{align*}
        \mathbb{E}[\bar W_t \bar W_s] & = s - \int_0^s \;\mathrm{d}\xi - \int_0^s \;\mathrm{d}\tau - \int_s^t \frac{s}{\tau}\;\mathrm{d}\tau + \int_0^s \int_0^\tau \frac{1}{\tau}\;\mathrm{d}\xi\;\mathrm{d}\tau + \int_0^s \int_\tau^s \frac{1}{\xi}\;\mathrm{d}\xi\;\mathrm{d}\tau + \int_s^t \int_0^s \frac{1}{\tau}\;\mathrm{d}\xi\;\mathrm{d}\tau\\
        & = s - s - s - \int_s^t \frac{s}{\tau}\;\mathrm{d}\tau + \int_0^s \;\mathrm{d}\tau + \int_0^s \int_0^\xi \frac{1}{\xi}\;\mathrm{d}\tau\;\mathrm{d}\xi + \int_s^t \frac{s}{\tau}\;\mathrm{d}\tau \\ 
        & = s - s - s + s + s \\
        & = s.
    \end{align*}
```
For $0 \leq t \leq s,$ we get
```math
    \mathbb{E}[\bar W_t \bar W_s] = t,
```
which means that
```math
    \mathbb{E}[\bar W_t \bar W_s] = \min\{t, s\}.
```
Thus, by the characterization of Wiener processes as the Gaussian processes with zero mean and correlation $\min\{t, s\},$ we see that $\{\bar W_t\}_{t \geq 0}$ is indeed a Wiener process.

As a final remark, notice that, if we revert time $\tilde t = T - t$ in the equation
```math
    \mathrm{d}X_t = \frac{X_t}{t}\;\mathrm{d}t + \sigma\;\mathrm{d}{\bar W}_t,
```
we find
```math
    \mathrm{d}{\tilde X}_{\tilde t} = - \frac{{\tilde X}_{\tilde t}}{T - \tilde t}\;\mathrm{d}{\tilde t} + \sigma\;\mathrm{d}{\tilde W}_{\tilde t},
```
for ${\tilde X}_{\tilde t} = X_{T - \tilde t} = X_t$ and ${\tilde W}_{\tilde t} = {\bar W}_{T - \tilde t}.$ This is the Brownian bridge equation, except that we start at ${\tilde X}_0 = X_T$ and end up at ${\tilde X}_T = X_0.$

## Numerics

```@example reverseflow
using StatsPlots
using Random
using Distributions
using Markdown

nothing # hide
```

```@example reverseflow
rng = Xoshiro(12345)
nothing # hide
```

```@example reverseflow
trange = 0.0:0.01:1.0
trangeback = 1.0:-0.01:0.0
numsamples = 1024

sigma(t) = t
sigmaprime(t) = 1
g(t) = sqrt(2 * sigma(t) * sigmaprime(t))

x0 = 0.0

Xt = zeros(size(trange, 1), numsamples)
dt = Float64(trange.step)
dWt = sqrt(dt) .* randn(length(trange), numsamples)
Wt = zero(Xt)
@assert axes(Xt, 1) == axes(trange, 1)
@inbounds for m in axes(Xt, 2)
    n1 = first(eachindex(axes(Xt, 1), axes(trange, 1)))
    Xt[n1, m] = x0
    Wt[n1, m] = 0.0
    @inbounds for n in Iterators.drop(eachindex(axes(trange,1), axes(Xt, 1)), 1)
        Xt[n, m] = Xt[n1, m] + g(trange[n1]) * dWt[n1, m]
        Wt[n, m] = Wt[n1, m] + dWt[n1, m]
        n1 = n
    end
end
```

```@example reverseflow
histogram(title="histogram of Xt", titlefont=10, Xt[end, :], bins=40)
```

```@example reverseflow
histogram(title="histogram of Wt", titlefont=10, Wt[end, :], bins=40)
```

```@example reverseflow
plot(title="Sample paths of Xt", titlefont=10)
plot!(trange, Xt[:, 1:200], color=1, alpha=0.2, legend=false)
plot!(trange, Xt[:, 1:5], color=2, linewidth=1.5, legend=false)
```

```@example reverseflow
barWt = zero(Xt)
Vt = zero(Xt)
for m in axes(barWt, 2)
    n1 = first(eachindex(axes(barWt, 1), axes(trange, 1)))
    barWt[n1, m] = 0.0
    Vt[n1, m] = 0.0
    @inbounds for n in Iterators.drop(eachindex(axes(trange,1), axes(barWt, 1), axes(barXt, 1), axes(Vt, 1)), 1)
        Vt[n, m] = Vt[n1, m] - g(trange[n]) / sigma(trange[n])^2 * Xt[n1, m] * dt
        barWt[n, m] = Wt[n1, m] + Vt[n1, m]
        n1 = n
    end
end
```

```@example reverseflow
histogram(title="histogram of barWt", titlefont=10, barWt[end, :], bins=40)
```

```@example reverseflow
plot(title="Sample paths Wt", titlefont=10, ylims=(-4, 4))
plot!(trange, Wt[:, 1:200], color=1, alpha=0.2, legend=false)
plot!(trange, Wt[:, 1:5], color=2, linewidth=1.5, legend=false)
```

```@example reverseflow
plot(title="Sample paths barWt", titlefont=10, ylims=(-4, 4))
plot!(trange, barWt[:, 1:200], color=1, alpha=0.2, legend=false)
plot!(trange, barWt[:, 1:5], color=2, linewidth=1.5, legend=false)
```

```@example reverseflow
Xtback = zeros(size(trange, 1), numsamples)

dt = Float64(trange.step)
@assert axes(Xtback, 1) == axes(trangeback, 1)
for m in axes(Xtback, 2)
    n1 = last(eachindex(axes(Xtback, 1), axes(trange, 1)))
    Xtback[n1, m] = Xt[end, m]
    for n in Iterators.drop(Iterators.reverse(eachindex(axes(trange,1), axes(Xtback, 1))), 1)
        Xtback[n, m] = Xtback[n1, m] - 2 * sigmaprime(trange[n1]) / sigma(trange[n1]) * Xtback[n1, m] * dt - g(trange[n1]) * (barWt[n1, m] - barWt[n, m])
        n1 = n
    end
end
```

```@example reverseflow
plot(title="Sample paths reverse Xt", titlefont=10)
plot!(trange, Xtback[:, 1:200], color=1, alpha=0.2, legend=false)
plot!(trange, Xtback[:, 1:5], color=2, linewidth=1.5, legend=false)
```


```@example reverseflow
plot(title="Sample paths Xt (blue) and reverse Xt (red)", titlefont=10, legend=false)
plot!(trange, Xt[:, 1:5], color=1, label="forward")
plot!(trange, Xtback[:, 1:5], color=2, linewidth=1.5, label="reverse")
```

```@example reverseflow
nothing
```

Hmm, let us try something simpler. We start with $X_0 = 0$ and consider the SDE with $f=0$ and $g=g(t) = \sqrt{2\sigma(t)\sigma'(t)},$ for a given $\sigma=\sigma(t),$
```math
\begin{cases}
    \mathrm{d}X_t = \sqrt{2\sigma(t)\sigma'(t)}\;\mathrm{d}W_t, \\
    X_t\bigg|_{t=0} = 0.
\end{cases}
```
The solution is a time-changed Brownian motion,
```math
    X_t = \int_0^t \sqrt{2\sigma(s)\sigma'(s)} \;\mathrm{d}W_s = W_{\sigma(t)^2}.
```
The probability density function for the process is
```math
    p(t, x) = G(\sigma(t)) = \frac{1}{\sqrt{2\pi\sigma(t)^2}} e^{-\frac{1}{2}\frac{x^2}{\sigma(t)^2}},
```
where $G=G(\sigma)$ is the probability density function of the normal distribution $\mathcal{N}(0, \sigma^2).$

Since
```math
\ln p(t, x) = -\frac{1}{2}\frac{x^2}{\sigma(t)^2} - \ln(\sqrt{2\pi\sigma(t)^2}),
```
the Stein score function of the process $\{X_t\}_{t\geq 0}$ is
```math
    \nabla_x \ln p(t, x) = -\frac{x}{\sigma(t)^2}.
```

Hence, the reverse equation
```math
    \mathrm{d}{X}_t = -g(t)^2\nabla_x \ln p(t, {X}_t) \;\mathrm{d}t + g(t)\mathrm{d}{\hat W}_t,
```
becomes
```math
    \mathrm{d}{X}_t = \frac{g(t)^2}{\sigma(t)^2} X_t\;\mathrm{d}t + g(t)\mathrm{d}{\hat W}_t,
```
or, in terms of $\sigma$ and $\sigma',$
```math
    \mathrm{d}{X}_t = 2\frac{\sigma'(t)}{\sigma(t)} X_t\;\mathrm{d}t + \sqrt{2\sigma(t)\sigma'(t)}\mathrm{d}{\hat W}_t,
```
where
```math
    {\hat W}_t = {\bar W}_{T - t}, \qquad {\bar W}_t = W_t - \int_0^t \frac{g(s)}{\sigma(s)^2} X_s \;\mathrm{d}s = W_t - \int_0^t \sqrt{\frac{2\sigma'(s)}{\sigma(s)}} X_s \;\mathrm{d}s.
```

This is iterated recursively backwards in time, with
```math
X_{t_j} - X_{t_{j-1}} = \int_{t_{j-1}}^{t_j} 2\frac{\sigma'(s)}{\sigma(s)} X_s \;\mathrm{d}s + \int_{t_{j-1}}^{t_j} g(s) \;\mathrm{d}{\hat W}_s.
```
which we approximate with
```math
X_{t_j} - X_{t_{j-1}} \approx 2\frac{\sigma'(t_j)}{\sigma(t_j)} X_{t_j} (t_j - t_{j-1}) + g(t_j) ({\hat W}_{t_j} - {\hat W}_{t_{j-1}}).
```

## References

1. [B. D. O. Anderson (1982). Reverse-time diffusion equation models, Stochastic Process. Appl., vol. 12, no. 3, 313–326, DOI: 10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)
1. [U. G. Haussmann, E. Pardoux (1986). Time reversal of diffusions, Ann. Probab. 14, no. 4, 1188-1205](https://doi.org/10.1214/aop/1176992362)
1. [D. Maoutsa, S. Reich, M. Opper (2020), "Interacting particle solutions of Fokker-Planck equations through gradient-log-density estimation", Entropy, 22(8), 802, DOI: 10.3390/e22080802](https://doi.org/10.3390/e22080802)
1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)