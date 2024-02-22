# Score function and Langevin sampling

```@setup scoreandlangevin
using StatsPlots
using Random
using Distributions
using Markdown

rng = Xoshiro(123)
```

```@setup scoreandlangevin
function Distributions.gradlogpdf(d::UnivariateMixture, x::Real)
    ps = probs(d)
    cs = components(d)
    ps1 = first(ps)
    cs1 = first(cs)
    pdfx1 = pdf(cs1, x)
    pdfx = ps1 * pdfx1
    glp = pdfx * gradlogpdf(cs1, x)
    if iszero(ps1)
        glp = zero(glp)
    end
    @inbounds for (psi, csi) in Iterators.drop(zip(ps, cs), 1)
        if !iszero(psi)
            pdfxi = pdf(csi, x)
            if !iszero(pdfxi)
                pipdfxi = psi * pdfxi
                pdfx += pipdfxi
                glp += pipdfxi * gradlogpdf(csi, x)
            end
        end
    end
    if !iszero(pdfx) # else glp is already zero
        glp /= pdfx
    end 
    return glp
end
```

```@setup scoreandlangevin
function solve_sde_1d!(rng, xt, x0, tt, f, g, params)
    @assert axes(xt) == (eachindex(tt), eachindex(x0))

    dt = (tt[end] - tt[begin])/(tt.len - 1)
    sqrtdt = √dt

    xt[1, :] .= x0
    k1 = firstindex(tt)
    @inbounds for k in Iterators.drop(eachindex(tt), 1)
        randn!(rng, view(xt, lastindex(tt), :))
        for j in axes(xt, 2)
            xt[k, j] = xt[k1, j] + f(tt[k1], xt[k1, j], params) * dt + g(tt[k1], xt[k1, j], params) * sqrtdt * xt[end, j]
        end
        k1 = k
    end
    return xt
end

function solve_sde_1d(rng, x0, tt, f, g, params)
    
    xt = zeros(length(tt), length(x0))

    solve_sde_1d!(rng, xt, x0, tt, f, g, params)

    return xt
end
```

```@setup scoreandlangevin
function solve_fokkerplanck!(pt, p0, tt, xx, f, g, params)
    @assert axes(pt) == (eachindex(tt), eachindex(p0))

    dt = Float64(tt.step)
    dx = Float64(xx.step)

    twicedx = 2dx
    dxsquare = dx^2

    pt[begin, :] .= p0
    k1 = firstindex(tt)
    # Julia is column-major, which means the next code is not optimized with respect to the order of the indices of `xt`,
    # but this is for ilustrative purposes and is good enough this way.
    for i in Iterators.drop(eachindex(tt), 1)
        
        jm1, jc = Iterators.take(eachindex(p0), 2)
        pt[i, jm1] = 0.0
        for jp1 in Iterators.drop(eachindex(p0), 2)
            pt[i, jc] = pt[k1, jc] + 
                dt * (
                        - (
                        f(tt[k1], xx[jp1], params) * pt[k1, jp1] - f(tt[k1], xx[jm1], params) * pt[k1, jm1]
                    ) / twicedx + (
                        g(tt[k1], xx[jp1], params)^2 * pt[k1, jp1] - 2 * g(tt[k1], xx[jc], params)^2 * pt[k1, jc] + g(tt[k1], xx[jm1], params)^2 * pt[k1, jm1]
                    ) / dxsquare / 2
                )
            jm1, jc = jc, jp1
        end
        pt[i, jc] = 0.0
        k1 = i
    end
    return pt
end

function solve_fokkerplanck(p0, t0, tf, xx, f, g, params; snapshots = 200)
    
    ndt = Int.(snapshots * ceil(max(
        maximum(
            abs2(g(t, x, params)) for t in range(t0, tf, length=length(xx)), x in xx) * (tf - t0) / ((xx[end] - xx[begin]) / (length(xx) - 1)
        )^2,
        maximum(
            abs(f(t, x, params)) for t in range(t0, tf, length=length(xx)), x in xx) * (tf - t0) / ((xx[end] - xx[begin]) / (length(xx) - 1)
        ))) / snapshots
    )
    tt = range(t0, tf, length=ndt+1)
    pt = zeros(ndt+1, length(p0))

    solve_fokkerplanck!(pt, p0, tt, xx, f, g, params)


    return tt[begin:div(ndt, snapshots):end], pt[begin:div(ndt, snapshots):end, :]
end
```

## Introduction

One of the cornerstones of score-based generative models is the method of sampling from the score function of a distribution via Langevin dynamics. Our aim here is to review the notion of score function and the method of sampling via Langevin dynamics based on the score function.

## The score function

Given a random variable $\mathbf{X}$ in $\mathbb{R}^d$, $d\in\mathbb{N},$ we denote its pdf by $p_X(\mathbf{x})$, while its **score function**, also known as **gradlogpdf**, is defined by
```math
    \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x}) = \boldsymbol{\nabla}_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\boldsymbol{\partial}\log p(\mathbf{x})}{\boldsymbol{\partial}\mathbf{x}} = \left( \frac{\partial}{\partial x_j} \log p(\mathbf{x})\right)_{j=1, \ldots, d},
```
where we may use either notation $\boldsymbol{\nabla}_{\mathbf{x}}$ or ${\boldsymbol{\partial}}/{\boldsymbol{\partial}\mathbf{x}}$ for the gradient of a scalar function. (For the differential of a vector-valued function, we will use either $\mathrm{D}_{\mathbf{x}}$ or ${\boldsymbol{\partial}}/{\boldsymbol{\partial}\mathbf{x}}$.)

For a parametrized model with pdf denoted by $p(\mathbf{x}; \boldsymbol{\theta})$, or $p(\mathbf{x} | \boldsymbol{\theta})$, and parameters $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_m),$ $m\in \mathbb{N}$, the score function becomes
```math
    \boldsymbol{\psi}(\mathbf{x}; \boldsymbol{\theta}) = \boldsymbol{\nabla}_{\mathbf{x}}p(\mathbf{x}; \boldsymbol{\theta}) = \left( \frac{\partial}{\partial x_j} p(\mathbf{x}; \boldsymbol{\theta})\right)_{j=1, \ldots, d}.
```

In the univariate case, the score function is also univariate and is given by the derivative of the log of the pdf. For example, for a univariate Normal distribution $\mathcal{N}(\mu, \sigma^2)$, $\mu\in\mathbb{R}$, $\sigma > 0$, the pdf, logpdf and gradlogpdf are
```math
    \begin{align*}
        p_X(x) & = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}, \\
        \log p_X(x) & = -\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2 - \log(\sqrt{2\pi}\sigma), \\
        \psi_X(x) & = - \frac{x - \mu}{\sigma^2}.
    \end{align*}
```
Notice the score function in this case is just a linear function vanishing at the mean of the distribution and with the slope being minus the multiplicative inverse of its variance.

```@setup scoreandlangevin
xrange = range(0, 10, length=200)
mu = 4
sigma = 0.5
prob = Normal(mu, sigma)
plt1 = plot(xrange, x -> pdf(prob, x), title="pdf of \$\\mathcal{N}($mu, $(sigma)^2)\$", titlefont=8)
plt2 = plot(xrange, x -> logpdf(prob, x), title="logpdf of \$\\mathcal{N}($mu, $(sigma)^2)\$", titlefont=8)
plt3 = plot(xrange, x -> gradlogpdf(prob, x), title="gradlogpdf (score) of \$\\mathcal{N}($mu, $(sigma)^2)\$", titlefont=8)
```

```@example scoreandlangevin
plot(plt1, plt2, plt3, layout=(3, 1), legend=false, size=(600, 400)) # hide
```

In the multivariate case, the score function is a *vector field* in the event space $\mathbb{R}^d$.

```@setup scoreandlangevin
xxrange = range(-4, 8, length=100)
yyrange = range(-3, 7, length=100)
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
xx, yy = meshgrid(xxrange[begin:8:end], yyrange[begin:8:end])
mu = [2, 1]
sigma = [1 0.2; 0.2 0.6]
prob = MvNormal(mu, sigma)
uu = reduce(hcat, gradlogpdf(prob, [x, y]) for (x, y) in zip(xx, yy))
plt1a = surface(xxrange, yyrange, (x, y) -> pdf(prob, [x, y]), color=:vik, colorbar=false, title="graph of the pdf", titlefont=8)
plt1b = heatmap(xxrange, yyrange, (x, y) -> pdf(prob, [x, y]), color=:vik, title="heatmap of the pdf", titlefont=8)
plt2a = surface(xxrange, yyrange, (x, y) -> logpdf(prob, [x, y]), color=:vik, colorbar=false, title="graph of the logpdf", titlefont=8)
plt2b = heatmap(xxrange, yyrange, (x, y) -> logpdf(prob, [x, y]), color=:vik, title="heatmap of the logpdf with score vector field", titlefont=8)
quiver!(plt2b, xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
```

```@example scoreandlangevin
plot(plt1a, plt1b, plt2a, plt2b, size=(600, 400)) # hide
```

We should warn that this notion of score function used in generative models in machine learning is different from the more classical notion of score in Statistics. The classical score function is defined for a parametrized model and refers to the gradient of the log-likelyhood
```math
    \ell(\boldsymbol{\theta}|\mathbf{x}) = \log\mathcal{L}(\boldsymbol{\theta}|\mathbf{x}) = p(\mathbf{x}|\boldsymbol{\theta}),
```
of a parametrized model, with respect to the parameters, i.e.
```math
    s(\boldsymbol{\theta}; \mathbf{x}) = \boldsymbol{\nabla}_{\boldsymbol{\theta}}\log \mathcal{L}(\boldsymbol{\theta}|\mathbf{x}) = \frac{\boldsymbol{\partial}\log \mathcal{L}(\boldsymbol{\theta}|\mathbf{x})}{\boldsymbol{\partial}\mathbf{x}}.
```
This notion measures the sensitivity of the model with respect to changes in the parameters and is useful in parameter estimation, to maximize the maximum likelyhood function, for instance.

The score function given by the gradlogpdf of a distribution is, on the other hand, useful for drawing samples via Langevin dynamics.

## Langevin dynamics

The velocity of a particle moving in a fluid has long been known to be reduced by friction forces with the surrounding fluid particles. For relatively slowly moving particles, when the surrouding fluid flow is essentially laminar, this friction force is regarded to be proportional to the velocity, in what is known as the *Stokes law.* When the motion is relatively fast and the flow around the particle is turbulent, this friction tends to be proportional to the square of the velocity.

If this were the only force, though, a particle initially at rest would remain forever at rest. That is not the case, as observed by the botanist [Robert Brown (1828)](https://doi.org/10.1080%2F14786442808674769). His observations led to what is now known as Brownian motion, and which is formally modeled as a Wiener process (see e.g. [Einstein (1905)](https://doi.org/10.1002/andp.19053220806) and [Mörters and Peres (2010)](https://www.cambridge.org/il/academic/subjects/statistics-probability/probability-theory-and-stochastic-processes/brownian-motion?format=HB&isbn=9780521760188)). A Wiener process describes the (random) position of a particle which is initially at rest and is put into motion by the erratic collisions with the nearby fluid particles.

When in motion, both forces are actually in effect, a deterministic one dependent on the velocity and a random one due to irregular collisions. In a short time scale, the inertia forces are negligible and we recover the Brownian motion. For larger times scales, the Langevin is more appropriate since it takes both forces into account.

### Langevin equation

In the Langevin model, both viscous and random collision forces affect the momentum of the particle ([Paul Langevin (1908)](https://gallica.bnf.fr/ark:/12148/bpt6k3100t/f530.item)). In this model, the position $x_t$ of a particle of mass $m$ at time $t$ is given by
```math
    m \ddot x_t = - \mu \dot x_t + \alpha \xi_t,
```
where $\mu$ is a proportionality viscosity coefficient associated with the frictional drag force $-\mu\dot x_t$ proportional to the velocity; and $\alpha$ is a proportionality coefficient associated with a white noise term $\xi_t$ modeling the random collisions with the fluid particles. The two coefficients are connected by
```math
    \alpha = \sqrt{2\mu k_B T},
```
where $k_B$ is the Boltzmann constant and $T$ is the temperature of the fluid.

The white noise is highly irregular, so the equation above is made rigorous with the theory of stochastic differential equations when casted in the form
```math
    \mathrm{d}V_t = \nu V_t \;\mathrm{d}t + \sigma \;\mathrm{d}W_t,
```
where $\{V_t\}_t$ is a stochastic processes representing the evolution of the velocity in time, $\nu = \mu / m$, $\sigma$ is called the *diffusion* parameter, and $\{W_t\}_t$ is a Wiener process, whose formal derivative represents the white noise. The solution $\{V_t\}_t$ of the equation above is known as the **Ornstein-Uhlenbeck** stochastic process. The relation between $\sigma$ and $\nu$ becomes
```math
    \sigma = \sqrt{\frac{2\nu k_B T}{m}}.
```

This is all fine for a nearly free particle, affected only by friction and by smaller nearby particles. More generally, one also considers a particle under an extra force field with potential $U=U(x)$. In this case, the equation is modified to
```math
    m \ddot x_t = - \mu \dot x_t - \nabla U(x_t) + \alpha \xi_t,
```
The rigorous stochastic formulation takes the form of a system,
```math
    \begin{cases}
        \mathrm{d}X_t = V_t\;\mathrm{d}t, \\
        \mathrm{d}V_t = (-\nu V_t - \nabla U(X_t))\;\mathrm{d}t + \sigma \;\mathrm{d}W_t.
    \end{cases}
```
These are called the **Langevin equation** or **Langevin system.**

### The overdamped limit

The term $m \ddot x_t$ represents the inertial force. When the motion is relatively slow, this inertia might be negligible when compared with the viscous foce. Dropping this term yields the equation
```math
    0 = - \mu \dot x_t - \nabla U(x_t) + \alpha \xi_t.
```
The corresponding stochastic system reduces to a single equation
```math
    \nu \mathrm{d}X_t = - \nabla U(X_t)\;\mathrm{d}t + \sigma \;\mathrm{d}W_t.
```
The time scale for this approximation to hold is a short time scale represented by $\tilde t = t / \nu$. Rescaling leads to the *overdamped Langevin equation*
```math
    \mathrm{d}\tilde X_t = - \nabla_{\tilde x}\tilde U(\tilde X_{\tilde t})\;\mathrm{d}\tilde t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}\tilde W_{\tilde t}.
```
We will do the details of this change of variables below, when performing a different approach of taking the overdamped asymptotic limit $\nu \rightarrow \infty$.

For the moment, notice that, in the absence of a force field $U=U(x)$, we are left (dropping the tildes in the rescaled equation) with
```math
    \mathrm{d}X_t = \sqrt{\frac{2k_B T}{m}} \;\mathrm{d}W_t,
```
which is the Brownian motion equation, with the solution
```math
    X_t = \sqrt{\frac{2k_B T}{m}} W_t.
```
In this way, we recover the Brownian motion from the Langevin equation as the overdamped limit without a force field.

Now we go back to the rescaling to a more representative time scale and deduce the overdamped approximation at the limit $\nu \rightarrow \infty$. More details in [Section 6.5](https://doi.org/10.1007/978-1-4939-1323-7_6) of [Grigorios Pavliotis (2014)](https://doi.org/10.1007/978-1-4939-1323-7). Making the change of time scale to
```math
    \tilde t = \frac{t}{\nu},
```
we write the new position as
```math
    {\tilde X}_{\tilde t} = X_t = X_{\nu \tilde t}.
```
Thus, the velocity satisfies
```math
    {\tilde V}_{\tilde t} = \frac{\mathrm{d}{\tilde X}_{\tilde t}}{\mathrm{d}\tilde t} = \nu V_t.
```
Since ${\tilde X}_{\tilde t} = X_t$, we set
```math
    \tilde U(\tilde x) = U(x).
```
The differentials satisfy
```math
    \begin{align*}
        \mathrm{d}\tilde t & = \frac{1}{\nu}\;\mathrm{d}t, \\
        \mathrm{d}{\tilde X}_{\tilde t} & = \mathrm{d}X_t, \\
        \nabla_{\tilde x}\tilde U(\tilde x) & = \nabla_x U(x), \\
        \mathrm{d}{\tilde V}_{\tilde t} & = \nu\;\mathrm{d}V_t.
    \end{align*}
```

The Wiener noise $\tilde W_{\tilde t} = W_{t} = W_{\nu \tilde t}$ satifies the scaling
```math
    \mathrm{d}\tilde W_{\tilde t} = \sqrt{\nu}\;\mathrm{d}W_t.
```

Thus,
```math
    \mathrm{d}\tilde X_{\tilde t} = \mathrm{d}X_t = V_t\;\mathrm{d}t = \frac{1}{\nu} \tilde V_{\tilde t} \nu \mathrm{d}\tilde t = V_{\tilde t}\;\mathrm{d}\tilde t,
```
and
```math
    \begin{align*}
        \mathrm{d}\tilde V_{\tilde t} & = \nu\;\mathrm{d}V_t = \nu\left((-\nu V_t - \nabla_x U(X_t))\;\mathrm{d}t + \sigma \;\mathrm{d}W_t \right) \\
        & = \nu \left( \left( -\tilde V_{\tilde t} - \nabla_{\tilde x} \tilde U(\tilde X_{\tilde t}) \right)\nu \mathrm{d}\tilde t + \sigma \sqrt{\nu}\;\tilde W_{\tilde t} \right) \\
        & = \nu^2\left( -\tilde V_{\tilde t} - \nabla_{\tilde x}\tilde U(\tilde X_{\tilde t})\right)\mathrm{d}\tilde t + \nu^{3/2}\sigma\;\mathrm{d}\tilde W_{\tilde t}.
    \end{align*}
```
Since $\sigma = \sqrt{2\nu k_B T / m}$, this becomes
```math
    \mathrm{d}\tilde V_{\tilde t} = \nu^2\left( \left(-\tilde V_{\tilde t} - \nabla_{\tilde x}\tilde U(\tilde X_{\tilde t})\right)\mathrm{d}\tilde t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}\tilde W_{\tilde t} \right)
```
Dividing by $\nu^2$, we find the rescaled system
```math
    \begin{cases}
        \mathrm{d}\tilde X_t = \tilde V_{\tilde t}\;\mathrm{d}\tilde t, \\
        \frac{1}{\nu^2}\mathrm{d}\tilde V_t = \left(-\tilde V_{\tilde t} - \nabla_{\tilde x}\tilde U(\tilde X_{\tilde t})\right)\mathrm{d}\tilde t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}\tilde W_{\tilde t}.
    \end{cases}
```

Letting $\nu \rightarrow \infty$ on the second equation yields
```math
    0 = \left(-\tilde V_{\tilde t} - \nabla_{\tilde x}\tilde U(\tilde X_{\tilde t})\right)\mathrm{d}\tilde t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}\tilde W_{\tilde t}
```
Substituting $\mathrm{d}\tilde X_t = \tilde V_{\tilde t}\;\mathrm{d}\tilde t$ and moving this term to the left hand side lead to
```math
    \mathrm{d}\tilde X_t = - \nabla_{\tilde x}\tilde U(\tilde X_{\tilde t})\;\mathrm{d}\tilde t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}\tilde W_{\tilde t}.
```
Hence, we obtain the overdamped Langevin equation. Dropping the tildes, we write the equation as
```math
    \mathrm{d} X_t = - \nabla U(X_t)\;\mathrm{d}t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}W_{t}.
```

## The limit distribution

In the inviscid and deterministic case, the Langevin equation reads
```math
    m \ddot x_t = - \nabla U(x_t),
```
and the level sets
```math
    \left\{(x, v); \frac{1}{2}v^2 + U(x) = c\right\}
```
of the *total energy* are invariant by the solution group. In the viscous deterministic case
```math
    m \ddot x_t = - \mu \dot x_t - \nabla U(x_t),
```
the solutions tend to the equilibria states $\min_x U(x)$, or more precisely to the variety
```math
    \{(x, v); v = \nabla U(x) = 0\}.
```
In the full viscous, stochastic equation, the tendency to go to the equilibria is balanced by the diffusion, and the system tends to a *stochastic* equilibrium represented by a distribution with probability density function
```math
    p_U(x, v) = \frac{1}{Z} e^{-\frac{m}{k_B T}U(x) + \frac{v^2}{2}},
```
where $Z$ is a normalization constant to have $p_U(x, v)$ integrate to $1$.

A similar behavior occurs in the overdamped Langevin equation, for which the equilibrium distribution is given by
```math
    p_U(x) = \frac{1}{Z_0} e^{-\frac{m}{k_B T} U(x)},
```
where $Z_0$ is another normalization constant.

We now discuss in more details the equilibrium distribution in the overdamped equation. For that, we look at the Fokker-Plank equation for
```math
    \mathrm{d} X_t = - \nabla U(X_t)\;\mathrm{d}t + \sqrt{\frac{2k_B T}{m}}\;\mathrm{d}W_{t},
```
In general, a stochastic differential equation of the form
```math
    \mathrm{d} X_t = \mu(t, X_t) \;\mathrm{d}t + \sigma(t, X_t)\;\mathrm{d}W_{t}
```
is associated with the Fokker-Planck equation
```math
    \frac{\partial}{\partial t} p(t, x) = - \nabla_x \cdot \left(\mu(t, x) p(t, x)\right) + \frac{1}{2} \Delta_x \left( \sigma(t, x)^2 p(t, x) \right),
```
where $p(t, x)$ is such that $x \mapsto p(t, x)$ is the probability density function of the solution $X_t$, i.e. the marginal distribution at time $t$ of the process $\{X_t\}_t$. For the overdamped Langevin equation above, the Fokker-Planck equation takes the form
```math
    \frac{\partial}{\partial t} p(t, x) = \nabla_x \cdot \left(\nabla U(x) p(t, x)\right) + \frac{k_B T}{m}\Delta_x p(t, x),
```
which can also be written as
```math
    \frac{\partial}{\partial t} p(t, x) = \nabla_x \cdot \left(\nabla U(x) p(t, x) + \frac{k_B T}{m}\nabla_x p(t, x) \right),
```

At the equilibrium, the probability density function $p(t, x) = p_\infty(x)$ is independent of time and satisfies
```math
    \nabla_x \cdot \left(\nabla U(x) p_\infty(x) + \frac{k_B T}{m}\nabla_x p_\infty(x) \right) = 0.
```

In the one-dimensional case, this means that
```math
    \frac{\partial U(x)}{\partial x} p_\infty(x) + \frac{k_B T}{m}\frac{\partial p_\infty(x)}{\partial t} = C,
```
for some constant $C$. This can be written in the form of the first order differential equation
```math
    \frac{\partial p_\infty(x)}{\partial t} + \frac{m}{k_B T}\frac{\partial U(x)}{\partial x} p_\infty(x) = \frac{m}{k_B T} C.
```
This equation can be solve using the integrating factor
```math
    e^{\frac{m}{k_B T} U(x)}.
```
But assuming that the terms on the left hand side of the first order differential equation vanish when $|x| \rightarrow \infty$, the constant $C$ must be zero. In this case, the solution simplifies to
```math
    p_\infty(x) = C_0 e^{-\frac{m}{k_B T} U(x)},
```
for some other constant $C_0$, which we may write as $C_0 = 1/Z_0$ and obtain the expression
```math
    p_\infty(x) = \frac{1}{Z_0} e^{-\frac{m}{k_B T} U(x)}.
```
The constant $Z_0$ is a normalization factor to yield that $p(x)$ is a proper probability density function, with total mass $1$, i.e.
```math
    Z_0 = \int_{-\infty}^\infty e^{-\frac{m}{k_B T} U(x)}\;\mathrm{d}x.
```

One can check that the PDF above is also a solution in the multi-dimensional case.

When the potential $U=U(x)$ grows sufficiently rapidly as $|x|\rightarrow \infty$, this distribution is the unique equilibrium. In the case of thermodynamics, this corresponds to thermodynamic equilibrium and this is known as the *Boltzmann distribution.* The condition that the potential grows sufficiently rapidly at infinite means that the potential well is deep enough to confine the particle.

Abstracting away from the physical model and considering the overdamped Langevin equation in the form
```math
    \mathrm{d} X_t = - \nabla U(X_t)\;\mathrm{d}t + \sqrt{2\gamma}\;\mathrm{d}W_{t},
```
for some constant $\gamma > 0$, then the Fokker-Planck equation reads
```math
    \frac{\partial}{\partial t} p(t, x) = \nabla_x \cdot \left(\nabla U(x) p(t, x)\right) + \gamma\Delta_x p(t, x),
```
and the stationary distribution takes the form
```math
    p_\infty(x) = \frac{1}{Z_0} e^{-\frac{U(x)}{\gamma}}.
```

## Convergence to the limit distribution

There are many rigorous results concerning the convergence to the equilibrium distribution, discussing conditions for the convergence, metrics, and rate of convergence. We will discuss them in more details in due course.

## Sampling from the score function via Langevin dynamics

Now, suppose we take for the potential $U=U(x)$ minus a multiple of the logpdf of a random variable $X$ with probability density function $p_X(x)$ up to an arbitrary constant, i.e.
```math
    U(x) = - \gamma \log p_X(x) + C,
```
for some constants $\gamma$ and $C$. The score function is connected to the gradient of the potential by
```math
    \psi_X(x) = \nabla_x \log p_X(x) = - \frac{1}{\gamma} \nabla_x U(x).
```
In this case, we can write the PDF as
```math
    p_X(x) = e^{\log p_X(x)} = e^{-U(x)/\gamma + C/\gamma} = \frac{1}{Z} e^{-U(x)/\gamma},
```
for a (normalizing) constant $Z = e^{-C/\gamma}$. Then we see that the PDF $p_X(x)$ is exactly the equilibrium of the Langevin equation
```math
    \mathrm{d}X_t = -\nabla U(X_t)\;\mathrm{d}t + \sqrt{2\gamma}\mathrm{d}W_t.
```
which can be written as
```math
    \mathrm{d}X_t = \gamma\psi_X(X_t)\;\mathrm{d}t + \sqrt{2\gamma}\mathrm{d}W_t.
```

This lead to a sampling method to draw samples from a distribution using its score function, as introduced by [Gareth Roberts and Richard Tweedie (1996)](https://doi.org/10.2307/3318418).

As mentioned above, questions about the conditions for the convergence, rate of converge and convergence metric are of great importance, and they are also important for sampling purposes. Other relevant question concern the stability of the equilibrium solution, when for instance an approximate score function is used. This is also relevant when the modeled score function (say via a neural network) might even not be exactly the gradient of a potential. We will leave these questions for another opportunity.

## One-dimensional numerical example

We first illustrate the Langevin sampling by drawing samples for a univariate Gaussian mixture distribution.

```@setup scoreandlangevin
xrange = range(0.0, 10.0, 200)
ns = [(3, 1), (7, 0.5)]
ps = [0.6, 0.4]
prob = MixtureModel([Normal(mu, sigma) for (mu, sigma) in ns], ps)
prob_score = gradlogpdf.(prob, xrange)
```

```@example scoreandlangevin
Markdown.parse("""The Gaussian mixture distribution is composed of Normal distributions ``\\mathcal{N}($(ns[1][1]), $(ns[1][2]))`` and ``\\mathcal{N}($(ns[2][1]), $(ns[2][2]))``, with weights $(ps[1]) and $(ps[2]), respectively. The PDF and score functions look as follows.""") # hide
plt1 = plot(xrange, x -> pdf(prob, x), title="PDF of the Gaussian mixture", titlefont=10, legend=false) # hide
plt2 = plot(xrange, x -> gradlogpdf(prob, x), title="score function of the Gaussian mixture", titlefont=10, legend=false) # hide
plot(plt1, plt2, size=(600, 600), layout=(2, 1)) # hide
```

```@setup scoreandlangevin
gamma = 1/2
mu0 = 5
sigma0 = 1
prob0 = Normal(5, 1)
# prob0 = Uniform(1, 7)
t0 = 0
tf = 10
```

```@example scoreandlangevin
Markdown.parse("""Now we draw samples from it using the overdamped Langevin equation with ``\\gamma = $gamma``. We start with samples from a normal distribution ``\\mathcal{N}($mu0, $sigma0^2)`` and evolve them according to the equation, from the initial time ``t_0 = $t0`` up to time ``t_f = $tf``.""") # hide
```

```@setup scoreandlangevin
params = (prob=prob, gamma=gamma)
f(t, x, params) = params.gamma * gradlogpdf(params.prob, x)
g(t, x, params) = sqrt(2 * params.gamma)

p0 = pdf.(prob0, xrange)
tt, pt = solve_fokkerplanck(p0, t0, tf, xrange, f, g, params, snapshots=200)

x0 = rand(rng, prob0, 200)
xt = solve_sde_1d(rng, x0, tt, f, g, params)
# nothing
```

Here is $tx$ plot of the ensemble of solutions of the stochastic overdamped Langevin equation.

```@example scoreandlangevin
plot(tt, xt, legend=false) # hide
```

We obtain the following histogram at the final time:

```@example scoreandlangevin
histogram(xt[end, :], nbins=40, xlims=extrema(xrange), label="normalized histogram", normalize=true) # hide
plot!(xrange, pt[end, :], label="Langevin evolved PDF") # hide
plot!(xrange, x -> pdf(prob, x), label="desired PDF") # hide
```

We may also visualize a surface plot and a heatmap of the evolution of the Fokker-Planck equation for the density.

```@example scoreandlangevin
surface(tt, xrange, pt', color=:vik, title="Surface plot of the evolution of the PDF", titlefont=10, xlabel="t", ylabel="x") # hide
```

```@example scoreandlangevin
heatmap(tt, xrange, pt', color=:vik, title="Heatmap of the evolution of the PDF", titlefont=10, xlabel="t", ylabel="x") # hide
```

With an animation just for fun.

```@setup scoreandlangevin
anim = @gif for i in axes(pt, 1)
    histogram(view(xt, i, :), nbins=40, normalize=:pdf,legend=false)
    plot!(xrange, view(pt, i, :))
    plot!(xrange, x -> pdf(prob, x), title="t=$(round(tt[i], digits=3))", xlims=extrema(xrange), ylims=(0.0, 0.5), legend=false)
end
```

```@example scoreandlangevin
anim # hide
```

## Two-dimensional numerical example

Let's do a two-dimensional example, now. We consider a trimodel bivariate normal distribution and use the two-dimensional overdamped Langevin equations to obtain samples from the score function of the distribution.

```@setup scoreandlangevin2d
using StatsPlots
using Random
using Distributions
using Markdown

rng = Xoshiro(123)
```

```@setup scoreandlangevin2d
function Distributions.gradlogpdf(d::MultivariateMixture, x::AbstractVector{<:Real})
    ps = probs(d)
    cs = components(d)

    # `d` is expected to have at least one distribution, otherwise this will just error
    psi, idxps = iterate(ps)
    csi, idxcs = iterate(cs)
    pdfx1 = pdf(csi, x)
    pdfx = psi * pdfx1
    glp = pdfx * gradlogpdf(csi, x)
    if iszero(psi)
        fill!(glp, zero(eltype(glp)))
    end
    
    while (iterps = iterate(ps, idxps)) !== nothing && (itercs = iterate(cs, idxcs)) !== nothing
        psi, idxps = iterps
        csi, idxcs = itercs
        if !iszero(psi)
            pdfxi = pdf(csi, x)
            if !iszero(pdfxi)
                pipdfxi = psi * pdfxi
                pdfx += pipdfxi
                glp .+= pipdfxi .* gradlogpdf(csi, x)
            end
        end
    end
    if !iszero(pdfx) # else glp is already zero
        glp ./= pdfx
    end 
    return glp
end
```

```@setup scoreandlangevin2d
function solve_sde!(rng, xt, x0, tt, f, g, params)
    @assert axes(xt) == (axes(x0)..., eachindex(tt))
    # in particular, this implies firstindex(xt, 3) == firstindex(tt)

    dt = Float64(tt.step)
    sqrtdt = √dt

    xt[:, :, 1] .= x0
    k1 = firstindex(tt)
    kend = lastindex(tt)
    @inbounds for k in Iterators.drop(eachindex(tt), 1)
        randn!(rng, view(xt, :, :, lastindex(tt)))
        for j in axes(xt, 2)
            xt[:, j, k] .= view(xt, :, j, k1) .+ f(tt[k1], view(xt, :, j, k1), params) .* dt .+ g(tt[k1], view(xt, :, j, k1), params) .* sqrtdt .* view(xt, :, j, kend)
        end
        k1 = k
    end
    return xt
end

function solve_sde(rng, x0, tt, f, g, params)
    
    xt = zeros(size(initial_sample)..., length(tt))

    solve_sde!(rng, xt, x0, tt, f, g, params)

    return xt
end
```

Here is a visualization of the distribution and its score vector field.

```@setup scoreandlangevin2d
xrange = range(-8, 8, 120)
yrange = range(-8, 8, 120)
dx = Float64(xrange.step)
dy = Float64(yrange.step)

target_prob = MixtureModel([MvNormal([-3, -3], [1 0; 0 1]), MvNormal([3, 3], [1 0; 0 1]), MvNormal([-1, 1], [1 0; 0 1])], [0.4, 0.4, 0.2])

target_pdf = [pdf(target_prob, [x, y]) for y in yrange, x in xrange]
target_score = reduce(hcat, gradlogpdf(target_prob, [x, y]) for y in yrange, x in xrange)
```

```@setup scoreandlangevin2d
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
xx, yy = meshgrid(xrange[begin:8:end], yrange[begin:8:end])
uu = reduce(hcat, gradlogpdf(target_prob, [x, y]) for (x, y) in zip(xx, yy))
```

```@example scoreandlangevin2d
heatmap(xrange, yrange, (x, y) -> pdf(target_prob, [x, y]), title="pdf (heatmap) and score function (vector field)", titlefont=10, legend=false, color=:vik) # hide
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5) # hide
```

```@setup scoreandlangevin2d
gamma = 1/2
t0 = 0
tf = 20
nt = 128
tt = range(t0, tf, length=nt+1)
```

```@setup scoreandlangevin2d
params = (prob=target_prob, gamma=gamma)
f(t, x, params) = params.gamma .* gradlogpdf(params.prob, x)
g(t, x, params) = sqrt(2 * params.gamma)
```

```@setup scoreandlangevin2d
initial_prob = MvNormal([0, 0], [1 0; 0 1])
initial_sample = rand(rng, initial_prob, 256)
```

```@example scoreandlangevin2d
Markdown.parse("""Now we draw samples from it using the overdamped Langevin equation with ``\\gamma = $gamma``, and starting with samples from a standard normal distribution, evolving the particles from the initial time ``t_0 = $t0`` up to time ``t_f = $tf``.\n""") # hide
```

```@setup scoreandlangevin2d
xt = solve_sde(rng, initial_sample, tt, f, g, params)
```

```@example scoreandlangevin2d
heatmap(xrange, yrange, (x, y) -> pdf(target_prob, [x, y]), title="pdf (heatmap), score function (vector field)and particle sample", titlefont=10, legend=false, color=:vik) # hide
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5) # hide
scatter!(xt[1, :, end], xt[2, :, end], markersize=2, markercolor=:white) # hide
```

Observe how many particles get trapped near the smallest modal point in the middle.

Let us see a animation for fun.
```@setup scoreandlangevin2d
anim = @gif for k in axes(xt, 3)
    heatmap(xrange, yrange, (x, y) -> pdf(target_prob, [x, y]), title="pdf, score function, and particles at t = $(round(tt[k],digits=2))", titlefont=10, legend=false, color=:vik)
    quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
    scatter!(xt[1, :, k], xt[2, :, k], markersize=2, markercolor=:white)
end
```

```@example scoreandlangevin2d
anim # hide
```

Now we draw samples starting from a uniform distribution of points.

```@setup scoreandlangevin2d
initial_sample = 12 .* (rand(rng, 2, 256) .- [0.5, 0.5])
```

```@setup scoreandlangevin2d
xt = solve_sde(rng, initial_sample, tt, f, g, params)
```

Here is what we get.

```@example scoreandlangevin2d
heatmap(xrange, yrange, (x, y) -> pdf(target_prob, [x, y]), title="pdf (heatmap), score function (vector field)and particle sample", titlefont=10, legend=false, color=:vik) # hide
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5) # hide
scatter!(xt[1, :, end], xt[2, :, end], markersize=2, markercolor=:white) # hide
```

Again, let us see a animation.
```@setup scoreandlangevin2d
anim = @gif for k in axes(xt, 3)
    heatmap(xrange, yrange, (x, y) -> pdf(target_prob, [x, y]), title="pdf, score function, and particles at t = $(round(tt[k],digits=2))", titlefont=10, legend=false, color=:vik)
    quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
    scatter!(xt[1, :, k], xt[2, :, k], markersize=2, markercolor=:white)
end
```

```@example scoreandlangevin2d
anim # hide
```

## Score function in the Julia language

The distributions and their pdf are obtained from the [JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package. The score function is also implemented in [JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl) as `gradlogpdf`, but only for some distributions. Since we are interested on Gaussian mixtures, we did some *pirating* and extended `Distributions.gradlogpdf` to *univariate* `MixtureModels`, both univariate and multivariate. These are the codes for that.

```julia
function Distributions.gradlogpdf(d::UnivariateMixture, x::Real)
    ps = probs(d)
    cs = components(d)
    ps1 = first(ps)
    cs1 = first(cs)
    pdfx1 = pdf(cs1, x)
    pdfx = ps1 * pdfx1
    glp = pdfx * gradlogpdf(cs1, x)
    if iszero(ps1)
        glp = zero(glp)
    end
    @inbounds for (psi, csi) in Iterators.drop(zip(ps, cs), 1)
        if !iszero(psi)
            pdfxi = pdf(csi, x)
            if !iszero(pdfxi)
                pipdfxi = psi * pdfxi
                pdfx += pipdfxi
                glp += pipdfxi * gradlogpdf(csi, x)
            end
        end
    end
    if !iszero(pdfx) # else glp is already zero
        glp /= pdfx
    end 
    return glp
end
```

```julia
function Distributions.gradlogpdf(d::MultivariateMixture, x::AbstractVector{<:Real})
    ps = probs(d)
    cs = components(d)

    # `d` is expected to have at least one distribution, otherwise this will just error
    psi, idxps = iterate(ps)
    csi, idxcs = iterate(cs)
    pdfx1 = pdf(csi, x)
    pdfx = psi * pdfx1
    glp = pdfx * gradlogpdf(csi, x)
    if iszero(psi)
        fill!(glp, zero(eltype(glp)))
    end
    
    while (iterps = iterate(ps, idxps)) !== nothing && (itercs = iterate(cs, idxcs)) !== nothing
        psi, idxps = iterps
        csi, idxcs = itercs
        if !iszero(psi)
            pdfxi = pdf(csi, x)
            if !iszero(pdfxi)
                pipdfxi = psi * pdfxi
                pdfx += pipdfxi
                glp .+= pipdfxi .* gradlogpdf(csi, x)
            end
        end
    end
    if !iszero(pdfx) # else glp is already zero
        glp ./= pdfx
    end 
    return glp
end
```

## References

1. [R. Brown (1828), "A brief account of microscopical observations made in the months of June, July and August, 1827, on the particles contained in the pollen of plants; and on the general existence of active molecules in organic and inorganic bodies". Philosophical Magazine. 4 (21), 161-173. doi:10.1080/14786442808674769](https://doi.org/10.1080%2F14786442808674769)
2. [A. Einstein (1905), "Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen" [On the Movement of Small Particles Suspended in Stationary Liquids Required by the Molecular-Kinetic Theory of Heat], Annalen der Physik, 322 (8), 549-560, doi:10.1002/andp.19053220806](https://doi.org/10.1002/andp.19053220806)
3. [P. Mörters and Y. Peres (2010), "Brownian motion", Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge. With an appendix by Oded Schramm and Wendelin Werner](https://www.cambridge.org/il/academic/subjects/statistics-probability/probability-theory-and-stochastic-processes/brownian-motion?format=HB&isbn=9780521760188)
4. [P. Langevin (1908), "Sur la théorie du mouvement brownien [On the Theory of Brownian Motion]". C. R. Acad. Sci. Paris. 146: 530–533](https://gallica.bnf.fr/ark:/12148/bpt6k3100t/f530.item) 
5. [G. A. Pavliotis (2014), "The Langevin Equation. In: Stochastic Processes and Applications". Texts in Applied Mathematics, vol 60. Springer, New York, NY, doi:10.1007/978-1-4939-1323-7_6](https://doi.org/10.1007/978-1-4939-1323-7)
6. [G. O. Roberts, R. L. Tweedie (1996), "Exponential Convergence of Langevin Distributions and Their Discrete Approximations", Bernoulli, Vol. 2, No. 4, 341-363, doi:10.2307/3318418](https://doi.org/10.2307/3318418)
