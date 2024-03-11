# Stein score function

```@meta
    Draft = false
```

```@setup steinscore
using StatsPlots
using Random
using Distributions
using Markdown

rng = Xoshiro(123)
```

```@setup steinscore
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

```@setup steinscore
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

## Aim

Revisit the origin of the score function. The Stein score function is the basis of score-based generative models.

## The Stein score function

Given a random variable $\mathbf{X}$ in $\mathbb{R}^d$, $d\in\mathbb{N},$ we denote its pdf by $p_X(\mathbf{x})$, while its **(Stein) score function**, also known as **gradlogpdf**, is defined by
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

```@setup steinscore
xrange = range(0, 10, length=200)
mu = 4
sigma = 0.5
prob = Normal(mu, sigma)
plt1 = plot(xrange, x -> pdf(prob, x), title="pdf of \$\\mathcal{N}($mu, $(sigma)^2)\$", titlefont=8)
plt2 = plot(xrange, x -> logpdf(prob, x), title="logpdf of \$\\mathcal{N}($mu, $(sigma)^2)\$", titlefont=8)
plt3 = plot(xrange, x -> gradlogpdf(prob, x), title="gradlogpdf (score) of \$\\mathcal{N}($mu, $(sigma)^2)\$", titlefont=8)
```

```@example steinscore
plot(plt1, plt2, plt3, layout=(3, 1), legend=false, size=(600, 400)) # hide
```

In the multivariate case, the score function is a *vector field* in the event space $\mathbb{R}^d$.

```@setup steinscore
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

```@example steinscore
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
This notion measures the sensitivity of the model with respect to changes in the parameters and is useful, for instance, in the maximization of the maximum likelyhood function fitting a parametrized distribution to data.

The score function given by the gradlogpdf of a distribution is, on the other hand, useful for drawing samples via Langevin dynamics.

## Score function in the Julia language

The distributions and their pdf are obtained from the [JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package. The score function is also implemented in [JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl) as `gradlogpdf`, but only for some distributions. Since we are interested on Gaussian mixtures, we did some *pirating* and extended `Distributions.gradlogpdf` to `MixtureModels`, both univariate and multivariate. These are the codes for that.

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

1. [C. Stein(1972), "A bound for the error in the Normal approximation to the distribution of a sum of dependent random variables", Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, 583-602](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Sixth-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/A-bound-for-the-error-in-the-normal-approximation-to/bsmsp/1200514239)
1. [Q. Liu, J. Lee, M. Jordan (2016), "A kernelized Stein discrepancy for goodness-of-fit tests", Proceedings of The 33rd International Conference on Machine Learning, PMLR 48, 276-284](https://proceedings.mlr.press/v48/liub16.html)