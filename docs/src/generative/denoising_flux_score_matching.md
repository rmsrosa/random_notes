# Denoising score-matching in Flux.jl

```@meta
Draft = true
```

## Introduction

### Aim

### Motivation

## Numerical Example

### Julia language setup

We use the [Julia programming language](https://julialang.org) with suitable packages.

### Packages

```@example kdescorematching
using StatsPlots
using Random
using Distributions
using Flux

nothing # hide
```

There are several Julia libraries for artificial neural networks and for automatic differentiation (AD). The most established package for artificial neural networks is the [FluxML/Flux.jl](https://github.com/FluxML/Flux.jl) library, which handles the parameters implicitly. A newer library that handles the parameters explicitly is the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library, which is also taylored to the differential equations [SciML](https://sciml.ai) ecosystem.

There are heated discussions, in the community, about the pros and cons of both implicit and explicit parametrized models, but we do not go into these here. Since we aim to combine score-matching with neural networks and, eventually, with stochastic differential equations, we thought it was a reasonable idea to experiment with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library, regardless of it being explicit or not.

As we mentioned, the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library is a newer package and not as well developed. In particular, it seems the only AD that works with it is the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library. Unfortunately for our use case, the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library is not fit to do AD on top of AD, as one can see from e.g. [Zygote: Design limitations](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1). Nevertheless, since at the end it is more efficient to approximate the derivative of the modeled score function by finite differences, we do not let that stop us short from using the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl)/[FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) combination.

### Reproducibility

We set the random seed for reproducibility purposes.

```@example kdescorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup kdescorematching
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

### Data

Now we build the target model and draw samples from it.

The target model is a univariate random variable denoted by $X$ and defined by a probability distribution. Associated with that we consider its PDF and its score-function.

We played with different models, but settled here with one of those used in [Eric J. Ma (2021)](https://ericmjl.github.io/score-models/).

We need enough sample points to capture the transition region in the mixture of Gaussians.

```@example kdescorematching
xrange = range(-8, 8, 100)
dx = Float64(xrange.step)

target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

sample_points = permutedims(rand(rng, target_prob, 1024))
sample_gradlogpdf = gradlogpdf.(target_prob, sample_points)
```

Notice the data `x` and `sample_points` are defined as row vectors so we can apply the model in batch to all of their values at once. The values `y` are also row vectors for easy comparison with the predicted values. When, plotting, though, we need to revert them to vectors.

For the theoretical discussion, we denote the PDF by $p_X(x)$ and the score function by
```math
    \psi_X(x) = \frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x)).
```

Visualizing the sample data drawn from the distribution and the PDF.
```@example kdescorematching
plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(xrange, x -> pdf(target_prob, x), linewidth=2, label="pdf")
# scatter!(sample_points', s -> pdf(target_prob, s), markersize=2, label="sample")
```

Visualizing the score function.
```@example kdescorematching
plot(title="The score function and the sample", titlefont=10)

plot!(xrange, s -> gradlogpdf(target_prob, s), label="score function", markersize=2)
scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

### The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example kdescorematching
model = f64(Chain(Dense(1 => 8, relu), Dense(8 => 1)))
```

```@example kdescorematching
ps = Flux.params(model)
```

### Loss functions for score-matching

#### Mean squared error loss function

For educational purposes, since we have the pdf and the score function, one of the ways we may train the model is directly on $J({\boldsymbol{\theta}})$ itself. This is also useful to make sure that our network is able to model the desired score function.

Here is how we implement it.
```@example kdescorematching
function loss_function_mse(model, ps, st, data)
    x, y, target_pdf, sigma, sample_points = data
    y_pred = model(x)
    loss = mean(target_pdf .* (y_pred .- y) .^2)
    return loss, st, ()
end

function loss_function_mse(model, x, y)
    y_pred = model(x)
    loss = mean(abs2, y_pred .- y)
    return loss
end
```

#### Loss function

```@example kdescorematching
function loss_function_kde_old(model, ps, st, data)
    x, y, target_pdf, sigma, sample_points = data
    y_pred = model(x)
    loss = mean(abs2, y_pred[i] + (s - xi) / sigma ^ 2 for xi in x for (i, s) in enumerate(sample_points)) / 2
    return loss, st, ()
end
```

```julia
mean(abs2, model(sample_points) .+ (sample_points .- sample_points') ./ sigma ^ 2) / 2
```

```julia
sigma = 0.5
y_score_kse = (sample_points .- sample_points') ./ sigma ^ 2

loss_function_kse(y, y_score_kse) = mean(abs2, y .+ y_score_kse) / 2

loss_function_kse(model(sample_points), y_score_kse)

y_pred_tst = model(sample_points)
mean(abs2, y_pred_tst[i] + (s - xi) / sigma ^ 2 for xi in sample_points for (i, s) in enumerate(sample_points)) / 2

Flux.gradient(m -> loss_function_kse(m(sample_points), y_score_kse), model)
```

### Optimization setup

#### Optimization method

We use the classical Adam optimiser (see [Kingma and Ba (2015)](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8)), which is a stochastic gradient-based optimization method.

```@example kdescorematching
opt = Flux.setup(Adam(), model)
```

### Training

#### Preparing the data

```@example kdescorematching
data = (sample_points, gradlogpdf.(target_prob, sample_points))
data = (sample_points, y_score_kse)
```

```@example kdescorematching
Flux.gradient(m -> loss_function_kse(m(sample_points), y_score_kse), model)
```

#### Training with $J({\boldsymbol{\theta}})$

Now we attempt to train the model, starting with $J({\boldsymbol{\theta}})$.
```@example kdescorematching
Flux.train!(loss_function_kse, model, data, opt)
nothing # hide
```

Testing out the trained model.
```@example kdescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example kdescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup kdescorematching
ymin, ymax = extrema(y)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
    plot(title="Fitting evolution", titlefont=10)

    plot!(x', y', linewidth=4, label="score function")

    scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

    plot!(x', y_pred', linewidth=2, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example kdescorematching
gif(anim, fps = 20) # hide
```

We also visualize the evolution of the losses.
```@example kdescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example kdescorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Vincent]()
1. [D. P. Kingma, J. Ba (2015), Adam: A Method for Stochastic Optimization, In International Conference on Learning Representations (ICLR)](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8) -- see also the [arxiv version](https://arxiv.org/abs/1412.6980)
