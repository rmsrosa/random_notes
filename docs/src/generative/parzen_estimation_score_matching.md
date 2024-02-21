# Score matching with Parzen estimation

```@meta
Draft = false
```

## Introduction

### Aim

Explore the use of *Parzen kernel estimation* to approximate the **explicit score matching**, used as the basis for the **implicit score matching** objective proposed by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) and discussed *en passant* by [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) on their way to the **denoising score matching** objective. We illustrate the method by fiting a multi-layer perceptron to model the score function of a one-dimensional synthetic Gaussian-mixture distribution.

### Motivation

The motivation is to continue building a solid background on score-matching diffusion.

### Background

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) proposed modeling directly the score of a distribution. This is obtained, in theory, by minimizing an **explicit score matching** objective function (i.e. the Fisher divergence). However, this function requires knowing the supposedly unknown target score function. The trick used by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) was then to do an integration by parts and rewrite the optimization problem in terms of an **implicit score matching** objective function, which yields the same minima and does not require further information from the target distribution other than some sample points. The **implicit score matching** method requires, however, the derivative of the model score function, which is costly to compute in general.

Then, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) explored the idea of using *non-parametric Parzen density estimation* to directly approximate the explicit score matching objective, making a connection with denoising autoenconders, and proposing the **denoising (explicit) score matching** method.

We will detail denoising score matching in a separate note. Here, we stop at the Parzen density estimation idea, which was used in [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) only as a step towards the denoising score matching. We call this method the **Parzen estimated (explicit) score matching** method.

## Objective function approximating the explicit score matching objective

The score-matching method from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) aims to fit the model score function $\psi(\mathbf{x}; {\boldsymbol{\theta}})$ to the score function $\psi_X(\mathbf{x})$ of a random variable $\mathbf{X}$ by minimizing the 
**implicit score matching** objective
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
which is equivalent to minimizing the **explicit score matching** objective
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x},
```
due to the following identity obtained via integration by parts in the expectation
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = {\tilde J}_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C,
```
where $C$ is constant with respect to the parameters. The advantage of ${\tilde J}_{\mathrm{ISM}}({\boldsymbol{\theta}})$ is that it does not involve the unknown score function of $X$. It does, however, involve the gradient of the modeled score function, which is expensive to compute.

In practice, this is further approximated by the **empirical distribution** ${\tilde p}_0(\mathbf{x})$ given by
```math
    {\tilde p}_0(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^N \delta(\mathbf{x} - \mathbf{x}_n),
```
so the implemented objective is the **empirical implicit score matching** objective
```math
    {\tilde J}_{\mathrm{ISM}{\tilde p}_0}({\boldsymbol{\theta}}) = \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right).
```

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) briefly mentions that minimizing $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ directly is "basically a non-parametric estimation problem", but dismisses it for the "simple trick of partial integration to compute the objective function very easily". As we have seen, the trick is fine for model functions for which we can compute the gradient without much trouble, but for modeling it with a neural network, for instance, it becomes computationally expensive.

A few years later, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) considered the idea of using a Parzel kernel density estimation
```math
    {\tilde p}_\sigma(\mathbf{x}) = \frac{1}{\sigma^d}\int_{\mathbb{R}^d} K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right) \;\mathrm{d}{\tilde p}_0(\mathbf{x}) = \frac{1}{\sigma^d N}\sum_{n=1}^N K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right),
```
where $\sigma > 0$ is a kernel window parameter and $K(\mathbf{x})$ is a kernel density (properly normalized to have mass one). In this way, the explicit score matching objective function is approximated by the **Parzen-estimated explicit score matching** objective
```math
    {\tilde J}_{\mathrm{PESM_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_\sigma(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x},
```
which is then further approximated with the empirical distribution, yielding the **empirical Parzen-estimated explicit score matching**
```math
    \begin{align*}
        {\tilde J}_{\mathrm{PESM_\sigma, {\tilde p}_0}}({\boldsymbol{\theta}}) & = \frac{1}{2}\int_{\mathbb{R}^d} {\tilde p}_0(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x} \\
        & = \frac{1}{N} \sum_{n=1}^N \left\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}_n}\log {\tilde p}_\sigma(\mathbf{x})\right\|^2.
    \end{align*}
```

However, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) did not use this as a final objective function. Pascal further simplified the objective function ${\tilde J}_{\mathrm{PESM_\sigma}}({\boldsymbol{\theta}})$ by expanding the gradient of the logpdf of the Parzen estimator, writing a double integral with a conditional probability, and switching the order of integration. We will do this in a follow up note, but for the moment we will stop at ${\tilde J}_{\mathrm{PESM_\sigma}}({\boldsymbol{\theta}})$ and ${\tilde J}_{\mathrm{PESM_\sigma, {\tilde p}_0}}({\boldsymbol{\theta}})$, use a Gaussian estimator, and see how this works.

Computing the score function with the Parzen estimation amounts to
```math
    \begin{align*}
        \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}) & = \boldsymbol{\nabla}_{\mathbf{x}}\log\left( \frac{1}{\sigma^d N}\sum_{n=1}^N K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right)\right) \\
        & = \frac{1}{{\tilde p}_\sigma(\mathbf{x})} \frac{1}{\sigma^d N}\sum_{n=1}^N \boldsymbol{\nabla}_{\mathbf{x}} \left(K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right)\right) \\
        & = \frac{1}{{\tilde p}_\sigma(\mathbf{x})} \frac{1}{\sigma^d N}\sum_{n=1}^N \frac{1}{\sigma}\left(\boldsymbol{\nabla}_{\mathbf{x}} K\right)\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right).
    \end{align*}
```
If we use the standard Gaussian kernel
```math
    G(\mathbf{x}) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} \mathbf{x}^2},
```
then
```math
    \left(\boldsymbol{\nabla}_{\mathbf{x}} G\right)(\mathbf{x}) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2} \mathbf{x}^2} \mathbf{x} = G(\mathbf{x})\mathbf{x},
```
so that
```math
    \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}) = \frac{1}{{\tilde p}_\sigma(\mathbf{x})} \frac{1}{\sigma^d N}\sum_{n=1}^N G\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right)\frac{\mathbf{x} - \mathbf{x}_n}{\sigma^2}.
```

Notice that this can be computed beforehand at the sample points, just with the knowledge of the sample points themselves. Indeed, renaming the index above from $n$ to $j$, and computing the approximate score function at each sample point $\mathbf{x}_n$ yield
```math
    \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}_n) = \frac{1}{{\tilde p}_\sigma(\mathbf{x}_n)} \frac{1}{\sigma^d N}\sum_{n=1}^N G\left(\frac{\mathbf{x}_n - \mathbf{x}_j}{\sigma}\right)\frac{\mathbf{x}_n - \mathbf{x}_j}{\sigma^2},
```
where
```math
{\tilde p}_\sigma(\mathbf{x}_n) = \frac{1}{\sigma^d N}\sum_{j=1}^N G\left(\frac{\mathbf{x} - \mathbf{x}_j}{\sigma}\right).
```

Then, the explicit score matching objective approximated with the Parzen kernel estimator and with the empirical distribution yields the objective
```math
    {\tilde J}_{\mathrm{PESM_{\sigma, {\tilde p}_0}}}({\boldsymbol{\theta}}) = \frac{1}{2}\frac{1}{N} \sum_{n=1}^N \left\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log {\tilde p}_\sigma(\mathbf{x}_n)\right\|^2.
```

## Numerical example

We illustrate, numerically, the use of the **empirical Parzen-estimated explicit score matching** objective ${\tilde J}_{\mathrm{PESM_{\sigma, {\tilde p}_0}}}$ to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

We use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages.

#### Packages

```@example simplescorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation
using Markdown

nothing # hide
```

#### Reproducibility

We set the random seed for reproducibility purposes.

```@example simplescorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup simplescorematching
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

We build the target model and draw samples from it.

The target model is a univariate random variable denoted by $X$ and defined by a probability distribution. Associated with that we consider its PDF and its score-function.

```@example simplescorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sample_points = permutedims(rand(rng, target_prob, 1024))
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup simplescorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example simplescorematching
plt # hide
```

Visualizing the score function.
```@setup simplescorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example simplescorematching
plt # hide
```

For the Parzen estimated score matching, we need to pre-compute the score function of the Parzen estimation.
```@example simplescorematching
G(x) = exp(-x^2 / 2) / √(2π)

psigma(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) for xn in sample_points) / sigma

score_parzen(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) * (xn - x) / sigma^2 for xn in sample_points) / psigma(x, sigma, sample_points) / sigma
```

The Parzen estimated score function is highly sensitive to the window parameter $\sigma$:
```@setup simplescorematching
plt = plot(title="The score functions of the target density and kernel density estimations", titlefont=10)

plot!(plt, xrange, target_score', label="score target", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="score at data", markersize=2, alpha=0.5)
for sigma in (0.2, 0.5, 1.0)
    plot!(plt, range(-6, 6, length=200), x -> score_parzen(x, sigma, sample_points), label="score Parzen \$\\sigma=$sigma\$", markersize=2)
end
```

```@example simplescorematching
plt # hide
```

```@example simplescorematching
sigma = 0.5
score_parzen_points = map(x -> score_parzen(x, sigma, sample_points), sample_points)
data = (sample_points, score_parzen_points)
```

```@example simplescorematching
Markdown.parse("""We choose the value ``\\sigma = $sigma``.""") # hide
```

```@setup simplescorematching
plt = plot(title="The score functions of the target density and the kernel density estimation", titlefont=10, legend=:bottomleft)

plot!(plt, xrange, target_score', label="score target", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="score at data", markersize=2, alpha=0.5)
scatter!(plt, sample_points', score_parzen_points', label="score Parzen \$\\sigma=$sigma\$", markersize=2)
```

```@example simplescorematching
plt # hide
```

### The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example simplescorematching
model = Chain(Dense(1 => 8, relu), Dense(8 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example simplescorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

Here it is how we implement the objective ${\tilde J}_{\mathrm{PESM_{\sigma, {\tilde p}_0}}}({\boldsymbol{\theta}})$.
```@example simplescorematching
function loss_function_parzen(model, ps, st, data)
    sample_points, score_parzen_points = data
    y_score_pred, st = Lux.apply(model, sample_points, ps, st)
    loss = mean(abs2, y_score_pred .- score_parzen_points)
    return loss, st, ()
end
```

### Optimization setup

#### Optimization method

We use the Adam optimiser.

```@example simplescorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example simplescorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example simplescorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example simplescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_parzen, data, tstate_org)
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example simplescorematching
function train(tstate::Lux.Experimental.TrainState, vjp, data, loss_function, epochs, numshowepochs=20, numsavestates=0)
    losses = zeros(epochs)
    tstates = [(0, tstate)]
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
            loss_function, data, tstate)
        if ( epochs ≥ numshowepochs > 0 ) && rem(epoch, div(epochs, numshowepochs)) == 0
            println("Epoch: $(epoch) || Loss: $(loss)")
        end
        if ( epochs ≥ numsavestates > 0 ) && rem(epoch, div(epochs, numsavestates)) == 0
            push!(tstates, (epoch, tstate))
        end
        losses[epoch] = loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate, losses, tstates
end
```

### Training

Now we train the model with the objective function ${\tilde J}_{\mathrm{PESM_{\sigma, {\tilde p}_0}}}({\boldsymbol{\theta}})$.
```@example simplescorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_parzen, 500, 20, 125)
nothing # hide
```

### Results

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(xrange, target_score', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(xx', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup simplescorematching
ymin, ymax = extrema(target_score)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
    plot(title="Fitting evolution", titlefont=10)

    plot!(xrange, target_score', linewidth=4, label="score function")

    scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, y_pred', linewidth=2, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example simplescorematching
gif(anim, fps = 20) # hide
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(xrange, target_pdf', label="original")
plot!(xrange, pdf_pred', label="recoverd")
```

And the animation of the evolution of the PDF.
```@setup simplescorematching
ymin, ymax = extrema(target_pdf)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_pred) * dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot(title="Fitting evolution", titlefont=10, legend=:topleft)

    plot!(xrange, target_pdf', linewidth=4, fill=true, alpha=0.3, label="PDF")

    scatter!(sample_points', s -> pdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, pdf_pred', linewidth=2, fill=true, alpha=0.3, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example simplescorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
