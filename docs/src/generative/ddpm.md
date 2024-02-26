# Denoising diffusion probabilistic models

```@meta
Draft = false
```

## Introduction

### Aim

Review the **denoising diffusion probabilistic models** introduced in [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and further improved in [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html).

### Motivation

Build a solid foundation on generative diffusion models, for which DDPM is an integral part as a discretized analog of the SDE model.

### Background

The main idea in [Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) is to embed the random variable we want to model into a Markov chain and model the whole Markov chain. This is a much more complex task and greatly enlarge the dimension of the problem, but which yields more stability to the training and the generative processes. The desired random variable, for which we only have access to a sample, is considered as an initial condition to a Markov chain converging to a simple and tractable distribution, usually a normal distribution. The training process fits a model to the whole Markov chain. Then, the model is used to reverse the process and generate (aproximate) samples of our target distribution from samples of the tractable distribution. The tractable final distribution becomes the initial distribution of the reverse process, and the initial desired target distribution becomes the final distribution.

## Numerical example

We illustrate the method, numerically, to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

As usual, we use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages and set the seed for reproducibility purposes.

```@setup ddpmscorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation
using Markdown

nothing # hide
```

There are several Julia libraries for artificial neural networks and for automatic differentiation (AD). The most established package for artificial neural networks is the [FluxML/Flux.jl](https://github.com/FluxML/Flux.jl) library, which handles the parameters implicitly, but it is moving to explicit parameters. A newer library that handles the parameters explicitly is the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library, which is taylored to the differential equations [SciML](https://sciml.ai) ecosystem.

Since we aim to combine score-matching with neural networks and, eventually, with stochastic differential equations, we thought it was a reasonable idea to experiment with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library.

As we mentioned, the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library is a newer package and not as well developed. In particular, it seems the only AD that works with it is the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library. Unfortunately, the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library is not so much fit to do AD on top of AD, as one can see from e.g. [Zygote: Design limitations](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1). Thus we only illustrate this with a small network on a simple univariate problem.

#### Reproducibility

We set the random seed for reproducibility purposes.

```@setup ddpmscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup ddpmscorematching
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

```@setup ddpmscorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sample_points = permutedims(rand(rng, target_prob, 1024))
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup ddpmscorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example ddpmscorematching
plt # hide
```

Visualizing the score function.
```@setup ddpmscorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example ddpmscorematching
plt # hide
```

```@example ddpmscorematching
G(x) = exp(-x^2 / 2) / √(2π)

psigma(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) for xn in sample_points) / sigma

score_parzen(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) * (xn - x) / sigma^2 for xn in sample_points) / psigma(x, sigma, sample_points) / sigma
```

```@example ddpmscorematching
sigma = 0.5
score_parzen_points = map(x -> score_parzen(x, sigma, sample_points), sample_points)
data = (sample_points, score_parzen_points)
```

### Markov chain

Now we evolve the sample as a initial time of a Markov chain $\{\mathbf{X}_k\}_{k=0, 1, \ldots, k_f}$

```math
    X_{k+1} \sim \mathcal{N}(\sqrt{1 - \beta_k} X_k, \beta_k \mathbf{I}),
```
where $\{\beta_k\}_{k=0}^{k_f}$.

```math
    X_{k+1}^2 = (1 - \beta_k)X_k^2
```
Assuming $\Delta k = 1$, we can write
```math
    \frac{X_{k+1}^2 - X_k^2}{\Delta k} = - \beta_k X_k^2,
```
which is like an exponential energy decay.

### The neural network model

The neural network we consider is a again a feed-forward neural network made, but now it is a two-dimensional model, since it takes both the variate $x$ and the discrete time $n$, to account for the evolution of the Markov chain.

```@example ddpmscorematching
model = Chain(Dense(1 => 8, sigmoid), Dense(8 => 1))
```

We initialize the *parameters* and the *state* of the model.
```@example ddpmscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

Here it is how we implement the objective ${\tilde J}_{\mathrm{P_\sigma ESM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example ddpmscorematching
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

```@example ddpmscorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example ddpmscorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example ddpmscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example ddpmscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_parzen, data, tstate_org)
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example ddpmscorematching
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

Now we train the model with the objective function ${\tilde J}_{\mathrm{P_\sigma ESM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example ddpmscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_parzen, 500, 20, 125)
nothing # hide
```

### Results

Testing out the trained model.
```@example ddpmscorematching
y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example ddpmscorematching
plot(title="Fitting", titlefont=10)

plot!(xrange, target_score', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(xx', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup ddpmscorematching
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

```@example ddpmscorematching
gif(anim, fps = 20) # hide
```

Recovering the PDF of the distribution from the trained score function.
```@example ddpmscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(xrange, target_pdf', label="original")
plot!(xrange, pdf_pred', label="recoverd")
```

And the animation of the evolution of the PDF.
```@setup ddpmscorematching
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

```@example ddpmscorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example ddpmscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", ICML'15: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, 2256-2265](https://dl.acm.org/doi/10.5555/3045118.3045358)
1. [J. Ho, A. Jain, P. Abbeel (2020), "Denoising diffusion probabilistic models", in Advances in Neural Information Processing Systems 33, NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)