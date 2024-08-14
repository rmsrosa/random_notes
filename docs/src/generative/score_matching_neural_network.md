# Score matching a neural network

## Introduction

### Aim

Apply the score-matching method of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) to fit a neural network model of the score function to a univariate Gaussian distribution. This borrows ideas from [Kingma and LeCun (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html), of using automatic differentiation to differentiate the neural network, and from [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), of modeling directly the score function, instead of the pdf or an energy potential for the pdf.

### Motivation

The motivation is to revisit the original idea of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) and see how it performs for training a neural network to model the score function.

### Background

The idea of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) is to directly fit the score function from the sample data, using a suitable **implicit score matching** loss function not depending on the unknown score function of the random variable. This loss function is obtained by a simple integration by parts on the **explicit score matching** objective function given by the expected square distance between the score of the model and the score of the unknown target distribution, also known as the *Fisher divergence.* The integration by parts separates the dependence on the unknown target score function from the parameters of the model, so the fitting process (minimization over the parameters of the model) does not depend on the unknown distribution.

The implicit score matching method requires, however, the derivative of the score function of the model pdf, which is costly to compute in general. In Hyvärinen's original work, all the examples considered models for which the gradient can be computed somewhat more explicitly. There was no artificial neural network involved.

In a subsequent work, [Köster and Hyvärinen (2010)](https://doi.org/10.1162/neco_a_00010) applied the method to fit the score function from a model probability with log-likelyhood obtained from a two-layer neural network, so that the gradient of the score function could still be expressed somehow explicitly.

After that, [Kingma and LeCun (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html) considered a larger artificial neural network and used automatic differentiation to optimize the model. They also proposed a penalization term in the loss function, to regularize and stabilize the optimization process, yielding a **regularized implicit score matching** method. The model in [Kingma and LeCun (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html) was not of the pdf directly, but of an energy potential, i.e. with
```math
    p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{1}{Z(\boldsymbol{\theta})} e^{-U(\mathbf{x}; \boldsymbol{\theta})},
```
where $U(\mathbf{x}; \boldsymbol{\theta})$ is modeled after a neural network.

Finally, [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) proposed modeling directly the score function as a neural network $s(\mathbf{x}; \boldsymbol{\theta})$, i.e.
```math
    \boldsymbol{\nabla}_{\mathbf{x}}p_{\boldsymbol{\theta}}(\mathbf{x}) = s(\mathbf{x}; \boldsymbol{\theta}).
```
[Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), however, went further and proposed a different method (based on several perturbations of the data, each of which akin to denoising score matching). At this point, we do not address the main method proposed in [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), we only borrow the idea of modeling directly the score function instead of the pdf or an energy potential of the pdf.

In a sense, we do an analysis in hindsight, combining ideas proposed in subsequent articles, to implement the **implicit score matching** method in a different way. In summary, we illustrate the use of automatic differentiation to allow the application of the **implicit score matching** and the **regularized implicit score matching** methods to directly fit the score function as modeled by a neural networks.

## Loss function for implicit score matching

The score-matching method of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) aims to minimize the **empirical implicit score matching** loss function ${\tilde J}_{\mathrm{ISM}{\tilde p}_0}$ given by
```math
    {\tilde J}_{\mathrm{ISM}{\tilde p}_0} = \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right),
```
where $(\mathbf{x}_n)_{n=1}^N$ is the sample data from a unknown target distribution and where $\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})$ is a parametrized model for the desired score function.

The method rests on the idea of rewriting the **explicit score matching** loss function $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ (essentially the Fisher divergence) in terms of the **implicit score matching** loss function $J_{\mathrm{ISM}}({\boldsymbol{\theta}})$, showing that 
```math
J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = J_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C,
```
and then approximating the latter by the **empirical implicit score matching** loss function ${\tilde J}_{\mathrm{ISM}{\tilde p}_0}({\boldsymbol{\theta}})$.

## Numerical example

We illustrate the method, numerically, to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

We use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages.

#### Packages

```@example adscorematching
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

```@example adscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup adscorematching
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

```@example adscorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

lambda = 0.1
sample_points = permutedims(rand(rng, target_prob, 1024))
data = (sample_points, lambda)
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup adscorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example adscorematching
plt # hide
```

Visualizing the score function.
```@setup adscorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example adscorematching
plt # hide
```

### The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example adscorematching
model = Chain(Dense(1 => 8, sigmoid), Dense(8 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example adscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

Here it is how we implement the objective ${\tilde J}_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example adscorematching
function loss_function_EISM_Zygote(model, ps, st, sample_points)
    smodel = StatefulLuxLayer{true}(model, ps, st)
    y_pred = smodel(sample_points)
    dy_pred = only(Zygote.gradient(sum ∘ smodel, sample_points))
    loss = mean(dy_pred .+ y_pred .^2 / 2)
    return loss, smodel.st, ()
end
```

We also implement a regularized version as proposed by [Kingma and LeCun (2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html).

```@example adscorematching
function loss_function_EISM_Zygote_regularized(model, ps, st, data)
    sample_points, lambda = data
    smodel = StatefulLuxLayer{true}(model, ps, st)
    y_pred = smodel(sample_points)
    dy_pred = only(Zygote.gradient(sum ∘ smodel, sample_points))
    loss = mean(dy_pred .+ y_pred .^2 / 2 .+ lambda .* dy_pred .^2 )
    return loss, smodel.st, ()
end
```

### Optimization setup

#### Optimization method

We use the Adam optimiser.

```@example adscorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example adscorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example adscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example adscorematching
@time Lux.Training.compute_gradients(vjp_rule, loss_function_EISM_Zygote, sample_points, tstate_org)
nothing # hide
```

It is pretty slow to run it the first time, since it envolves compiling a specialized method for it. Remember there is already a gradient on the loss function, so this amounts to a double automatic differentiation. The subsequent times are faster, but still slow for training:

```@example adscorematching
@time Lux.Training.compute_gradients(vjp_rule, loss_function_EISM_Zygote, sample_points, tstate_org)
nothing # hide
```

Now the version with regularization.

```@example adscorematching
@time Lux.Training.compute_gradients(vjp_rule, loss_function_EISM_Zygote_regularized, data, tstate_org)
nothing # hide
```

```@example adscorematching
@time Lux.Training.compute_gradients(vjp_rule, loss_function_EISM_Zygote_regularized, data, tstate_org)
nothing # hide
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example adscorematching
function train(tstate, vjp, data, loss_function, epochs, numshowepochs=20, numsavestates=0)
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

Now we train the model with the objective function ${\tilde J}_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example adscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, sample_points, loss_function_EISM_Zygote, 500, 20, 100)
nothing # hide
```

### Results

Testing out the trained model.
```@example adscorematching
y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example adscorematching
plot(title="Fitting", titlefont=10)

plot!(xrange, target_score', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(xx', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup adscorematching
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

```@example adscorematching
gif(anim, fps = 20) # hide
```

Recovering the PDF of the distribution from the trained score function.
```@example adscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(xrange, target_pdf', label="original")
plot!(xrange, pdf_pred', label="recoverd")
```

And the animation of the evolution of the PDF.
```@setup adscorematching
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

```@example adscorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example adscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

### Training with the regularization term

Now we train the model with the objective function ${\tilde J}_{\mathrm{ISM{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example adscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_EISM_Zygote_regularized, 500, 20, 100)
nothing # hide
```

### Results

Testing out the trained model.
```@example adscorematching
y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example adscorematching
plot(title="Fitting", titlefont=10)

plot!(xrange, target_score', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(xx', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup adscorematching
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

```@example adscorematching
gif(anim, fps = 20) # hide
```

Recovering the PDF of the distribution from the trained score function.
```@example adscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(xrange, target_pdf', label="original")
plot!(xrange, pdf_pred', label="recoverd")
```

And the animation of the evolution of the PDF.
```@setup adscorematching
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

```@example adscorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example adscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [U. Köster, A. Hyvärinen (2010), "A two-layer model of natural stimuli estimated with score matching", Neural. Comput. 22 (no. 9), 2308-33, doi: 10.1162/NECO_a_00010](https://doi.org/10.1162/neco_a_00010)
1. [Durk P. Kingma, Yann Cun (2010), "Regularized estimation of image statistics by Score Matching", Advances in Neural Information Processing Systems 23 (NIPS 2010)](https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)