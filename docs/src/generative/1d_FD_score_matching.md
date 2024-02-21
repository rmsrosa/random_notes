# Finite-difference score-matching of a one-dimensional Gaussian mixture model

## Introduction

### Aim

The aim, this time, is to fit a neural network via **finite-difference score matching**, following the pioneering work of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) about score-matching, combined with the work of [Pang, Xu, Li, Song, Ermon, and Zhu (2020)](https://openreview.net/forum?id=LVRoKppWczk), which uses finite differences to efficiently approximate the gradient in the loss function proposed by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html).

### Background

Generative score-matching diffusion methods use Langevin dynamics to draw samples from a modeled score function. It rests on the idea of [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) that one can directly model the score function, from the sample data, using a suitable loss function (associated with the Fisher divergence) not depending on the unknown score function of the random variable. This is obtained by a simple integration by parts on the expected square distance between the model score function and the actual score function. The integration by parts separates the dependence on the actual score function from the parameters of the model, so the fitting process (minimization over the parameters of the model) does not depend on the unknown score function.

The obtained loss function, however, depends on the gradient of the model, which is computationally expensive. [Pang, Xu, Li, Song, Ermon, and Zhu (2020)](https://openreview.net/forum?id=LVRoKppWczk) proposed to use finite differences to approximate the derivative of the model to significantly reduce the computational cost of training the model.

The differentiation for the optimization is with respect to the parameters, while the differentiation of the modeled score function is on the variate, but still this is a great computational challenge and not all AD are fit for that. For this reason, we resort to centered finite differences to approximate the derivative of the modeled score function.

For a python version of a similar pedagogical example, see [Eric J. Ma (2021)](https://ericmjl.github.io/score-models/). There, they use AD on top of AD, via the [google/jax](https://github.com/google/jax) library, which apparently handles this double-AD not so badly.

### Take away

We'll see that, in this simple example at least, we don't need a large or deep neural network. It is much more important to have enough sample points to capture the transition region in the mixture of Gaussians.

## The finite-difference implicit score matching method

The score-matching method from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) rests on the following ideas:

**1.** Fit the model by minimizing the expected square distance between the model score function $\psi(x; {\boldsymbol{\theta}})$ and the score function $\psi_X(x)$ of the random variable $X$, via the **explicit score matching (ESM)** objective
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x}.
```

**2.** Use integration by parts in the expectation to write that
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = J_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C,
```
where $C$ is constant with respect to the parameters, so we only need to minimize the **implicit score matching (ISM)** objective ${\tilde J}_{\mathrm{ISM}}$, given by
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
which does not involve the unknown score function of ${\mathbf{X}}$. It does, however, involve the gradient of the modeled score function, which is expensive to compute.

**3.** In practice, the implicit score-matching loss function, which depends on the unknown $p_\mathbf{X}(\mathbf{x})$, is estimated via the empirical distribution, obtained from the sample data $(\mathbf{x}_n)_n$. Thus, we minimize the **empirical implicit score matching** objective
```math
    {\tilde J}_{\mathrm{ISM}{\tilde p}_0} =  \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right).
```
where the *empirical distribution* is given by ${\tilde p}_0 = (1/N)\sum_{n=1}^N \delta_{\mathbf{x}_n}.$

On top of that, we add one more step.

**4.** As mentioned before, computing a derivative to form the loss function becomes expensive when combined with the usual optimization methods to fit a neural network, as they require the gradient of the loss function itself, i.e. the optimization process involves the gradient of the gradient of something. Because of that, one alternative is to approximate the derivative of the model score function by centered finite differences, i.e.
```math
    \frac{\partial}{\partial x} \psi(x_n; {\boldsymbol{\theta}}) \approx \frac{\psi(x_n + \delta; {\boldsymbol{\theta}}) - \psi(x_n - \delta; {\boldsymbol{\theta}})}{2\delta},
```
for a suitably small $\delta > 0$.

In this case, since we need compute $\psi(x_n + \delta; {\boldsymbol{\theta}})$ and $\psi(x_n - \delta; {\boldsymbol{\theta}})$, we avoid computing also $\psi(x_n; {\boldsymbol{\theta}})$ and approximate it with the average
```math
    \psi(x_n; {\boldsymbol{\theta}}) \approx \frac{\psi(x_n + \delta; {\boldsymbol{\theta}}) + \psi(x_n - \delta; {\boldsymbol{\theta}})}{2}.
```

Hence, we approximate the implicit score matching ${\tilde J}_{\mathrm{ISM}{\tilde p}_0}$ by the **finite-difference (implicit) score matching**
```math
    {\tilde J}_{\mathrm{FDSM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_X(x) \Bigg( \frac{1}{2}\left(\frac{\psi(x + \delta; {\boldsymbol{\theta}}) + \psi(x - \delta; {\boldsymbol{\theta}})}{2}\right)^2 + \frac{\psi(x + \delta; {\boldsymbol{\theta}}) - \psi(x - \delta; {\boldsymbol{\theta}})}{2\delta} \Bigg)\;\mathrm{d}x,
```

And the empirical implicit score matching ${\tilde J}_{\mathrm{ISM}{\tilde p}_0}$ is approximated by
```math
    {\tilde J}_{\mathrm{FDSM}{\tilde p}_0} =  \frac{1}{N}\sum_{n=1}^N \Bigg( \frac{1}{2}\left(\frac{\psi(x + \delta; {\boldsymbol{\theta}}) + \psi(x - \delta; {\boldsymbol{\theta}})}{2}\right)^2 + \frac{\psi(x + \delta; {\boldsymbol{\theta}}) - \psi(x - \delta; {\boldsymbol{\theta}})}{2\delta} \Bigg).
```

## Numerical example

We illustrate the above method by fitting a neural network to a univariate Gaussian mixture distribution.

We played with different target distributions and settled here with a bimodal distribution used in [Eric J. Ma (2021)](https://ericmjl.github.io/score-models/).

### Julia language setup

We use the [Julia programming language](https://julialang.org) with suitable packages.

#### Packages

```@example onedimscorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation

nothing # hide
```

#### Reproducibility

We set the random seed for reproducibility purposes.

```@example onedimscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup onedimscorematching
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

## Code introspection

We do not attempt to overly optimize the code here since this is a simple one-dimensional problem. Nevertheless, it is always healthy to check the type stability of the critical parts (like the loss functions) with `@code_warntype`. One should also check for any unusual time and allocation with `BenchmarkTools.@btime` or `BenchmarkTools.@benchmark`. We performed these analysis and everything seems good. We found it unnecessary to clutter the notebook with their outputs here, though.

## Data

We build the target model and draw samples from it. We need enough sample points to capture the transition region in the mixture of Gaussians.

```@example onedimscorematching
xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
x = permutedims(collect(xrange))

target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

target_pdf = pdf.(target_prob, x)
target_score = gradlogpdf.(target_prob, x)

y = target_score # just to simplify the notation
sample_points = permutedims(rand(rng, target_prob, 1024))
data = (x, y, target_pdf, sample_points)
```

Notice the data `x` and `sample_points` are defined as row vectors so we can apply the model in batch to all of their values at once. The values `y` are also row vectors for easy comparison with the predicted values. When, plotting, though, we need to revert them to vectors.

Visualizing the sample data drawn from the distribution and the PDF.
```@example onedimscorematching
plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(x', target_pdf', linewidth=4, label="pdf")
scatter!(sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

Visualizing the score function.
```@example onedimscorematching
plot(title="The score function and the sample", titlefont=10)

plot!(x', target_score', label="score function", markersize=2)
scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

## The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see, again, that we don't need a big neural network in this simple example. We go as low as it works.

```@example onedimscorematching
model = Chain(Dense(1 => 8, relu), Dense(8 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example onedimscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Explicit score matching loss function $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$

For educational purposes, since we have the pdf and the score function, one of the ways we may train the model is directly with $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$. This is also useful to make sure that our network is able to model the desired score function.

Here is how we implement it.
```@example onedimscorematching
function loss_function_esm(model, ps, st, data)
    x, y, target_pdf, sample_points = data
    y_pred, st = Lux.apply(model, x, ps, st)
    loss = mean(target_pdf .* (y_pred .- y) .^2)
    return loss, st, ()
end
```

### Plain square error loss function

Still for educational purposes, we modify $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ for training, without weighting on the distribution of the random variable itself, as in $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$. This has the benefit of giving more weight to the transition region. Here is how we implement it.
```@example onedimscorematching
function loss_function_esm_plain(model, ps, st, data)
    x, y, target_pdf, sample_points = data
    y_pred, st = Lux.apply(model, x, ps, st)
    loss = mean(abs2, y_pred .- y)
    return loss, st, ()
end
```

### Finite-difference score matching ${\tilde J}_{\mathrm{FDSM}}$

Again, for educational purposes, we may implement ${\tilde J}_{\mathrm{FDSM}}({\boldsymbol{\theta}})$, as follows.

```@example onedimscorematching
function loss_function_FDSM(model, ps, st, data)
    x, y, target_pdf, sample_points = data
    xmin, xmax = extrema(x)
    delta = (xmax - xmin) / 2length(x)
    y_pred_fwd, = Lux.apply(model, x .+ delta, ps, st)
    y_pred_bwd, = Lux.apply(model, x .- delta, ps, st)
    y_pred = ( y_pred_bwd .+ y_pred_fwd ) ./ 2
    dy_pred = (y_pred_fwd .- y_pred_bwd ) ./ 2delta
    loss = mean(target_pdf .* (dy_pred + y_pred .^ 2 / 2))
    return loss, st, ()
end
```

### Empirical finite-difference score matching loss function ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}$

In practice we would use the sample data, not the supposedly unknown score function and PDF themselves. Here would be one implementation using finite differences, along with Monte-Carlo.
```@example onedimscorematching
function loss_function_FDSM_over_sample(model, ps, st, data)
    x, y, target_pdf, sample_points = data
    xmin, xmax = extrema(sample_points)
    delta = (xmax - xmin) / 2length(sample_points)
    y_pred_fwd, = Lux.apply(model, sample_points .+ delta, ps, st)
    y_pred_bwd, = Lux.apply(model, sample_points .- delta, ps, st)
    y_pred = ( y_pred_bwd .+ y_pred_fwd ) ./ 2
    dy_pred = (y_pred_fwd .- y_pred_bwd ) ./ 2delta
    loss = mean(dy_pred + y_pred .^ 2 / 2)
    return loss, st, ()
end
```

### Empirical implicit score matching loss function $J_{\mathrm{ISM}{\tilde p}_0}({\boldsymbol{\theta}})$

We can implement the actual implicit loss function with the derivative of the model score function using some automatic differentiation tool, as follows, but we do not optimize with it here. We do this in a separate note, not to render this note too slowly.
```@example onedimscorematching
function loss_function_EISM_Zygote(model, ps, st, data)
    x, y, target_pdf, sample_points = data
    y_pred, st = Lux.apply(model, sample_points, ps, st)
    dy_pred = Zygote.gradient(s -> sum(model(s, ps, st)[1]), sample_points)[1]
    loss = mean(dy_pred .+ y_pred .^2 / 2)
    return loss, st, ()
end
```

## Optimization setup

### Optimization method

We use the classical Adam optimiser (see [Kingma and Ba (2015)](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8)), which is a stochastic gradient-based optimization method.

```@example onedimscorematching
opt = Adam(0.03)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library, which is currently the only one implemented (there are pre-defined methods for `AutoForwardDiff()`, `AutoReverseDiff()`, `AutoFiniteDifferences()`, etc., but not implemented yet).
```@example onedimscorematching
vjp_rule = Lux.Training.AutoZygote()
```

### Processor

We use the CPU instead of the GPU.
```@example onedimscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example onedimscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_esm, data, tstate_org)
```

```@example onedimscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_esm_plain, data, tstate_org)
```

```@example onedimscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_FDSM, data, tstate_org)
```

```@example onedimscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_FDSM_over_sample, data, tstate_org)
```

### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example onedimscorematching
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

## Training

### Training with $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$

Now we attempt to train the model, starting with $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$.
```@example onedimscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_esm, 500, 20, 125)
nothing # hide
```

Testing out the trained model.
```@example onedimscorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example onedimscorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup onedimscorematching
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

```@example onedimscorematching
gif(anim, fps = 20) # hide
```

We also visualize the evolution of the losses.
```@example onedimscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example onedimscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

### Training with plain square error loss

Now we attempt to train it with the plain square error loss function. We do not reuse the state from the previous optimization. We start over at the initial state, for the sake of comparison of the different loss functions.
```@example onedimscorematching
@time tstate, losses, = train(tstate_org, vjp_rule, data, loss_function_esm_plain, 500)
nothing # hide
```

Testing out the trained model.
```@example onedimscorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example onedimscorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

And evolution of the losses.
```@example onedimscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example onedimscorematching
paux = exp.(accumulate(+, y_pred) * dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

That is an almost perfect matching.

### Training with ${\tilde J}_{\mathrm{FDSM}}({\boldsymbol{\theta}})$

Now we attempt to train it with ${\tilde J}_{\mathrm{FDSM}}$. Again we start over with the untrained state of the model.
```@example onedimscorematching
@time tstate, losses, = train(tstate_org, vjp_rule, data, loss_function_FDSM, 500)
nothing # hide
```

We may try a little longer from this state on.
```@example onedimscorematching
@time tstate, losses_more, = train(tstate, vjp_rule, data, loss_function_FDSM, 500)
append!(losses, losses_more)
nothing # hide
```

Testing out the trained model.
```@example onedimscorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example onedimscorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

And evolution of the losses.
```@example onedimscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example onedimscorematching
paux = exp.(accumulate(+, y_pred) * dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

### Training with ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}})$

Finally we attemp to train with the sample data. This is the real thing, without anything from the supposedly unknown target distribution other than the sample data.
```@example onedimscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_FDSM_over_sample, 500, 20, 125)
nothing # hide
```

Testing out the trained model.
```@example onedimscorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example onedimscorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Let us see an animation of the optimization process in this case, as well, since it is the one of interest.
```@setup onedimscorematching
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

```@example onedimscorematching
gif(anim, fps = 20) # hide
```

Here is the evolution of the losses.
```@example onedimscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example onedimscorematching
paux = exp.(accumulate(+, y_pred) * dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

And the evolution of the PDF.
```@setup onedimscorematching
ymin, ymax = extrema(target_pdf)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_pred) * dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot(title="Fitting evolution", titlefont=10, legend=:topleft)

    plot!(x', target_pdf', linewidth=4, fill=true, alpha=0.3, label="PDF")

    scatter!(sample_points', s -> pdf(target_prob, s), label="data", markersize=2)

    plot!(x', pdf_pred', linewidth=2, fill=true, alpha=0.3, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example onedimscorematching
gif(anim, fps = 10) # hide
```

### Pre-training ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}{\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}})$ with $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$

Let us now pre-train the model with the $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ and see if ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}})$ improves.

```@example onedimscorematching
tstate, = train(tstate_org, vjp_rule, data, loss_function_esm, 500)
nothing # hide
```

```@example onedimscorematching
tstate, losses, = train(tstate, vjp_rule, data, loss_function_FDSM_over_sample, 500)
nothing # hide
```

Testing out the trained model.
```@example onedimscorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example onedimscorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Recovering the PDF of the distribution from the trained score function.
```@example onedimscorematching
paux = exp.(accumulate(+, y_pred) * dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

And evolution of the losses.
```@example onedimscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## The need for enough sample points

One interesting thing is that enough sample points in the low-probability transition region is required for a proper fit, as the following example with few samples illustrates.

```@example onedimscorematching
y = target_score # just to simplify the notation
sample_points = permutedims(rand(rng, target_prob, 128))
data = (x, y, target_pdf, sample_points)
```

```@example onedimscorematching
tstate, losses, = train(tstate_org, vjp_rule, data, loss_function_FDSM_over_sample, 500)
nothing # hide
```

Testing out the trained model.
```@example onedimscorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example onedimscorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s)', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Recovering the PDF of the distribution from the trained score function.
```@example onedimscorematching
paux = exp.(accumulate(+, y_pred) * dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

And evolution of the losses.
```@example onedimscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [T. Pang, K. Xu, C. Li, Y. Song, S. Ermon, J. Zhu (2020), Efficient Learning of Generative Models via Finite-Difference Score Matching, NeurIPS](https://openreview.net/forum?id=LVRoKppWczk) - see also the [arxiv version](https://arxiv.org/abs/2007.03317)
1. [Eric J. Ma, A Pedagogical Introduction to Score Models, webpage, April 21, 2021](https://ericmjl.github.io/score-models/) - with the associated [github repo](https://github.com/ericmjl/score-models/blob/main/score_models/losses/diffusion.py#L7)
1. [D. P. Kingma, J. Ba (2015), Adam: A Method for Stochastic Optimization, In International Conference on Learning Representations (ICLR)](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8) -- see also the [arxiv version](https://arxiv.org/abs/1412.6980)
