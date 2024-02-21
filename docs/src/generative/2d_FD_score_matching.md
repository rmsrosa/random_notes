# Finite-difference score-matching of a two-dimensional Gaussian mixture model

## Introduction

Here, we modify the previous finite-difference score-matching example to fit a two-dimensional model.

## Julia language setup

We use the [Julia programming language](https://julialang.org) with suitable packages.

```@example 2dscorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation

nothing # hide
```

We set the random seed for reproducibility purposes.

```@example 2dscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup 2dscorematching
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

## Data

We build the target model and draw samples from it. This time the target model is a bivariate random variable.

```@example 2dscorematching
xrange = range(-8, 8, 120)
yrange = range(-8, 8, 120)
dx = Float64(xrange.step)
dy = Float64(yrange.step)

target_prob = MixtureModel([MvNormal([-3, -3], [1 0; 0 1]), MvNormal([3, 3], [1 0; 0 1]), MvNormal([-1, 1], [1 0; 0 1])], [0.4, 0.4, 0.2])

target_pdf = [pdf(target_prob, [x, y]) for y in yrange, x in xrange]
target_score = reduce(hcat, gradlogpdf(target_prob, [x, y]) for y in yrange, x in xrange)
```

```@example 2dscorematching
sample = rand(rng, target_prob, 1024)
```

```@example 2dscorematching
surface(xrange, yrange, target_pdf, title="PDF", titlefont=10, legend=false, color=:vik)
scatter!(sample[1, :], sample[2, :], [pdf(target_prob, [x, y]) for (x, y) in eachcol(sample)], markercolor=:lightgreen, markersize=2, alpha=0.5)
```

```@example 2dscorematching
heatmap(xrange, yrange, target_pdf, title="PDF", titlefont=10, legend=false, color=:vik)
scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
```

```@example 2dscorematching
surface(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]), title="Logpdf", titlefont=10, legend=false, color=:vik)
scatter!(sample[1, :], sample[2, :], [logpdf(target_prob, [x, y]) for (x, y) in eachcol(sample)], markercolor=:lightgreen, alpha=0.5, markersize=2)
```

```@example 2dscorematching
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
xx, yy = meshgrid(xrange[begin:8:end], yrange[begin:8:end])
uu = reduce(hcat, gradlogpdf(target_prob, [x, y]) for (x, y) in zip(xx, yy))
```

```@example 2dscorematching
heatmap(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]), title="Logpdf (heatmap) and score function (vector field)", titlefont=10, legend=false, color=:vik)
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
```

## The neural network model

The neural network we consider is again a simple feed-forward neural network made of a single hidden layer. For the 2d case, we need to bump it a little bit, doubling the width of the hidden layer.

```@example 2dscorematching
model = Chain(Dense(2 => 16, relu), Dense(16 => 2))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example 2dscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

## Loss functions for score-matching

The loss function is again based on [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), combined with the work of [Pang, Xu, Li, Song, Ermon, and Zhu (2020)](https://openreview.net/forum?id=LVRoKppWczk) using finite differences to approximate the divergence of the modeled score function.

In the multidimensional case, say on $\mathbb{R}^d$, $d\in\mathbb{N}$, the **explicit score matching** loss function is given by
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\|^2\;\mathrm{d}\mathbf{x};
```
where $p_{\mathbf{X}}(\mathbf{x})$ is the PDF of the target distribution.

The integration by parts in the expectation yields $J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = J_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C$, where $C$ is constant with respect to the parameters and the **implicit score matching** loss function $J_{\mathrm{ISM}}({\boldsymbol{\theta}})$ is given by
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
which does not involve the unknown score function of ${\mathbf{X}}$. It does, however, involve the divergence of the modeled score function, which is expensive to compute.

In practice, the loss function is estimated via the empirical distribution, so the unknown $p_{\mathbf{X}}(\mathbf{x})$ is handled implicitly by the sample data $(\mathbf{x}_n)_n$, and we minimize the **empirical implicit score matching** loss function
```math
    {\tilde J}_{\mathrm{ISM}{\tilde p}_0} =  \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}})\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) \right).
```

Componentwise, with $\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) = (\psi_i(\mathbf{x}; {\boldsymbol{\theta}}))_{i=1}^d$, this is written as
```math
    {\tilde J}_{\mathrm{ISM}{\tilde p}_0} = \frac{1}{N}\sum_{n=1}^N \sum_{i=1}^d \left( \frac{1}{2}\psi_i(\mathbf{x}_n; {\boldsymbol{\theta}})^2 + \frac{\partial}{\partial x_i} \psi_i(\mathbf{x}_n; {\boldsymbol{\theta}}) \right).
```

As mentioned before, computing a derivative to form the loss function becomes expensive when combined with the usual optimization methods to fit a neural network, as they require the gradient of the loss function itself, so we approximate the derivative of the modeled score function by centered finite differences. With the model calculated at the displaced points, we just average them to avoid computing the model at the sample point itself. This leads to the **empirical finite-difference (implicit) score matching** loss function
```math
    {\tilde J}_{\mathrm{FDSM}{\tilde p}_0} = \frac{1}{N}\sum_{n=1}^N \sum_{i=1}^d \Bigg( \frac{1}{2}\left(\frac{1}{d}\sum_{j=1}^d \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_j; {\boldsymbol{\theta}}) + \psi_i(\mathbf{x}_n - \delta\mathbf{e}_j; {\boldsymbol{\theta}})}{2}\right)^2 \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad + \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_i; {\boldsymbol{\theta}}) - \psi_i(\mathbf{x}_n - \delta\mathbf{e}_i; {\boldsymbol{\theta}})}{2\delta} \Bigg).
```

Since this is a synthetic problem and we actually know the target distribution, we implement the **empirical explicit score matching** loss function
```math
    {\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}}) = \frac{1}{2}\frac{1}{N}\sum_{n=1}^N \|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x}_n)\|^2.
```
This is used as a sure check whether the neural network is sufficient to model the score function and for checking the optimization process, since in theory this should be roughly (apart from the approximations by the empirical distribution, the finite-difference approximation, and the round-off errors) a constant different from the loss function for ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}$.

### Implementation of ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}})$

In the two-dimensional case, $d = 2$, this becomes
```math
    \begin{align*}
        {\tilde J}_{\mathrm{FDSM}{\tilde p}_0} & = \frac{1}{N}\sum_{n=1}^N \sum_{i=1}^d \Bigg( \frac{1}{2}\left(\frac{1}{d}\sum_{j=1}^d \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_j; {\boldsymbol{\theta}}) + \psi_i(\mathbf{x}_n - \delta\mathbf{e}_j; {\boldsymbol{\theta}})}{2}\right)^2 \\
        & \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad + \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_i; {\boldsymbol{\theta}}) - \psi_i(\mathbf{x}_n - \delta\mathbf{e}_i; {\boldsymbol{\theta}})}{2\delta} \Bigg) \\
        & = \frac{1}{N}\sum_{n=1}^N \sum_{i=1}^2 \Bigg( \frac{1}{2}\left(\sum_{j=1}^2 \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_j; {\boldsymbol{\theta}}) + \psi_i(\mathbf{x}_n - \delta\mathbf{e}_j; {\boldsymbol{\theta}})}{4}\right)^2 \\
        & \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad + \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_i; {\boldsymbol{\theta}}) - \psi_i(\mathbf{x}_n - \delta\mathbf{e}_i; {\boldsymbol{\theta}})}{2\delta} \Bigg) \\
        & = \frac{1}{N} \frac{1}{2} \sum_{n=1}^N \sum_{i=1}^2 \left(\sum_{j=1}^2 \frac{\psi_i(\mathbf{x}_n + \delta\mathbf{e}_j; {\boldsymbol{\theta}}) + \psi_i(\mathbf{x}_n - \delta\mathbf{e}_j; {\boldsymbol{\theta}})}{4}\right)^2 \\
        & \qquad \qquad \qquad \qquad \qquad \qquad + \frac{1}{N}\sum_{n=1}^N \frac{\psi_1(\mathbf{x}_n + \delta\mathbf{e}_1; {\boldsymbol{\theta}}) - \psi_1(\mathbf{x}_n - \delta\mathbf{e}_1; {\boldsymbol{\theta}})}{2\delta} \\
        & \qquad \qquad \qquad \qquad \qquad \qquad + \frac{1}{N}\sum_{n=1}^N \frac{\psi_2(\mathbf{x}_n + \delta\mathbf{e}_2; {\boldsymbol{\theta}}) - \psi_2(\mathbf{x}_n - \delta\mathbf{e}_2; {\boldsymbol{\theta}})}{2\delta}
    \end{align*}
```

```@example 2dscorematching
function loss_function(model, ps, st, data)
    sample, deltax, deltay = data
    s_pred_fwd_x, = Lux.apply(model, sample .+ [deltax, 0.0], ps, st)
    s_pred_bwd_x, = Lux.apply(model, sample .- [deltax, 0.0], ps, st)
    s_pred_fwd_y, = Lux.apply(model, sample .+ [0.0, deltay], ps, st)
    s_pred_bwd_y, = Lux.apply(model, sample .- [0.0, deltay], ps, st)
    s_pred = ( s_pred_bwd_x .+ s_pred_fwd_x .+ s_pred_bwd_y .+ s_pred_fwd_y) ./ 4
    dsdx_pred = (s_pred_fwd_x .- s_pred_bwd_x ) ./ 2deltax
    dsdy_pred = (s_pred_fwd_y .- s_pred_bwd_y ) ./ 2deltay
    loss = mean(abs2, s_pred) + mean(view(dsdx_pred, 1, :)) +  mean(view(dsdy_pred, 2, :)) 
    return loss, st, ()
end
```

We included the steps for the finite difference computations in the `data` passed to training to avoid repeated computations.
```@example 2dscorematching
xmin, xmax = extrema(sample[1, :])
ymin, ymax = extrema(sample[2, :])
deltax, deltay = (xmax - xmin) / 2size(sample, 2), (ymax - ymin) / 2size(sample, 2)
```

```@example 2dscorematching
data = sample, deltax, deltay
```

### Implementation of ${\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}})$

As a sanity check, we also include the empirical explicit score matching loss function, which uses the know score functions of the target model.

In the two-dimensional case, this is simply the mean square value of all the components.
```math
    {\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}}) = \frac{1}{2} \frac{1}{N}\sum_{n=1}^N \|\boldsymbol{\psi}(\mathbf{x}_n; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x}_n)\|^2 = \frac{1}{2} \frac{1}{N}\sum_{n=1}^N \sum_{i=1}^2 \left(\psi_i(\mathbf{x}_n; {\boldsymbol{\theta}}) - \psi_{\mathbf{X}, i}(\mathbf{x}_n) \right)^2.
```

```@example 2dscorematching
function loss_function_cheat(model, ps, st, data)
    sample, score_cheat = data
    score_pred, st = Lux.apply(model, sample, ps, st)
    loss = mean(abs2, score_pred .- score_cheat)
    return loss, st, ()
end
```

The data in this case includes information about the target distribution.
```@example 2dscorematching
score_cheat = reduce(hcat, gradlogpdf(target_prob, u) for u in eachcol(sample))
data_cheat = sample, score_cheat
```

### Computing the constant

The expression ${\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}}) \approx {\tilde J}_{\mathrm{ISM}{\tilde p}_0}({\boldsymbol{\theta}}) + C$ can be used to test the implementation of the different loss functions. For that, we need to compute the constant $C$. This can be computed with a fine mesh or with a Monte-Carlo approximation. We do both just for fun.

```@example 2dscorematching
function compute_constante(target_prob, xrange, yrange)
    dx = Float64(xrange.step)
    dy = Float64(yrange.step)
    Jconstant = sum(pdf(target_prob, [x, y]) * sum(abs2, gradlogpdf(target_prob, [x, y])) for y in yrange, x in xrange) * dx * dy / 2
    return Jconstant
end    
```

```@example 2dscorematching
function compute_constante_MC(target_prob, sample)
    Jconstant = mean(sum(abs2, gradlogpdf(target_prob, s)) for s in eachcol(sample)) / 2
    return Jconstant
end    
```

```@example 2dscorematching
Jconstant = compute_constante(target_prob, xrange, yrange)
```

```@example 2dscorematching
Jconstant_MC = compute_constante_MC(target_prob, sample)
```

```@example 2dscorematching
constants = [(n, compute_constante_MC(target_prob, rand(rng, target_prob, n))) for _ in 1:100 for n in (1, 10, 20, 50, 100, 500, 1000, 2000, 4000)]
```

```@example 2dscorematching
scatter(constants, markersize=2, title="constant computed by MC and fine mesh", titlefont=10, xlabel="sample size", ylabel="value", label="via various samples")
hline!([Jconstant], label="via fine mesh")
hline!([Jconstant_MC], label="via working sample", linestyle=:dash)
```

### A test for the implementations of the loss functions

Notice that, for a sufficiently large sample and sufficiently small discretization step $\delta$, we should have
```math
{\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}}) \approx J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = J_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C \approx {\tilde J}_{\mathrm{FDSM}}({\boldsymbol{\theta}}) + C \approx {\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}}) + C.
```
which is a good test for the implementations of the loss functions. For example:

```@example 2dscorematching
first(loss_function_cheat(model, ps, st, data_cheat))
```

```@example 2dscorematching
first(loss_function(model, ps, st, data)) + Jconstant
```

Let us do a more statistically significant test.
```@example 2dscorematching
test_losses = reduce(
    hcat,
    Lux.setup(rng, model) |> pstj -> 
    [
        first(loss_function_cheat(model, pstj[1], pstj[2], data_cheat)),
        first(loss_function(model, pstj[1], pstj[2], data))
    ]
    for _ in 1:30
)
```

```@example 2dscorematching
plot(title="Loss functions at random model parameters", titlefont=10)
scatter!(test_losses[1, :], label="{\\tilde J}_{\\mathrm{ESM}{\\tilde p}_0}")
scatter!(test_losses[2, :], label="\$\\tilde {\\tilde J}_{\\mathrm{FDSM}{\\tilde p}_0}\$")
scatter!(test_losses[2, :] .+ Jconstant, label="\${\\tilde J}_{\\mathrm{FDSM}{\\tilde p}_0} + C\$")
```

One can check by visual inspection that the agreement between ${\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}}) - C$ and ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}})$ seems reasonably good. Let us estimate the relative error.

```@example 2dscorematching
rel_errors = abs.( ( test_losses[2, :] .+ Jconstant .- test_losses[1, :] ) ./ test_losses[1, :] )
plot(title="Relative error at random model parameters", titlefont=10, legend=false)
scatter!(rel_errors, markercolor=2, label="error")
mm = mean(rel_errors)
mmstd = std(rel_errors)
hline!([mm], label="mean")
hspan!([mm+mmstd, mm-mmstd], fillbetween=true, alpha=0.3, label="65% margin")
```

Ok, good enough, just a few percentage points.

### An extra test for the implementations of the loss functions and the gradient computation

We also have
```math
\boldsymbol{\nabla}_{\boldsymbol{\theta}} {\tilde J}_{\mathrm{ESM}{\tilde p}_0}({\boldsymbol{\theta}}) \approx \boldsymbol{\nabla}_{\boldsymbol{\theta}} {\tilde J}_{\mathrm{FDSM}{\tilde p}_0}({\boldsymbol{\theta}}),
```
which is another good test, which also checks the gradient computation, but everything seems fine, so no need to push this further.

## Optimization setup

### Optimization method

As usual, we use the ADAM optimization.

```@example 2dscorematching
opt = Adam(0.003)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

### Automatic differentiation in the optimization

[FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) is used for the automatic differentiation as it is currently the only AD backend working with [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl).

```@example 2dscorematching
vjp_rule = Lux.Training.AutoZygote()
```

### Processor

We use the CPU instead of the GPU.
```@example 2dscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

### Check differentiation

Check if AD is working fine to differentiate the loss functions for training.

```@example 2dscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function, data, tstate_org)
```

```@example 2dscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_cheat, data_cheat, tstate_org)
```

### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration and the model state for animation.
```@example 2dscorematching
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

## Cheat training with ${\tilde J}_{\mathrm{ESM}{\tilde p}_0}$

We first train the model with the known score function on the sample data. That is cheating. The aim is a sanity check, to make sure the proposed model is good enough to fit the desired score function and that the setup is right.

```@example 2dscorematching
@time tstate_cheat, losses_cheat, tstates_cheat = train(tstate_org, vjp_rule, data_cheat, loss_function_cheat, 2000, 20, 100)
nothing # hide
```

Testing out the trained model.
```@example 2dscorematching
uu_cheat = Lux.apply(tstate_cheat.model, vcat(xx', yy'), tstate_cheat.parameters, tstate_cheat.states)[1]
```

```@example 2dscorematching
heatmap(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]), title="Logpdf (heatmap) and score functions (vector fields)", titlefont=10, color=:vik, xlims=extrema(xrange), ylims=extrema(yrange), legend=false)
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
quiver!(xx, yy, quiver = (uu_cheat[1, :] ./ 8, uu_cheat[2, :] ./ 8), color=:cyan, alpha=0.5)
```

```@setup 2dscorematching
anim = @animate for (epoch, tstate) in tstates_cheat
    uu_pred = Lux.apply(tstate.model, vcat(xx', yy'), tstate.parameters, tstate.states)[1]

    heatmap(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]),color=:vik)
    quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
    scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
    quiver!(xx, yy, quiver = (uu_pred[1, :] ./ 8, uu_pred[2, :] ./ 8), color=:cyan, alpha=0.5)

    plot!(title="Fitting evolution (epoch=$(lpad(epoch, (length(string(last(tstates_cheat)[1]))), '0')))", titlefont=10, xlims=extrema(xrange), ylims=extrema(yrange), legend=false)
end
```

```@example 2dscorematching
gif(anim, fps = 10) # hide
```

```@example 2dscorematching
plot(losses_cheat, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## Real training with ${\tilde J}_{\mathrm{FDSM}{\tilde p}_0}$

Now we go to the real thing.

```@example 2dscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function, 2000, 20, 100)
nothing # hide
```

Testing out the trained model.
```@example 2dscorematching
uu_pred = Lux.apply(tstate.model, vcat(xx', yy'), tstate.parameters, tstate.states)[1]
```

```@example 2dscorematching
heatmap(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]), title="Logpdf (heatmap) and score functions (vector fields)", titlefont=10, color=:vik, xlims=extrema(xrange), ylims=extrema(yrange), legend=false)
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
quiver!(xx, yy, quiver = (uu_pred[1, :] ./ 8, uu_pred[2, :] ./ 8), color=:cyan, alpha=0.5)
```

```@setup 2dscorematching
anim = @animate for (epoch, tstate) in tstates
    uu_pred = Lux.apply(tstate.model, vcat(xx', yy'), tstate.parameters, tstate.states)[1]

    heatmap(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]),color=:vik)
    quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
    scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
    quiver!(xx, yy, quiver = (uu_pred[1, :] ./ 8, uu_pred[2, :] ./ 8), color=:cyan, alpha=0.5)

    plot!(title="Fitting evolution (epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0')))", titlefont=10, xlims=extrema(xrange), ylims=extrema(yrange), legend=false)
end
```

```@example 2dscorematching
gif(anim, fps = 10) # hide
```

```@example 2dscorematching
plot(losses, title="Evolution of the losses", titlefont=10, xlabel="iteration", ylabel="error", label="\${\\tilde J}_{\\mathrm{FDSM}{\\tilde p}_0}\$")
plot!(losses_cheat, linestyle=:dash, label="{\\tilde J}_{\\mathrm{ESM}{\\tilde p}_0}")
plot!(losses .+ Jconstant, linestyle=:dash, color=1, label="\${\\tilde J}_{\\mathrm{FDSM}{\\tilde p}_0} + C\$")
```

Ok, that seems visually good enough. We will later check the sampling from this score function via Langevin sampling.
