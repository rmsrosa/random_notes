# Score-matching a two-dimensional Gaussian mixture model

```@meta
Draft = false
```

## Introduction

### Aim

Here, we modify the previous score-matching example to a two-dimensional Gaussian mixture model. But it is not converging yet... :(

## Julia language setup

We use the [Julia programming language](https://julialang.org) with suitable packages.

### Packages

```@example 2dscorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation

nothing # hide
```

### Reproducibility

We set the random seed for reproducibility purposes.

```@example 2dscorematching
rng = Xoshiro(12345)
nothing # hide
```

### Extending the score function from Distributions.jl

This time, we extend `Distributions.gradlogpdf` to *multivariate* `MixtureModels`.

```@example 2dscorematching
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
xrange = range(-8, 8, 200)
yrange = range(-8, 8, 200)
dx = Float64(xrange.step)
dy = Float64(yrange.step)

target_prob = MixtureModel([MvNormal([-3, -3], [1 0; 0 1]), MvNormal([3, 3], [1 0; 0 1]), MvNormal([-1, 1], [1 0; 0 1])], [0.4, 0.4, 0.2])

target_pdf = [pdf(target_prob, [x, y]) for y in yrange, x in xrange]
target_score = reduce(hcat, gradlogpdf(target_prob, [x, y]) for y in yrange, x in xrange)
```

```@example 2dscorematching
sample = rand(rng, target_prob, 2048)
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
xx, yy = meshgrid(xrange[begin:10:end], yrange[begin:10:end])
uu = reduce(hcat, gradlogpdf(target_prob, [x, y]) for (x, y) in zip(xx, yy))
```

```@example 2dscorematching
heatmap(xrange, yrange, (x, y) -> logpdf(target_prob, [x, y]), title="Logpdf (heatmap) and score function (vector field)", titlefont=10, legend=false, color=:vik)
quiver!(xx, yy, quiver = (uu[1, :] ./ 8, uu[2, :] ./ 8), color=:yellow, alpha=0.5)
scatter!(sample[1, :], sample[2, :], markersize=2, markercolor=:lightgreen, alpha=0.5)
```

## The neural network model

The neural network we consider is again a simple feed-forward neural network made of a single hidden layer.

```@example 2dscorematching
model = Chain(Dense(2 => 16, relu), Dense(16 => 2))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example 2dscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

## Loss functions for score-matching

### Loss function ${\tilde J}_{\mathrm{MC, FD}}(\theta)$ with Monte-Carlo on the sample data with centered finite diferences

Here is the loss function based on [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), combined with the work of [Pang, Xu, Li, Song, Ermon, and Zhu (2020)](https://openreview.net/forum?id=LVRoKppWczk) using finite differences to approximate the divergence of the modeled score function.
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
    loss = mean(view(dsdx_pred, 1, :)) +  mean(view(dsdy_pred, 2, :))  + mean(abs2, s_pred)
    return loss, st, ()
end
```

We included the steps for the finite difference computations within the data to avoid repeated computations.
```@example 2dscorematching
xmin, xmax = extrema(sample[1, :])
ymin, ymax = extrema(sample[2, :])
deltax = (xmax - xmin) / 2size(sample, 1)
deltay = (ymax - ymin) / 2size(sample, 2)
data = sample, deltax, deltay
```

## Optimization setup

### Optimization method

The ADAM optimization.

```@example 2dscorematching
opt = Adam(0.03)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

### Automatic differentiation in the optimization

[FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) for the automatic differentiation.

```@example 2dscorematching
vjp_rule = Lux.Training.AutoZygote()
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

We use the CPU instead of the GPU.
```@example 2dscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.

```@example 2dscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function, data, tstate_org)
```

## Training

```@example 2dscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function, 20000, 20, 100)
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
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```