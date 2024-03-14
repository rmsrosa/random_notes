# Multiple denoising score matching with annealed Langevin dynamics

```@meta
Draft = false
```

## Introduction

### Aim

Review the **multiple denoising score matching (MDSM),** or **denosing score matching with Langevin dynamics (SMLD),** which fits a **noise conditional score network (NCSN),** as introduced by [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), which together with DDPM was one step closer to the score-based SDE model.

### Background

After [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) suggested fitting the score function of a distribution, several directions were undertaken to improve the quality of the method and make it more practical.

One of the approaches was the *denoising score matching* of [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142), in which the data is corrupted by a Gaussian noise and the model was trained to correctly denoise the corrupted data. The model itself would either be of the pdf itself or of an energy potential for the pdf. In any case, one would have a model for the pdf and could draw samples directly using that.

[Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) came with two ideas tied together. The first idea was to model directly the score function and use the Langevin equation to draw samples from it. One difficulty with Langevin sampling, however, is in correctly estimating the weights of multimodal distributions, either superestimating or subestimating some modal regions, depending on where the initial distribution of points is located relative to the model regions. It may take a long time to reach the desired distribution.

In order to overcome that, [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) also proposed using an annealed version of Langevin dynamics, based on a scale of *denoising score matching* models, with different levels of noise, instead of a single denoising. Lower noises are closer to the target distribution but are challenging to the Langevin sampling, while higher noises are better for Langevin sampling but depart from the target distributions. Combining different levels of noise and gradually sampling between different denoising models improve the modeling and sampling of a distribution. That is the idea of their proposed **noise conditional score network (NCSN)** framework, in a method that was later denominated **denosing score matching with Langevin dynamics (SMLD),** and for which a more precise description would be **multiple denosing score matching with annealed Langevin dynamics,** or simply **multiple denoising score matching (MDSM).**

## Multiple denoising score matching 

The idea is to consider a sequence of denoising score matching models, starting with a relatively large noise level $\sigma_1$, to avoid the difficulties with Langevin sampling described earlier, and end up with a relatively small noise level $\sigma_L$, to minimize the noisy effect on the data.

For training, one trains directly a score model according to a weighted loss involving all noise levels.

Then, for sampling, a corresponding sequence of Langevin dynamics, with decreasing levels of noise, driving new samples closer and closer to the target distribution.

### The model

More precisely, one starts with a positive geometric sequence of noise levels $\sigma_1, \ldots, \sigma_L$ satisfying
```math
    \frac{\sigma_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1,
```
which is the same as
```math
    \sigma_i = \theta^{i-1} \sigma_1, \quad i = 1, \ldots, L,
```
for a starting $\sigma_1 > 0$ and a rate $0 < \theta < 1$ given by $\theta = \sigma_2/\sigma_1 = \ldots = \sigma_L/\sigma_{L-1}$.

For each $\sigma=\sigma_i$, $i=1, \ldots, L$, one considers the perturbed distribution
```math
    p_{\sigma}(\tilde{\mathbf{x}}) = \int_{\mathbb{R}^d} p(\mathbf{x})p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})\;\mathrm{d}\mathbf{x},
```
with a perturbation kernel
```math
    p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}\left(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I}\right).
```
This yields a sequence of perturbed distributions
```math
    \{p_{\sigma_i}\}_{i=1}^L.
```

We model the corresponding family of score functions $\{s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i)\}$, i.e. such that $s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i)$ approximates the score function of $p_{\sigma_i}$, i.e.
```math
    s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) \approx \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log p_{\sigma_i}(\tilde{\mathbf{x}}).
```

The **noise conditional score network (NCSN)** is precisely 
```math
    s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma).
```

### The loss function

One wants to train the noise conditional score network $s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)$ by weighting together the denosing loss function of each perturbation, i.e.
```math
    J_{\textrm{MDSM}}(\boldsymbol{\theta}) = \frac{1}{2L}\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p(\mathbf{x})p_{\sigma_i}(\tilde{\mathbf{x}}|\mathbf{x})}\left[ \left\| s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma_i^2} \right\|^2 \right],
```
where $\lambda = \lambda(\sigma_i)$ is a weighting factor.

In practice, we use the empirical distribution and a single corrupted data for each sample data, i.e.
```math
    {\tilde J}_{\textrm{MDSM}}(\boldsymbol{\theta}) = \frac{1}{2LN} \sum_{n=1}^N \sum_{i=1}^L \lambda(\sigma_i)\left\| s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}_{n, i}, \sigma_i) - \frac{\mathbf{x}_n - \tilde{\mathbf{x}}_{n, i}}{\sigma_i^2} \right\|^2, \quad \tilde{\mathbf{x}}_{n, i} \sim \mathcal{N}\left(\mathbf{x}_n, \sigma^2 \mathbf{I}\right).
```

This can also be written with a reparametrization,
```math
    {\tilde J}_{\textrm{MDSM}}(\boldsymbol{\theta}) = \frac{1}{2LN} \sum_{n=1}^N \sum_{i=1}^L \lambda(\sigma_i) \left\| s_{\boldsymbol{\theta}}(\mathbf{x}_n + \boldsymbol{\epsilon}_{n, i}, \sigma_i) + \frac{\boldsymbol{\epsilon}_{n, i}}{\sigma_i} \right\|^2, \quad \boldsymbol{\epsilon}_{n, i} \sim \mathcal{N}\left(\mathbf{0}_n, \mathbf{I}\right).
```

As for the choice of $\lambda(\sigma)$, [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) suggested choosing
```math
    \lambda(\sigma) = \sigma^2.
```

This comes from the observation that,
```math
    \|s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}_n, \sigma_i)\|^2 \sim \frac{1}{\sigma_i},
```
hence
```math
    \lambda(\sigma_i)\left\| s_{\boldsymbol{\theta}}(\mathbf{x}_n + \boldsymbol{\epsilon}_{n, i}, \sigma_i) + \frac{\boldsymbol{\epsilon}_{n, i}}{\sigma_i} \right\|^2 \sim 1
```
is independent of $i=1, \ldots, L$. Choosing such weighting, the loss function becomes
```math
    {\tilde J}_{\textrm{MDSM}}(\boldsymbol{\theta}) = \frac{1}{2LN} \sum_{n=1}^N \sum_{i=1}^L \left\| \sigma_i s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}_{n, i}, \sigma_i) - (\mathbf{x}_n - \tilde{\mathbf{x}}_{n, i})\right\|^2, \quad \tilde{\mathbf{x}}_{n, i} \sim \mathcal{N}\left(\mathbf{x}_n, \sigma^2 \mathbf{I}\right).
```

### Sampling

For each $i=1, \ldots, L$, the dynamics of the overdamped Langevin equation
```math
    \mathrm{d}X_t = \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log p_{\sigma_i}(\tilde{\mathbf{X}}_t)\;\mathrm{d}t + \sqrt{2}\;\mathrm{d}W_t
```
drives any initial sample towards the distribution defined by $p_{\sigma_i}$. With $s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i)$ being an approximation of $\boldsymbol{\nabla}_{\tilde{\mathbf{x}}} \log p_{\sigma_i}(\tilde{\mathbf{x}})$ and with $p_{\sigma_i}(\tilde{\mathbf{x}})$ being closer to the target $p(\mathbf{x})$ the smaller the $\sigma_i$, the idea is to run batches of Langevin dynamics for decreasing values of noise, i.e. for $\sigma_1$ down to $\sigma_L$.

More precisely, given $K\in\mathbb{N}$, we run the Langevin equation for $K$ steps, for each $i=1, \ldots, L$:

**1.** Start with a $M$ sample points $\mathbf{y}_m$, $m=1, \ldots, M$, $M\in\mathbb{N}$, of a multivariate Normal distribution, or a uniform distribution, or any other known distribution.

**2.** Then for each $i=1, \ldots, L$, run the discretized overdamped Langevin equation for $K$ steps
```math
    \mathbf{y}^i_{m, k} = \mathbf{y}^i_{m, k-1} + s_{\boldsymbol{\theta}}(\tilde{\mathbf{y}}^{i-1}_{m, k-1}, \sigma_i) \tau_i + \sqrt{2\tau_i}\mathbf{z}^i_{m, k},
```
where $\tau_i > 0$ is a given time step (which may or may not vary with $i$); the $\mathbf{z}^i_{m, k} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ are independent; and the initial conditions are given by
```math
    \mathbf{y}^1_{m, 0} = \mathbf{y}_m,
```
for $i=1$, and
```math
    \mathbf{y}^i_{m, 0} = \mathbf{y}^{i-1}_{m, K},
```
for $i = 2, \ldots, L$, i.e. the final $K$th-step of the solution of the Langevin equation with a given $i = 1, \ldots, K-1$ is the initial step of the Langevin equation for $i+1$.

**3.** The final points $\mathbf{y}^L_{m, K}$, $m=1, \ldots, M$, are the $M$ desired new generated samples of the distribution approximating the data distribution.

## Numerical example

We illustrate, numerically, the use of **multiple denoising score matching** to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

We use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages.

#### Packages

```@example multipledenoisingscorematching
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

```@example multipledenoisingscorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup multipledenoisingscorematching
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

We build the usual target model and draw samples from it.

```@setup multipledenoisingscorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sample_points = rand(rng, target_prob, 1, 1024)
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup multipledenoisingscorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example multipledenoisingscorematching
plt # hide
```

Visualizing the score function.
```@setup multipledenoisingscorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example multipledenoisingscorematching
plt # hide
```

### Parameters

Here we set some parameters for the model and prepare any necessary data. For instance, the corrupted/perturbed sample points can be computed beforehand

```@example multipledenoisingscorematching
L = 6
sigma_1 = 2.0
sigma_L = 0.5
theta = ( sigma_L / sigma_1 )^(1/(L-1))
sigmas = [sigma_1 * theta ^ (i-1) for i in 1:L]
```

```@example multipledenoisingscorematching
noisy_sample_points = sample_points .+ sigmas .* randn(rng, size(sample_points))
flattened_noisy_sample_points = reshape(noisy_sample_points, 1, :)
flattened_sigmas = repeat(sigmas', 1, length(sample_points))
model_input = [flattened_noisy_sample_points; flattened_sigmas]
scores = ( sample_points .- noisy_sample_points ) ./ sigmas .^ 2
flattened_scores = reshape(scores, 1, :)
data = (model_input, flattened_scores, flattened_sigmas)
```

### The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example multipledenoisingscorematching
model = Chain(Dense(2 => 64, relu), Dense(64 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example multipledenoisingscorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

```@example multipledenoisingscorematching
function loss_function_mdsm(model, ps, st, data)
    model_input, flattened_scores, flattened_sigmas = data
    y_score_pred, st = Lux.apply(model, model_input, ps, st)
    loss = mean(abs2, flattened_sigmas .* (y_score_pred .- flattened_scores)) / 2
    return loss, st, ()
end
```

### Optimization setup

#### Optimization method

We use the Adam optimiser.

```@example multipledenoisingscorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example multipledenoisingscorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example multipledenoisingscorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example multipledenoisingscorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_mdsm, data, tstate_org)
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example multipledenoisingscorematching
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

Now we train the model with the objective function ${\tilde J}_{\mathrm{ESM{\tilde p}_\sigma{\tilde p}_0}}({\boldsymbol{\theta}})$.
```@example multipledenoisingscorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_mdsm, 1000, 20, 125)
nothing # hide
```

### Results

Checking out the trained model.
```@setup multipledenoisingscorematching
plt = plot(title="Fitting", titlefont=10)
plot!(plt, xrange, target_score', linewidth=4, label="score function", legend=:topright)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
for sigmai in sigmas
    y_pred = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmai], tstate.parameters, tstate.states)[1]
    plot!(plt, xx', y_pred', linewidth=2, label="\$\\sigma = $(round(sigmai, digits=3))\$")
end
```

```@example multipledenoisingscorematching
plt # hide
```

Visualizing the result with the smallest noise.
```@setup multipledenoisingscorematching
y_pred = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmas[end]], tstate.parameters, tstate.states)[1]
plt = plot(title="Last Fitting", titlefont=10)

plot!(plt, xrange, target_score', linewidth=4, label="score function")

scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(plt, xx', y_pred', linewidth=2, label="predicted MLP")
```

```@example multipledenoisingscorematching
plt # hide
```

Recovering the PDF of the distribution from the trained score function.
```@setup multipledenoisingscorematching
plt = plot(title="Original PDF and PDF from predicted score functions", titlefont=10, legend=:topleft)
plot!(plt, xrange, target_pdf', label="original")
scatter!(plt, sample_points', s -> pdf(target_prob, s), label="data", markersize=2)
for sigmai in sigmas
    y_pred = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmai], tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_pred) .* dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot!(plt, xrange, pdf_pred', label="\$\\sigma = $(round(sigmai, digits=3))\$")
end
```

```@example multipledenoisingscorematching
plt # hide
```

With the smallest noise.
```@setup multipledenoisingscorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plt = plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(plt, xrange, target_pdf', label="original")
plot!(plt, xrange, pdf_pred', label="recoverd")
```

```@example multipledenoisingscorematching
plt # hide
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup multipledenoisingscorematching
ymin, ymax = extrema(target_score)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmas[end]], tstate.parameters, tstate.states)[1]
    plot(title="Fitting evolution", titlefont=10)

    plot!(xrange, target_score', linewidth=4, label="score function")

    scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, y_pred', linewidth=2, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example multipledenoisingscorematching
gif(anim, fps = 20) # hide
```

And the animation of the evolution of the PDF.
```@setup multipledenoisingscorematching
ymin, ymax = extrema(target_pdf)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_pred = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmas[end]], tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_pred) * dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot(title="Fitting evolution", titlefont=10, legend=:topleft)

    plot!(xrange, target_pdf', linewidth=4, fill=true, alpha=0.3, label="PDF")

    scatter!(sample_points', s -> pdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, pdf_pred', linewidth=2, fill=true, alpha=0.3, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example multipledenoisingscorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example multipledenoisingscorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false) # hide
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)