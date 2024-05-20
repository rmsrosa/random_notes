# Score-based generative modeling through stochastic differential equations

```@meta
Draft = false
```

## Introduction

### Aim

Review the work of [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) that takes a complex data distribution, adds noise to it via a stochastic differential equation and generates new samples by modeling the reverse process. It is a generalization to the continuous case of the previous discrete processes of *denoising diffusion probabilistic models* and *multiple denoising score matching.*

### Background

After [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) proposed the **implicit score matching** to model a distribution by fitting its score function, several works followed it, including the **denosing score matching** of [Paul Vincent (2011)](https://doi.org/10.1162/NECO_a_00142), which perturbed the data so the analytic expression of the score function of the perturbation could be used. Then the **denoising diffusion probabilistic models,** of [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), and the **multiple denoising score matching,** of [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354), went one step further by adding several levels of noise, facilitating the generation process. The work of [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) extended that idea to the continuous case, adding noise via a stochastic differential equation.

## Forward SDE

A initial unknown probability distribution with density $p_0=p_0(x),$ associated with a random variable $X_0,$ is embedded into the distribution of an SDE of the form
```math
    \mathrm{d}X_t = f(t)X_t\;\mathrm{d}t + g(t)\;\mathrm{d}W_t,
```
with initial condition $X_0.$ The solution can be obtained with the help of the integrating factor $e^{-\int_0^t f(s)\;\mathrm{d}s}$ associated with the deterministic part of the equation. In this case,
```math
    \begin{aligned}
        \mathrm{d}\left(X_te^{-\int_0^t f(s)\;\mathrm{d}s}\right) & = \mathrm{d}X_t e^{-\int_0^t f(s)\;\mathrm{d}s} - X_t f(t) e^{\int_0^t f(s)\;\mathrm{d}s} \;\mathrm{d}t \\
        & = \left(f(t)X_t\;\mathrm{d}t + g(t)\;\mathrm{d}W_t\right)e^{-\int_0^t f(s)\;\mathrm{d}s} - X_t f(t) e^{-\int_0^t f(s)\;\mathrm{d}s} \;\mathrm{d}t \\
        & = g(t)e^{-\int_0^t f(s)\;\mathrm{d}s}\;\mathrm{d}W_t.
    \end{aligned}
```
Integrating yields
```math
    X_te^{-\int_0^t f(s)\;\mathrm{d}s} - X_0 = \int_0^t g(s)e^{-\int_0^s f(\tau)\;\mathrm{d}\tau}\;\mathrm{d}W_s.
```
Moving the exponential term to the right hand side yields the solution
```math
    X_t = X_0 e^{\int_0^t f(s)\;\mathrm{d}s} + \int_0^t e^{\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)\;\mathrm{d}W_s.
```
The mean value evolves according to
```math
    \mathbb{E}[X_t] = \mathbb{E}[X_0] e^{\int_0^t f(s)\;\mathrm{d}s}.
```
Using the Itô isometry, the second moment evolves with
```math
    \mathbb{E}[X_t^2] = \mathbb{E}[X_0^2]e^{2\int_0^t f(s)\;\mathrm{d}s} + \int_0^t e^{2\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)^2\;\mathrm{d}s.
```
Hence, the variance is given by
```math
    \operatorname{Var}(X_t) = \operatorname{Var}(X_0)e^{2\int_0^t f(s)\;\mathrm{d}s} + \int_0^t e^{2\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)^2\;\mathrm{d}s.
```

Thus, the probability density function $p(t, x)$ can be obtained by conditioning it at each initial point, with
```math
    p(t, x) = \int_{\mathbb{R}} p(t, x | 0, x_0) p_0(x_0)\;\mathrm{d}x_0,
```
and
```math
    p(t, x | 0, x_0) = \mathcal{N}(x; \mu(t)x_0, \zeta(t)^2),
```
where
```math
    \mu(t) = e^{\int_0^t f(s)\;\mathrm{d}s}
```
and
```math
    \zeta(t)^2 = \int_0^t e^{2\int_s^t f(\tau)\;\mathrm{d}\tau}g(s)^2\;\mathrm{d}s.
```

The probability density function $p(t, x)$ can also be obtained with the help of the Fokker-Planck equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(t) p(t, x)) = \frac{1}{2}\Delta_x \left( g(t)^2 p(t, x) \right),
```
whose fundamental solutions are precisely $p(t, x | 0, x_0) = \mathcal{N}(x; \mu(t)x_0, \zeta(t)^2).$

### Examples

#### Variance-exploding SDE

For example, in the variance-exploding case (VE SDE), as discussed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), as the continuous limit of the *Multiple Denoising Score Matching,* we have
```math
    f(t) = 0, \quad g(t) = \sqrt{\frac{\mathrm{d}(\sigma(t)^2)}{\mathrm{d}t}},
```
so that
```math
    \mu(t) = 1
```
and
```math
    \zeta(t)^2 = \int_0^t \frac{\mathrm{d}(\sigma(s)^2)}{\mathrm{d}s}\;\mathrm{d}s = \sigma(t)^2 - \sigma(0)^2.
```

Thus,
```math
    p(t, x | 0, x_0) = \mathcal{N}\left( x; 1, \sigma(t)^2 - \sigma(0)^2\right).
```

#### Variance-preserving SDE

In the variance-preserving case (VP SDE), as discussed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), as the continuous limit of the *Denoising Diffusion Probabilistic Model,*
```math
    f(t) = -\frac{1}{2}\beta(t), \quad g(t) = \sqrt{\beta(t)},
```
so that
```math
    \mu(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}
```
and
```math
    \zeta(t)^2 = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s = \left. -e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau} \right|_{s=0}^{s=t} = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}.
```

Thus,
```math
    p(t, x | 0, x_0) = \mathcal{N}\left( x; e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}, 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}\right).
```

#### Sub-variance-preserving SDE

In the sub-variance-preserving case (VP SDE), proposed in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) as an alternative to the previous ones,
```math
    f(t) = -\frac{1}{2}\beta(t), \quad g(t) = \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)\;\mathrm{d}s})},
```
so that
```math
    \mu(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}
```
and
```math
    \begin{align*}
        \zeta(t)^2 & = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)(1 - e^{-2\int_0^s \beta(\tau)\;\mathrm{d}\tau})\;\mathrm{d}s \\
        & = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s - \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}e^{-2\int_0^s \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s \\
        & = \int_0^t e^{-\int_s^t \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s - \int_0^t e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}e^{-\int_0^s \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s \\
        & = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} \int_0^t e^{-\int_0^s \beta(\tau)\;\mathrm{d}\tau}\beta(s)\;\mathrm{d}s \\
        & = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} + e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} \left.e^{-\int_0^s \beta(\tau)\;\mathrm{d}\tau}\right|_{s=0}^t \\
        & = 1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} + e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} \left(e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} - 1\right) \\
        & = 1 - 2e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau} + e^{-2\int_0^t \beta(\tau)\;\mathrm{d}\tau} \\
        & = \left(1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}\right)^2.
    \end{align*}
```

Thus,
```math
    p(t, x | 0, x_0) = \mathcal{N}\left( x; e^{-\frac{1}{2}\int_0^t \beta(s)\;\mathrm{d}s}, \left(1 - e^{-\int_0^t \beta(\tau)\;\mathrm{d}\tau}\right)^2\right).
```

## Loss function

The loss function for training is a continuous version of the loss for the multiple denoising score-matching. In that case, we had
```math
    J_{\textrm{SMLD}}(\boldsymbol{\theta}) = \frac{1}{2L}\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p(\mathbf{x})p_{\sigma_i}(\tilde{\mathbf{x}}|\mathbf{x})}\left[ \left\| s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma_i^2} \right\|^2 \right],
```
where $\lambda = \lambda(\sigma_i)$ is a weighting factor. When too many levels are considered, one takes a stochastic approach and approximate the loss $J_{\textrm{SMLD}}(\boldsymbol{\theta})$ by
```math
    J_{\textrm{SMLD}}^*(\boldsymbol{\theta}) = \frac{1}{2}\lambda(\sigma_i) \mathbb{E}_{p(\mathbf{x})p_{\sigma_i}(\tilde{\mathbf{x}}|\mathbf{x})}\left[ \left\| s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma_i^2} \right\|^2 \right],
```
with
```math
    \sigma_i \sim \operatorname{Uniform}[\{1, 2, \ldots, L\}].
```

The continuous version becomes
```math
    J_{\textrm{SDE}}^*(\boldsymbol{\theta}) = \frac{1}{2}\lambda(t) \mathbb{E}_{p_0(\mathbf{x}_0)p(t, \tilde{\mathbf{x}}|0, \mathbf{x}_0)}\left[ \left\| s_{\boldsymbol{\theta}}(t, \tilde{\mathbf{x}}) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} p(t, \tilde{\mathbf{x}}|0, \mathbf{x}_0) \right\|^2 \right],
```
with
```math
    t \sim \operatorname{Uniform}[0, T].
```

In practice, the empirical distribution is considered for $p_0(\mathbf{x}_0),$ and a stochastic approach is taken by sampling a single $\tilde{\mathbf{x}}_n \sim p(t_n, \tilde{\mathbf{x}}|0, \mathbf{x}_n),$ besides $t_n \sim \operatorname{Uniform}([0, T]).$ Thus, the loss takes the form
```math
    {\tilde J}_{\textrm{SDE}}^*(\boldsymbol{\theta}) = \frac{1}{2N}\sum_{n=1}^N \lambda(t_n) \left[ \left\| s_{\boldsymbol{\theta}}(t_n, \tilde{\mathbf{x}}_n) - \boldsymbol{\nabla}_{\tilde{\mathbf{x}}} p(t_n, \tilde{\mathbf{x}}_n|0, \mathbf{x}_n) \right\|^2 \right],
```
with
```math
    \mathbf{x}_n \sim p_0, \quad t_n \sim \operatorname{Uniform}[0, T], \quad \mathbf{x}_n \sim p(t_n, x | 0, \mathbf{x}_n).
```
The explicit form for the distribution $p(t_n, x | 0, \mathbf{x}_n)$ and its score $\boldsymbol{\nabla}_{\tilde{\mathbf{x}}} p(t_n, \tilde{\mathbf{x}}_n|0, \mathbf{x}_n)$ depends on the choice of the SDE.

## Numerical example

We illustrate, numerically, the use of the **score-based SDE method** to model a synthetic univariate Gaussian mixture distribution.

### Julia language setup

We use the [Julia programming language](https://julialang.org) for the numerical simulations, with suitable packages.

#### Packages

```@example sdescorematching
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

```@example sdescorematching
rng = Xoshiro(12345)
nothing # hide
```

```@setup sdescorematching
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

```@setup sdescorematching
target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
xx = permutedims(collect(xrange))
target_pdf = pdf.(target_prob, xrange')
target_score = gradlogpdf.(target_prob, xrange')

sample_points = rand(rng, target_prob, 1, 1024)
```

Visualizing the sample data drawn from the distribution and the PDF.
```@setup sdescorematching
plt = plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(plt, sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(plt, xrange, target_pdf', linewidth=4, label="pdf")
scatter!(plt, sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

```@example sdescorematching
plt # hide
```

Visualizing the score function.
```@setup sdescorematching
plt = plot(title="The score function and the sample", titlefont=10)

plot!(plt, xrange, target_score', label="score function", markersize=2)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example sdescorematching
plt # hide
```

### Parameters

Here we set some parameters for the model and prepare any necessary data.

```@example sdescorematching
trange = 0.0:0.01:1.0
```

#### Variance exploding

```@example sdescorematching
sigma_min = 0.01
sigma_max = 10.0

f_ve(t) = 0.0
g_ve(t; σₘᵢₙ = sigma_min, σₘₐₓ = sigma_max) = σₘᵢₙ * ( σₘₐₓ / σₘₐₓ )^t * √(2 * log(σₘₐₓ/σₘᵢₙ))

prob_kernel_ve(t, x0; σₘᵢₙ = sigma_min, σₘₐₓ = sigma_max) = Normal( x0, σₘᵢₙ^2 * (σₘₐₓ/σₘᵢₙ)^(2t) )
p_kernel_ve(t, x, x0) = pdf(prob_kernel_ve(t, x0), x)
score_kernel_ve(t, x, x0) = gradlogpdf(prob_kernel_ve(t, x, x0), x)
```

```@example sdescorematching
surface(trange, xrange, (t, x) -> sum(x0 -> pdf(prob_kernel_ve(t, x0), x) * pdf(target_prob, x0), xrange))
```

```@example sdescorematching
heatmap(trange, xrange, (t, x) -> sum(x0 -> pdf(prob_kernel_ve(t, x0), x) * pdf(target_prob, x0), xrange))
```

#### Variance preserving

```@example sdescorematching
beta_min = 0.1
beta_max = 20.0

f_vp(t; βₘᵢₙ=beta_min, βₘₐₓ=beta_max) = ( βₘᵢₙ + t * ( βₘₐₓ - βₘᵢₙ ) ) / 2
g_vp(t; βₘᵢₙ=beta_min, βₘₐₓ=beta_max) = √( βₘᵢₙ + t * ( βₘₐₓ - βₘᵢₙ ) )

prob_kernel_vp(t, x0; βₘᵢₙ=beta_min, βₘₐₓ=beta_max) = Normal( x0 * exp( - t^4 * ( βₘₐₓ - βₘᵢₙ ) / 4 - t * βₘᵢₙ / 2 ), 1 - exp( - t^4 * ( βₘₐₓ - βₘᵢₙ ) / 2 - t * βₘᵢₙ ))
```

```@example sdescorematching
surface(trange, xrange, (t, x) -> sum(x0 -> pdf(prob_kernel_vp(t, x0), x) * pdf(target_prob, x0), xrange))
```

```@example sdescorematching
heatmap(trange, xrange, (t, x) -> sum(x0 -> pdf(prob_kernel_vp(t, x0), x) * pdf(target_prob, x0), xrange))
```

```@example sdescorematching
L = 6
sigma_1 = 2.0
sigma_L = 0.5
theta = ( sigma_L / sigma_1 )^(1/(L-1))
sigmas = [sigma_1 * theta ^ (i-1) for i in 1:L]
```

```@example sdescorematching
data = (sample_points, sigmas)
```

### The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example sdescorematching
model = Chain(Dense(2 => 64, relu), Dense(64 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example sdescorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

### Loss function

```@example sdescorematching
function loss_function_sde(model, ps, st, data)
    sample_points, sigmas = data

    noisy_sample_points = sample_points .+ sigmas .* randn(rng, size(sample_points))
    scores = ( sample_points .- noisy_sample_points ) ./ sigmas .^ 2

    flattened_noisy_sample_points = reshape(noisy_sample_points, 1, :)
    flattened_sigmas = repeat(sigmas', 1, length(sample_points))
    model_input = [flattened_noisy_sample_points; flattened_sigmas]

    y_score_pred, st = Lux.apply(model, model_input, ps, st)
    
    flattened_scores = reshape(scores, 1, :)

    loss = mean(abs2, flattened_sigmas .* (y_score_pred .- flattened_scores)) / 2

    return loss, st, ()
end
```

### Optimization setup

#### Optimization method

We use the Adam optimiser.

```@example sdescorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

#### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library.
```@example sdescorematching
vjp_rule = Lux.Training.AutoZygote()
```

#### Processor

We use the CPU instead of the GPU.
```@example sdescorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

#### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example sdescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_sde, data, tstate_org)
```

#### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example sdescorematching
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
```@example sdescorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_sde, 1000, 20, 125)
nothing # hide
```

### Results

Checking out the trained model.
```@setup sdescorematching
plt = plot(title="Fitting", titlefont=10)
plot!(plt, xrange, target_score', linewidth=4, label="score function", legend=:topright)
scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
for sigmai in sigmas
    y_predi = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmai], tstate.parameters, tstate.states)[1]
    plot!(plt, xx', y_predi', linewidth=2, label="\$\\sigma = $(round(sigmai, digits=3))\$")
end
```

```@example sdescorematching
plt # hide
```

Visualizing the result with the smallest noise.
```@setup sdescorematching
y_pred = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmas[end]], tstate.parameters, tstate.states)[1]
plt = plot(title="Last Fitting", titlefont=10)

plot!(plt, xrange, target_score', linewidth=4, label="score function")

scatter!(plt, sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(plt, xx', y_pred', linewidth=2, label="predicted MLP")
```

```@example sdescorematching
plt # hide
```

Recovering the PDF of the distribution from the trained score function.
```@setup sdescorematching
plt = plot(title="Original PDF and PDF from predicted score functions", titlefont=10, legend=:topleft)
plot!(plt, xrange, target_pdf', label="original")
scatter!(plt, sample_points', s -> pdf(target_prob, s), label="data", markersize=2)
for sigmai in sigmas
    y_predi = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmai], tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_predi) .* dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot!(plt, xrange, pdf_pred', label="\$\\sigma = $(round(sigmai, digits=3))\$")
end
```

```@example sdescorematching
plt # hide
```

With the smallest noise.
```@setup sdescorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plt = plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(plt, xrange, target_pdf', label="original")
plot!(plt, xrange, pdf_pred', label="recoverd")
```

```@example sdescorematching
plt # hide
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup sdescorematching
ymin, ymax = extrema(target_score)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_predi = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmas[end]], tstate.parameters, tstate.states)[1]
    plot(title="Fitting evolution", titlefont=10)

    plot!(xrange, target_score', linewidth=4, label="score function")

    scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, y_predi', linewidth=2, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example sdescorematching
gif(anim, fps = 20) # hide
```

And the animation of the evolution of the PDF.
```@setup sdescorematching
ymin, ymax = extrema(target_pdf)
epsilon = (ymax - ymin) / 10
anim = @animate for (epoch, tstate) in tstates
    y_predi = Lux.apply(tstate.model, [xrange'; zero(xrange') .+ sigmas[end]], tstate.parameters, tstate.states)[1]
    paux = exp.(accumulate(+, y_predi) * dx)
    pdf_pred = paux ./ sum(paux) ./ dx
    plot(title="Fitting evolution", titlefont=10, legend=:topleft)

    plot!(xrange, target_pdf', linewidth=4, fill=true, alpha=0.3, label="PDF")

    scatter!(sample_points', s -> pdf(target_prob, s), label="data", markersize=2)

    plot!(xrange, pdf_pred', linewidth=2, fill=true, alpha=0.3, label="predicted at epoch=$(lpad(epoch, (length(string(last(tstates)[1]))), '0'))", ylims=(ymin-epsilon, ymax+epsilon))
end
```

```@example sdescorematching
gif(anim, fps = 10) # hide
```

We also visualize the evolution of the losses.
```@example sdescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false) # hide
```

## Sampling with annealed Langevin

Now we sample the modeled distribution with the annealed Langevin method described earlier.

```@setup sdescorematching
function solve_annealed_langevin_1d(rng, x0, M, tstate, sigmas, dt)
    
    L = length(sigmas)
    xt = zeros(L * M, length(x0))
    sqrt2dt = √(2dt)
    zs = zeros(length(x0))
    tt = range(0, L*M*dt, length=L*M)
    xt[1, :] .= x0
    k1 = 1
    for k in 2:L*M
        i = div(k, M+1) + 1
        sigma = sigmas[i]
        randn!(rng, zs)
        for j in axes(xt, 2)
            xt[k, j] = xt[k1, j] + first(tstate.model([xt[k1, j], sigma], tstate.parameters, tstate.states))[1] * dt + sqrt2dt * zs[j]
        end
        k1 = k
    end
    return tt, xt
end
```

```@setup sdescorematching
x0 = randn(rng, 256)
M = 50
dt = 10 / L / M
tt, xt = solve_annealed_langevin_1d(rng, x0, M, tstate, sigmas, dt)
# nothing
```

Here are the trajectories.
```@example sdescorematching
plot(title="$(length(x0)) annealed Langevin trajectories with \$M=$M\$ and \$\\mathrm{dt}=$(round(dt, sigdigits=2))\$", titlefont=10, legend=false) # hide
plot!(tt, xt, xlabel="\$t\$", ylabel="\$x\$") # hide
```

The sample histogram obtained at the end of the trajectories.

```@example sdescorematching
plot(title="Histogram at the end of sampling", titlefont=10) # hide
histogram(xt[end, :], bins=40, normalize=:pdf, label="sample") # hide
plot!(range(-6, 6, length=200), x -> pdf(target_prob, x), label="target PDF", xlabel="\$x\$") # hide
```

## References

1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", ICML'15: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, 2256-2265](https://dl.acm.org/doi/10.5555/3045118.3045358)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)
1. [J. Ho, A. Jain, P. Abbeel (2020), "Denoising diffusion probabilistic models", in Advances in Neural Information Processing Systems 33, NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)