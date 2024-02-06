# Score-matching a multi-layer perceptron to a one-dimensional Gaussian mixture model

```@meta
Draft = false
```

## Introduction

### Aim

Here, the aim is to fit a neural network (more specifically a multi-layer perceptron - MLP) to model the score function of a one-dimensional synthetic Gaussian-mixture distribution, following the pioneering work of [Aapo Hyvärinen, "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6 (2005), 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html) about score-matching, combined with the work of [T. Pang, K. Xu, C. Li, Y. Song, S. Ermon, J. Zhu, Efficient Learning of Generative Models via Finite-Difference Score Matching, NeurIPS 2020](https://openreview.net/forum?id=LVRoKppWczk), which uses finite differences to efficiently approximate the gradient in the loss function proposed by Aapo Hyvärinen.

### Motivation

The motivation is to revisit the original idea of Aapo Hyvärinen (2005), as a first step towards building a solid background on score-matching diffusion.

Generative score-matching diffusion methods use Langevin dynamics to draw samples from a modeled score function. It rests on the original idea of Aapo Hyvärinen (2005) that one can directly model the score function, from the sample data, using a suitable loss function not depending on the unknown score function of the random variable. This is obtained by a simple integration by parts on the MSE loss function between the modeled score function and the actual score function. The integration by parts separates the dependence on the actual score function from the parameters of the model, so the fitting process (minimization over the parameters of the model) does not depend on the unknown score function.

The obtained loss function, however, depends on the gradient of the model, which is computationally expensive. Xu, Li, Song, Ermon & Zhu (2000) proposed to use finite differences to approximate the derivative of the model to significantly reduce the computational cost of training the model. We also use this idea here.

It is worth noticing, in light of our project on score-matching diffusion, that the original work of Aapo Hyvärinen has no diffusion. It is simply a direct modeling of the score function in the original probability space. But the same idea of integration by parts is carried over to the loss function in the case of diffusion score-matching.

We also mention that the work of Aapo Hyvärinen uses the modified loss function to fit some very specific predefined models. There are three examples. In these examples, the derivative of the model could be computed more explicitly. There was no artificial neural network involved and no need for automatic differention (AD). Here, however, we want something more general, as done in the generative methods, and attempt to fit a neural network instead. This presents practical difficulties for the minimization process, since we end up needing automatic differentiation (for the optimization) on top of automatic differentiation (of the score function).

The differentiation for the optimization is with respect to the parameters, while the differentiation of the modeled score function is on the variate, but still this is a great computational challenge and not all AD are fit for that. For this reason, we resort to centered finite differences to approximate the derivative of the modeled score function. We will use automatic differentiation of the modeled score function in a separate note, for illustrative purposes.

This is a one-dimensional example. For higher-dimensional examples, the *sliced-score matching* approach of [Y. Song, S. Garg, J. Shi, S. Ermon, Sliced Score Matching: A Scalable Approach to Density and Score Estimation, Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, PMLR 115:574-584, 2020](https://proceedings.mlr.press/v115/song20a.html) is quite useful (see also the blog discussion [Y. Song, Sliced Score Matching: A Scalable Approach to Density and Score Estimation](http://yang-song.net/blog/2019/ssm/))

For a python version of a similar pedagogical example, see [Eric J. Ma, A Pedagogical Introduction to Score Models, April 21, 2021](https://ericmjl.github.io/score-models/). The details of their implementation can be found in the github repository [Implemented python score_matching loss by Eric Ma](https://github.com/ericmjl/score-models/blob/main/score_models/losses/diffusion.py#L7). There, they use AD on top of AD, via the [google/jax](https://github.com/google/jax) library, which apparently handles this double-AD smoothly.

### Take away

We'll see that, in this simple example at least, we don't need a large or deep neural network. It is much more important to have enough sample points to capture the transition region in the mixture of Gaussians.

## Julia language setup

We use the [Julia programming language](https://julialang.org) with suitable packages.

### Packages

```@example simplescorematching
using StatsPlots
using Random
using Distributions
using Lux # artificial neural networks explicitly parametrized
using Optimisers
using Zygote # automatic differentiation

nothing
```

There are several Julia libraries for artificial neural networks and for automatic differentiation (AD). The most established package for artificial neural networks is the [FluxML/Flux.jl](https://github.com/FluxML/Flux.jl) library, which handles the parameters implicitly. A newer library that handles the parameters explicitly is the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library, which is also taylored to the differential equations [SciML](https://sciml.ai) ecosystem.

There are heated discussions, in the community, about the pros and cons of both implicit and explicit parametrized models, but we do not go into these here. Since we aim to combine score-matching with neural networks and, eventually, with stochastic differential equations, we thought it was a reasonable idea to experiment with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library, regardless of it being explicit or not.

As we mentioned, the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) library is a newer package and not as well developed. In particular, it seems the only AD that works with it is the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library. Unfortunately for our use case, the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library is not fit to do AD on top of AD, as one can see from e.g. [Zygote: Design limitations](https://fluxml.ai/Zygote.jl/dev/limitations/#Second-derivatives-1). Nevertheless, since at the end it is more efficient to approximate the derivative of the modeled score function by finite differences, we do not let that stop us short from using the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl)/[FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) combination.

### Reproducibility

We set the random seed for reproducibility purposes.

```@example simplescorematching
rng = Xoshiro(12345)
nothing
```

### Extending the score function from Distributions.jl

The distributions and their PDF are obtained from the [JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package. The score function is also implemented in [JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl) as `gradlogpdf`, but only for some distributions. Since we are interested on Gaussian mixtures, we do some *pirating* and extend `Distributions.gradlogpdf` to `MixtureModels` as follows.

```@example simplescorematching
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

As an illustration:
```@example simplescorematching
x_interval = range(0.0, 20.0, 200)
prob = MixtureModel([Normal(3, 1), Normal(8, 2)], [0.3, 0.7])
plot(x_interval, s -> pdf(prob, s), label=false, title="pdf", titlefont=10)
```

```@example simplescorematching
plot(x_interval, s -> logpdf(prob, s), label=false, title="logpdf", titlefont=10)
```

```@example simplescorematching
plot(x_interval, s -> gradlogpdf(prob, s), label=false, title="gradlogpdf", titlefont=10)
```

## Code introspection

We do not attempt to overly optimize the code here since this is a simple one-dimensional problem. Nevertheless, it is always healthy to check the type stability of the critical parts (like the loss functions) with `@code_warntype`. One should also check for any unusual time and allocation with `BenchmarkTools.@btime` or `BenchmarkTools.@benchmark`. We performed these analysis and everything seems good. We found it unnecessary to clutter the notebook with their outputs here, though.

## Data

Now we build the target model and draw samples from it.

The target model is a univariate random variable denoted by $X$ and defined by a probability distribution. Associated with that we consider its PDF and its score-function.

We played with different models, but settled here with one of those used in [Eric J. Ma, A Pedagogical Introduction to Score Models, April 21, 2021](https://ericmjl.github.io/score-models/).

We need enough sample points to capture the transition region in the mixture of Gaussians.

```@example simplescorematching
xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
x = permutedims(collect(xrange))

target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

target_pdf = pdf.(target_prob, x)
target_score = gradlogpdf.(target_prob, x)

y = target_score # just to simplify the notation
sample = permutedims(rand(rng, target_prob, 1024))
data = (x, y, target_pdf, sample)
```

Notice the data `x` and `sample` are defined as row vectors so we can apply the model in batch to all of their values at once. The values `y` are also row vectors for easy comparison with the predicted values. When, plotting, though, we need to revert them to vectors.

For the theoretical discussion, we denote the PDF by $p_X(x)$ and the score function by
```math
    \psi_X(x) = \frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x)).
```

Visualizing the sample data drawn from the distribution and the PDF.
```@example simplescorematching
plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(sample', normalize=:pdf, nbins=80, label="sample histogram")
plot!(x', target_pdf', linewidth=4, label="pdf")
scatter!(sample', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

Visualizing the score function.
```@example simplescorematching
plot(title="The score function and the discretized data", titlefont=10)

plot!(x', target_score', label="score function", markersize=2)
scatter!(sample', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

## The neural network model

The neural network we consider is a simple feed-forward neural network made of a single hidden layer, obtained as a chain of a couple of dense layers. This is implemented with the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package.

We will see that we don't need a big neural network in this simple example. We go as low as it works.

```@example simplescorematching
model = Chain(Dense(1 => 8, relu), Dense(8 => 1))
```

The [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) package uses explicit parameters, that are initialized (or obtained) with the `Lux.setup` function, giving us the *parameters* and the *state* of the model.
```@example simplescorematching
ps, st = Lux.setup(rng, model) # initialize and get the parameters and states of the model
```

## Loss functions for score-matching

The score-matching method from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) rests on the following ideas:

1. to fit the model by minimizing the expected square distance between the model score function $\psi(x; \theta)$ and the score function $\psi_X(x)$ of the random variable $X$,
```math
    J(\theta) = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \|\psi(x; \theta) - \psi_X(x)\|^2\;\mathrm{d}x;
```
2. Use a change of variables in the expectation and write that $J(\theta) = \tilde J(\theta) + C$, where $C$ is constant with respect to the parameters, so we only need to minimize $\tilde J$, given by
```math
    \tilde J(\theta) = \int_{\mathbb{R}} p_X(x) \left( \frac{1}{2}\psi(x; \theta)^2 + \frac{\partial}{\partial x} \psi(x; \theta) \right)\;\mathrm{d}x,
```
which does not involve the unknown score function of $X$. It does, however, involve the gradient of the modeled score function, which is expensive to compute.
1. In practice, the loss function is estimated via Monte-Carlo, so the unknown $p_X(x)$ is handled implicitly by the sample data $(x_n)_n$, and we minimize
```math
    {\tilde J}_{\mathrm{MC}} =  \frac{1}{N}\sum_{n=1}^N \left( \frac{1}{2}\psi(x_n; \theta)^2 + \frac{\partial}{\partial x} \psi(x_n; \theta) \right).
```

As mentioned before, computing a derivative to form the loss function becomes expensive when combined with the usual optimization methods to fit a neural network, as they require the gradient of the loss function itself, i.e. the optimization process involves the gradient of the gradient of something. Because of that, we approximate the derivative of the modeled score function by centered finite differences, i.e.
```math
    \frac{\partial}{\partial x} \psi(x_n; \theta) \approx \frac{\psi(x_n + \delta; \theta) - \psi(x_n - \delta; \theta)}{2\delta},
```
for a suitably small $\delta > 0$.

In this case, since we need compute $\psi(x_n + \delta; \theta)$ and $\psi(x_n - \delta; \theta)$, we avoid computing also $\psi(x_n; \theta)$ and approximate it with the average
```math
    \psi(x_n; \theta) \approx \frac{\psi(x_n + \delta; \theta) + \psi(x_n - \delta; \theta)}{2}.
```

Hence, we approximate $\tilde J(\theta)$ by the finite-difference version
```math
    \tilde J_{\mathrm{FD}}(\theta) = \int_{\mathbb{R}} p_X(x) \Bigg( \frac{1}{2}\left(\frac{\psi(x + \delta; \theta) + \psi(x - \delta; \theta)}{2}\right)^2 + \frac{\psi(x + \delta; \theta) - \psi(x - \delta; \theta)}{2\delta} \Bigg)\;\mathrm{d}x,
```

And we approximate ${\tilde J}_{\mathrm{MC}}$ by
```math
    {\tilde J}_{\mathrm{MC, FD}} =  \frac{1}{N}\sum_{n=1}^N \Bigg( \frac{1}{2}\left(\frac{\psi(x + \delta; \theta) + \psi(x - \delta; \theta)}{2}\right)^2 + \frac{\psi(x + \delta; \theta) - \psi(x - \delta; \theta)}{2\delta} \Bigg).
```

### Proof that $J(\theta) = \tilde J(\theta) + C$

Here is the one-dimensional version of the proof from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html).

Since this is a one-dimensional problem, the score function is a scalar and we have
```math
    \|\psi(x; \theta) - \psi_X(x)\|^2 = \psi(x; \theta)^2 - 2\psi(x; \theta) \psi_X(x) + \psi_X(x)^2.
```
Thus
```math
    J(\theta) = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \left(\psi(x; \theta)^2 - 2\psi(x; \theta)\psi_X(x)\right)\;\mathrm{d}x + C,
```
where
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \psi_X(x)^2\;\mathrm{d}x
```
does not depend on $\theta$.

For the mixed term, we use that the score function is
```math
    \psi_X(x) = \frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x)).
```
Differentiating the logarithm and using integration by parts, we find
```math
\begin{align*}
    -\int_{\mathbb{R}} p_X(x) \psi(x; \theta)\psi_X(x)\;\mathrm{d}x & = -\int_{\mathbb{R}} p_X(x) \psi(x; \theta)\frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x))\;\mathrm{d}x \\
    & = -\int_{\mathbb{R}} p_X(x) \psi(x; \theta)\frac{1}{p_X(x)}\frac{\mathrm{d}}{\mathrm{d}x}p_X(x)\;\mathrm{d}x \\
    & = -\int_{\mathbb{R}} \psi(x; \theta)\frac{\mathrm{d}}{\mathrm{d}x}p_X(x)\;\mathrm{d}x \\
    & = \int_{\mathbb{R}} \frac{\partial}{\partial x}\psi(x; \theta)p_X(x)\;\mathrm{d}x.
\end{align*}
```
Thus, we rewrite $J(\theta)$ as
```math
    J(\theta) = \int_{\mathbb{R}} p_X(x) \left(\frac{1}{2}\psi(x; \theta)^2 + \frac{\partial}{\partial x}\psi(x; \theta)\right)\;\mathrm{d}x + C,
```
which is precisely $J(\theta) = \tilde J(\theta) + C$.

For this proof to be justified, we need
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \psi_X(x)^2\;\mathrm{d}x < \infty,
```
and
```math
    \psi(x; \theta) p_X(x) \rightarrow 0, \quad |x| \rightarrow \infty,
```
for every $\theta$.

In our case, this looks fine, as $p_X(x)$ grows linearly, $\psi(x; \theta)$ grows at most linearly as well (as a composition of linear functions interposed with either ReLU or sigmoid or plain identity activation functions, which grow at most linearly), and $p_X(x)$ decays exponentially.

### Mean squared error loss function $J(\theta)$

For educational purposes, since we have the pdf and the score function, one of the ways we may train the model is directly on $J(\theta)$ itself. This is also useful to make sure that our network is able to model the desired score function.

Here is how we implement it.
```@example simplescorematching
function loss_function_mse(model, ps, st, data)
    x, y, target_pdf, sample = data
    y_pred, st = Lux.apply(model, x, ps, st)
    loss = mean(target_pdf .* (y_pred .- y) .^2)
    return loss, st, ()
end
```

### Mean squared error plain loss function

Still for educational purposes, we modify $J(\theta)$ for training, without weighting on the distribution of the random variable itself, as in $J(\theta)$. This has the benefit of giving more weight to the transition region. Here is how we implement it.
```@example simplescorematching
function loss_function_mse_plain(model, ps, st, data)
    x, y, target_pdf, sample = data
    y_pred, st = Lux.apply(model, x, ps, st)
    loss = mean(abs2, y_pred .- y)
    return loss, st, ()
end
```

### Mean squared error loss function with derivative

Again, for educational purposes, we may implement $\tilde J_{\mathrm{FD}}(\theta)$, as follows.

```@example simplescorematching
function loss_function_withFD(model, ps, st, data)
    x, y, target_pdf, sample = data
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

### Loss function ${\tilde J}_{\mathrm{MC, FD}}(\theta)$ with Monte-Carlo on the sample data with centered finite diferences

In practice we would use the sample data, not the supposedly unknown score function and PDF themselves. Here would be one implementation using finite differences, along with Monte-Carlo.
```@example simplescorematching
function loss_function_withFD_over_sample(model, ps, st, data)
    x, y, target_pdf, sample = data
    xmin, xmax = extrema(sample)
    delta = (xmax - xmin) / 2length(sample)
    y_pred_fwd, = Lux.apply(model, sample .+ delta, ps, st)
    y_pred_bwd, = Lux.apply(model, sample .- delta, ps, st)
    y_pred = ( y_pred_bwd .+ y_pred_fwd ) ./ 2
    dy_pred = (y_pred_fwd .- y_pred_bwd ) ./ 2delta
    loss = mean(dy_pred + y_pred .^ 2 / 2)
    return loss, st, ()
end
```

### Loss function $\tilde J_{\mathrm{MC}}(\theta)$ with Monte-Carlo on the sample data via Zygote

We can implement the actual loss function with the derivative of the modeled score function using some automatic differentiation tool, as follows, but we do not optimize with it here. We do this in a separate note, not to render this note too slowly.
```@example simplescorematching
function loss_function_Zygote_over_sample(model, ps, st, data)
    x, y, target_pdf, sample = data
    y_pred, st = Lux.apply(model, sample, ps, st)
    dy_pred = Zygote.gradient(s -> sum(model(s, ps, st)[1]), sample)[1]
    loss = mean(dy_pred .+ y_pred .^2 / 2)
    return loss, st, ()
end
```

## Optimization setup

### Optimization method

We use the classical Adam optimiser (see [Adam: A Method for Stochastic Optimization (Kingma, Diederik and Ba, Jimmy), In International Conference on Learning Representations (ICLR), 2015.](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8)), which is a stochastic gradient-based optimization method.

```@example simplescorematching
opt = Adam(0.03)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library, which is currently the only one implemented (there are pre-defined methods for `AutoForwardDiff()`, `AutoReverseDiff()`, `AutoFiniteDifferences()`, etc., but not implemented yet).
```@example simplescorematching
vjp_rule = Lux.Training.AutoZygote()
```

### Training loop

Here is the typical main training loop suggest in the [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) tutorials, but sligthly modified to save the history of losses per iteration.
```@example simplescorematching
function train(tstate::Lux.Experimental.TrainState, vjp, data, loss_function, epochs, numshowepochs=20)
    losses = zeros(epochs)
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
            loss_function, data, tstate)
        if rem(epoch, div(epochs, numshowepochs)) == 0
            println("Epoch: $(epoch) || Loss: $(loss)")
        end
        losses[epoch] = loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate, losses
end
```

We use the CPU instead of the GPU.
```@example simplescorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example simplescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_mse, data, tstate_org)
```

```@example simplescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_mse_plain, data, tstate_org)
```

```@example simplescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_withFD, data, tstate_org)
```

```@example simplescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_withFD_over_sample, data, tstate_org)
```

## Training

### Training with $J(\theta)$

Now we attempt to train the model, starting with $J(\theta)$.
```@example simplescorematching
@time tstate, losses = train(tstate_org, vjp_rule, data, loss_function_mse, 500)
nothing
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(x', y', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

And evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) .* dx)
target_pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', target_pdf_pred', label="recoverd")
```

### Training with plain MSE

Now we attempt to train it with the plain MSE. We do not reuse the state from the previous optimization. We start over at the initial state, for the sake of comparison of the different loss functions.
```@example simplescorematching
@time tstate, losses = train(tstate_org, vjp_rule, data, loss_function_mse_plain, 500)
nothing
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(x', y', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

And evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) * dx)
target_pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', target_pdf_pred', label="recoverd")
```

That is an almost perfect matching.

### Training with $\tilde J_{\mathrm{FD}}(\theta)$

Now we attempt to train it with $\tilde J_{\mathrm{FD}}(\theta)$. Again we start over with the untrained state of the model.
```@example simplescorematching
@time tstate, losses = train(tstate_org, vjp_rule, data, loss_function_withFD, 500)
nothing
```

We may try a little longer from this state on.
```@example simplescorematching
@time tstate, losses_more = train(tstate, vjp_rule, data, loss_function_withFD, 500)
append!(losses, losses_more)
nothing
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(x', y', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

And evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) * dx)
target_pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', target_pdf_pred', label="recoverd")
```

### Training with ${\tilde J}_{\mathrm{MC, FD}}$

Finally we attemp to train with the sample data.
```@example simplescorematching
@time tstate, losses = train(tstate_org, vjp_rule, data, loss_function_withFD_over_sample, 500)
nothing
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(x', y', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

And evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) * dx)
target_pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', target_pdf_pred', label="recoverd")
```

### Pre-training ${\tilde J}_{\mathrm{MC, FD}}$ with $J(\theta)$

Let us now pre-train the model with the $J(\theta)$ and see if $\tilde{\tilde{J}(\theta)}$ gets better.

```@example simplescorematching
tstate, = train(tstate_org, vjp_rule, data, loss_function_mse, 500)
nothing
```

```@example simplescorematching
tstate, losses = train(tstate, vjp_rule, data, loss_function_withFD_over_sample, 500)
nothing
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(x', y', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) * dx)
target_pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', target_pdf_pred', label="recoverd")
```

And evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## The need for enough sample points

One interesting thing is that enough sample points in the low-probability transition region is required for a proper fit, as the following example with few samples illustrates.

```@example simplescorematching
y = target_score # just to simplify the notation
sample = permutedims(rand(rng, target_prob, 128))
data = (x, y, target_pdf, sample)
```

```@example simplescorematching
tstate, losses = train(tstate_org, vjp_rule, data, loss_function_withFD_over_sample, 500)
nothing
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(x', y', label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) * dx)
target_pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', target_pdf_pred', label="recoverd")
```

And evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

## References

1. [Aapo Hyvärinen, "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6 (2005), 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [T. Pang, K. Xu, C. Li, Y. Song, S. Ermon, J. Zhu, Efficient Learning of Generative Models via Finite-Difference Score Matching, NeurIPS 2020](https://openreview.net/forum?id=LVRoKppWczk) - see also the [arxiv version](https://arxiv.org/abs/2007.03317)
1. [Eric J. Ma, A Pedagogical Introduction to Score Models, April 21, 2021](https://ericmjl.github.io/score-models/)
1. [Implemented python score_matching loss by Eric Ma](https://github.com/ericmjl/score-models/blob/main/score_models/losses/diffusion.py#L7)
1. [Y. Song, S. Garg, J. Shi, S. Ermon, Sliced Score Matching: A Scalable Approach to Density and Score Estimation, Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, PMLR 115:574-584, 2020](https://proceedings.mlr.press/v115/song20a.html) -- see also the [arxiv version](https://arxiv.org/abs/1905.07088)
1. [Y. Song, Sliced Score Matching: A Scalable Approach to Density and Score Estimation - blog](http://yang-song.net/blog/2019/ssm/)
1. [Adam: A Method for Stochastic Optimization (Kingma, Diederik and Ba, Jimmy), In International Conference on Learning Representations (ICLR), 2015.](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8) -- see also the [arxiv version](https://arxiv.org/abs/1412.6980).
