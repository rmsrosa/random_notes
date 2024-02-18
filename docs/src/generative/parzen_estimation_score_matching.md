# Score matching with Parzen estimation

```@meta
Draft = false
```

## Aim

Explore the use of *Parzen kernel estimation* to approximate the explicit score matching objective proposed by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) and discussed *en passant* by [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142). We illustrate the method by fiting a multi-layer perceptron to model the score function of a one-dimensional synthetic Gaussian-mixture distribution.

### Motivation

The motivation is to continue building a solid background on score-matching diffusion.

After the proposition of the **score-matching** method by [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html), which requires the derivative of the model score function, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) explored the idea of using **non-parametric Parzen density estimation** to circumvent this derivative, making a connection with denoising autoenconders, and proposing the **denoising score matching** method.

We will detail denoising score matching in a separate note. Here, we stop at the Parzen density estimation idea, which was used in [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) only as a step towards the denoising score matching.

## Objetive function approximating the explicit score matching objetive

The score-matching method from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) aims to fit the model score function $\psi(x; {\boldsymbol{\theta}})$ to the score function $\psi_X(x)$ of the random variable $X$, i.e. by minimizing the **explicit score matching** objective
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\psi}_{\mathbf{X}}(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x},
```
by using a change of variables in the expectation and writing that
```math
    J_{\mathrm{ESM}}({\boldsymbol{\theta}}) = {\tilde J}_{\mathrm{ISM}}({\boldsymbol{\theta}}) + C,
```
where $C$ is constant with respect to the parameters, so we only need to minimize the **implicit score matching** objetive J_{\mathrm{ISM}}({\boldsymbol{\theta}}), given by
```math
    J_{\mathrm{ISM}}({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_{\mathbf{X}}(\mathbf{x}) \left( \frac{1}{2}\left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}})\right\|^2 + \boldsymbol{\nabla}_{\mathbf{x}} \cdot \boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) \right)\;\mathrm{d}\mathbf{x},
```
which does not involve the unknown score function of $X$. It does, however, involve the gradient of the modeled score function, which is expensive to compute.

[Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) briefly mentioned that minimizing $J_{\mathrm{ESM}}({\boldsymbol{\theta}})$ directly is "basically a non-parametric estimation problem", but dismissed it for the "simple trick of partial integration to compute the objective function [$J_{\mathrm{ISM}}({\boldsymbol{\theta}})$] very easily". As we have seen, the trick is fine for model functions for which we can compute the gradient without much trouble, but for modeling it with a neural network, for instance, it becomes prohibitive.

A few years later, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) commented about the idea of using a Parzel kernel density estimation
```math
    \tilde p_\sigma(\mathbf{x}) = \int K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right) \;\mathrm{d}\tilde p_0(\mathbf{x}) = \frac{1}{\sigma N}\sum_{n=1}^N K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right),
```
where $\sigma > 0$ is a kernel window parameter, $K(\mathbf{x})$ is a kernel density (properly normalized to have mass one) and $\tilde p_0(\mathbf{x})$ is the **empirical distribution** given by
```math
    \tilde p_0(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^N \delta(\mathbf{x} - \mathbf{x}_n).
```
In this way, the explicit score matching objetive function is approximated by
```math
    {\tilde J}_{\mathrm{ESM_\sigma}}({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}^d} p_{\mathbf{X}}(\mathbf{x}) \left\|\boldsymbol{\psi}(\mathbf{x}; {\boldsymbol{\theta}}) - \boldsymbol{\nabla}_{\mathbf{x}}\log \tilde p_\sigma(\mathbf{x})\right\|^2\;\mathrm{d}\mathbf{x}.
```

However, [Pascal Vincent (2011)](https://doi.org/10.1162/NECO_a_00142) did not use this as a final objetive function. They further simplified the objective function by expanding the gradient of the logpdf of the Parzen estimator, writing a double integral with a conditional probability, and switching the order of integration. We will do this in a follow up note, but for the moment we will stop at ${\tilde J}_{\mathrm{ESM_\sigma}}({\boldsymbol{\theta}})$, use a Gaussian estimator, and see how this works.

Computing the score function with the Parzen estimation amounts to
```math
    \begin{align*}
        \boldsymbol{\nabla}_{\mathbf{x}}\log \tilde p_\sigma(\mathbf{x}) & = \boldsymbol{\nabla}_{\mathbf{x}}\log\left( \frac{1}{\sigma N}\sum_{n=1}^N K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right)\right) \\
        & = \frac{1}{\tilde p_\sigma(\mathbf{x})} \frac{1}{\sigma N}\sum_{n=1}^N \boldsymbol{\nabla}_{\mathbf{x}} \left(K\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right)\right) \\
        & = \frac{1}{\tilde p_\sigma(\mathbf{x})} \frac{1}{\sigma N}\sum_{n=1}^N \frac{1}{\sigma}\left(\boldsymbol{\nabla}_{\mathbf{x}} K\right)\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right).
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
    \boldsymbol{\nabla}_{\mathbf{x}}\log \tilde p_\sigma(\mathbf{x}) = \frac{1}{\tilde p_\sigma(\mathbf{x})} \frac{1}{\sigma N}\sum_{n=1}^N G\left(\frac{\mathbf{x} - \mathbf{x}_n}{\sigma}\right)\frac{\mathbf{x} - \mathbf{x}_n}{\sigma^2}.
```

### Proof that $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$

Here is the one-dimensional version of the proof from [Aapo Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html).

Since this is a one-dimensional problem, the score function is a scalar and we have
```math
    \|\psi(x; {\boldsymbol{\theta}}) - \psi_X(x)\|^2 = \psi(x; {\boldsymbol{\theta}})^2 - 2\psi(x; {\boldsymbol{\theta}}) \psi_X(x) + \psi_X(x)^2.
```
Thus
```math
    J({\boldsymbol{\theta}}) = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \left(\psi(x; {\boldsymbol{\theta}})^2 - 2\psi(x; {\boldsymbol{\theta}})\psi_X(x)\right)\;\mathrm{d}x + C,
```
where
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \psi_X(x)^2\;\mathrm{d}x
```
does not depend on ${\boldsymbol{\theta}}$.

For the mixed term, we use that the score function is
```math
    \psi_X(x) = \frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x)).
```
Differentiating the logarithm and using integration by parts, we find
```math
\begin{align*}
    -\int_{\mathbb{R}} p_X(x) \psi(x; {\boldsymbol{\theta}})\psi_X(x)\;\mathrm{d}x & = -\int_{\mathbb{R}} p_X(x) \psi(x; {\boldsymbol{\theta}})\frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x))\;\mathrm{d}x \\
    & = -\int_{\mathbb{R}} p_X(x) \psi(x; {\boldsymbol{\theta}})\frac{1}{p_X(x)}\frac{\mathrm{d}}{\mathrm{d}x}p_X(x)\;\mathrm{d}x \\
    & = -\int_{\mathbb{R}} \psi(x; {\boldsymbol{\theta}})\frac{\mathrm{d}}{\mathrm{d}x}p_X(x)\;\mathrm{d}x \\
    & = \int_{\mathbb{R}} \frac{\partial}{\partial x}\psi(x; {\boldsymbol{\theta}})p_X(x)\;\mathrm{d}x.
\end{align*}
```
Thus, we rewrite $J({\boldsymbol{\theta}})$ as
```math
    J({\boldsymbol{\theta}}) = \int_{\mathbb{R}} p_X(x) \left(\frac{1}{2}\psi(x; {\boldsymbol{\theta}})^2 + \frac{\partial}{\partial x}\psi(x; {\boldsymbol{\theta}})\right)\;\mathrm{d}x + C,
```
which is precisely $J({\boldsymbol{\theta}}) = \tilde J({\boldsymbol{\theta}}) + C$.

For this proof to be justified, we need
```math
    C = \frac{1}{2}\int_{\mathbb{R}} p_X(x) \psi_X(x)^2\;\mathrm{d}x < \infty,
```
and
```math
    \psi(x; {\boldsymbol{\theta}}) p_X(x) \rightarrow 0, \quad |x| \rightarrow \infty,
```
for every ${\boldsymbol{\theta}}$.

In our case, this looks fine, as $p_X(x)$ grows linearly, $\psi(x; {\boldsymbol{\theta}})$ grows at most linearly as well (as a composition of linear functions interposed with either ReLU or sigmoid or plain identity activation functions, which grow at most linearly), and $p_X(x)$ decays exponentially.


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

nothing # hide
```

### Reproducibility

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

## Data

Now we build the target model and draw samples from it.

The target model is a univariate random variable denoted by $X$ and defined by a probability distribution. Associated with that we consider its PDF and its score-function.

We need enough sample points to capture the transition region in the mixture of Gaussians.

```@example simplescorematching
xrange = range(-10, 10, 200)
dx = Float64(xrange.step)
x = permutedims(collect(xrange))

target_prob = MixtureModel([Normal(-3, 1), Normal(3, 1)], [0.1, 0.9])

target_pdf = pdf.(target_prob, x)
target_score = gradlogpdf.(target_prob, x)

y = target_score # just to simplify the notation
sample_points = permutedims(rand(rng, target_prob, 1024))
```

Notice the data `x` and `sample_points` are defined as row vectors so we can apply the model in batch to all of their values at once. The values `y` are also row vectors for easy comparison with the predicted values. When, plotting, though, we need to revert them to vectors.

For the theoretical discussion, we denote the PDF by $p_X(x)$ and the score function by
```math
    \psi_X(x) = \frac{\mathrm{d}}{\mathrm{d}x}\log(p_X(x)).
```

Visualizing the sample data drawn from the distribution and the PDF.
```@example simplescorematching
plot(title="PDF and histogram of sample data from the distribution", titlefont=10)
histogram!(sample_points', normalize=:pdf, nbins=80, label="sample histogram")
plot!(x', target_pdf', linewidth=4, label="pdf")
scatter!(sample_points', s -> pdf(target_prob, s), linewidth=4, label="sample")
```

Visualizing the score function.
```@example simplescorematching
plot(title="The score function and the sample", titlefont=10)

plot!(x', target_score', label="score function", markersize=2)
scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)
```

```@example simplescorematching
G(x) = exp(-x^2 / 2) / √(2π)
sigma = 0.4

psigma(x) = mean(G( (x - xn) / sigma ) for xn in sample_points) / sigma

score_parzen1(x, sigma, sample_points) = mean(G( (x - xn) / sigma ) * (xn - x) / sigma^2 for xn in sample_points) / psigma(x) / sigma

score_parzen = [score_parzen1(x, sigma, sample_points) for x in sample_points]

data = (sample_points, score_parzen)
```

```@example simplescorematching
plot(title="The score functions of the target density and the kernel density estimation", titlefont=10)

plot!(x', target_score', label="score function", markersize=2)
scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="score at data", markersize=2, alpha=0.5)

scatter!(sample_points', s -> score_parzen1(s, sigma, sample_points), label="score from parzen estimation", markersize=2)
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

### Mean squared error loss function $J({\boldsymbol{\theta}})$

For educational purposes, since we have the pdf and the score function, one of the ways we may train the model is directly on $J({\boldsymbol{\theta}})$ itself. This is also useful to make sure that our network is able to model the desired score function.

Here is how we implement it.
```@example simplescorematching
function loss_function_parzen(model, ps, st, data)
    sample_points, score_parzen = data
    y_score_pred, st = Lux.apply(model, sample_points, ps, st)
    loss = mean(abs2, y_score_pred - score_parzen)
    return loss, st, ()
end
```

## Optimization setup

### Optimization method

We use the classical Adam optimiser (see [Kingma and Ba (2015)](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8)), which is a stochastic gradient-based optimization method.

```@example simplescorematching
opt = Adam(0.01)

tstate_org = Lux.Training.TrainState(rng, model, opt)
```

### Automatic differentiation in the optimization

As mentioned, we setup differentiation in [LuxDL/Lux.jl](https://github.com/LuxDL/Lux.jl) with the [FluxML/Zygote.jl](https://github.com/FluxML/Zygote.jl) library, which is currently the only one implemented (there are pre-defined methods for `AutoForwardDiff()`, `AutoReverseDiff()`, `AutoFiniteDifferences()`, etc., but not implemented yet).
```@example simplescorematching
vjp_rule = Lux.Training.AutoZygote()
```

### Processor

We use the CPU instead of the GPU.
```@example simplescorematching
dev_cpu = cpu_device()
## dev_gpu = gpu_device()
```

### Check differentiation

Check if Zygote via Lux is working fine to differentiate the loss functions for training.
```@example simplescorematching
Lux.Training.compute_gradients(vjp_rule, loss_function_parzen, data, tstate_org)
```

### Training loop

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

## Training

### Training with $J({\boldsymbol{\theta}})$

Now we attempt to train the model, starting with $J({\boldsymbol{\theta}})$.
```@example simplescorematching
@time tstate, losses, tstates = train(tstate_org, vjp_rule, data, loss_function_parzen, 500, 20, 125)
nothing # hide
```

Testing out the trained model.
```@example simplescorematching
y_pred = Lux.apply(tstate.model, dev_cpu(x), tstate.parameters, tstate.states)[1]
```

Visualizing the result.
```@example simplescorematching
plot(title="Fitting", titlefont=10)

plot!(x', y', linewidth=4, label="score function")

scatter!(sample_points', s -> gradlogpdf(target_prob, s), label="data", markersize=2)

plot!(x', y_pred', linewidth=2, label="predicted MLP")
```

Just for the fun of it, let us see an animation of the optimization process.
```@setup simplescorematching
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

```@example simplescorematching
gif(anim, fps = 20) # hide
```

We also visualize the evolution of the losses.
```@example simplescorematching
plot(losses, title="Evolution of the loss", titlefont=10, xlabel="iteration", ylabel="error", legend=false)
```

Recovering the PDF of the distribution from the trained score function.
```@example simplescorematching
paux = exp.(accumulate(+, y_pred) .* dx)
pdf_pred = paux ./ sum(paux) ./ dx
plot(title="Original PDF and PDF from predicted score function", titlefont=10)
plot!(x', target_pdf', label="original")
plot!(x', pdf_pred', label="recoverd")
```

## References

1. [Pascal Vincent (2011), "A connection between score matching and denoising autoencoders," Neural Computation, 23 (7), 1661-1674, doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
1. [Aapo Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", Journal of Machine Learning Research 6, 695-709](https://jmlr.org/papers/v6/hyvarinen05a.html)
1. [D. P. Kingma, J. Ba (2015), Adam: A Method for Stochastic Optimization, In International Conference on Learning Representations (ICLR)](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8) -- see also the [arxiv version](https://arxiv.org/abs/1412.6980)
