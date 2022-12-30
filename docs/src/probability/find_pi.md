# Estimating the value of π via frequentist and Bayesian methods

In the frequentist approach, we draw a number of samples uniformly distributed in the unit square and compute how many of them fall into the quarter circle. This yields an estimate for the area of the quarter circle along with confidence intervals. 

In the Bayesian approach, we start with a prior estimating the value of pi and update our prior to refine the estimate and the confidence levels, according to the posterior.

## The Julia packages

For the numerical implementation, we need some packages.

We use `Distributions.jl` for the common distributions, such as Uniform, Beta, Bernoulli, Normal, etc.

```@example find_pi
using Distributions
```

For reproducibility, we set the seed for the pseudo random number generators.

```@example find_pi
using Random
Random.seed!(12)
```

For plotting, we use `StatsPlots.jl`

```@example find_pi
using StatsPlots
```

## The frequentist approach

This is a classical example illustrating the Monte Carlo method. We generate a bunch of samples $(x_i, y_i)$ from two independent uniform distributions on the interval $[0, 1]$ and check whether they belong to the unit circle (quarter circle, more precisely) or not, i.e. whether $x_i^2 + y_i^2 \leq 1.$ The distribution uniformly fills up the unit square, which has area one, and some of them will be in the quarter circle. The proportion of those in the circle approximates the area of the quarter circle over that of the unit square, i.e. $\pi/4$. Multiplying it by four, yields an estimate for $\pi$. The more samples we use, the closer we expect the mean to approximate this value.

We start choosing a maximum number `N` of samples. We will analyse the estimate for each `i` up to `N`, to have an idea how the value and our confidence on it improves with the number of samples.

```@example find_pi
N = 10_000
```

Now we sample `N` pairs of numbers uniformly on the unit square

```@example find_pi
positions_f = rand(N, 2)
```

With the sample at hand, we compute their distance to the origin and check whether they belong to the unit circle or not, giving a sequence `x_f` of random variables with values `1` or `0`, with the respective indication.

```@example find_pi
distance_f = sum(abs2, positions_f, dims=2)
x_f = vec(distance_f) .≤ 1
```

For each `n` between `1` and `N`, we compute the sample mean `q_f[n]` of `x_f[1], …, x_f[n]`, and the sample standard error `s_f[n]`.

```@example find_pi
q_f = cumsum(x_f) ./ (1:N)
s_f = [1.0; [√(var(view(x_f, 1:n), mean=q_f[n])/n) for n in 2:N]]
```

The sample means approximate the value of $\pi/4$, so we multiply it by $4$ to have an estimate of $\pi$. Accordingly, we multipy the standard error by $4$. The 95% confidence interval is estimated by twice the standard error around the mean. This is illustrated in the following plots. Of course, for small samples, we should use the t-Student distribution, but we concentrate on not-so-small samples and just use the normal distribution, relying on the Central Limit Theorem.

```@example find_pi
plot(10:N, 4q_f[10:N], ribbon = 8s_f[10:N], label="estimate")
hline!(10:N, [π], label="true value")
```

A close up of the 10% first samples

```@example find_pi
Nrange = 10:min(N, div(N, 10))
plot(Nrange, 4q_f[Nrange], ribbon = 8s_f[Nrange], label="estimate")
hline!(Nrange, [π], label="true value")
```

The probability distribution for the estimate of $pi$ after `N` samples is illustrated below in a few cases.

```@example find_pi
pp = 0.0:0.001:1.0

plt = plot(title="Evolution of our belief in the value of π\nwith respect to the sample size n", titlefont=10, xlims=(0.5, 1.0))
    
for n in (div(N, 1000), div(N, 100), div(N, 10), N)
    plot!(plt, pp, pdf.(Normal(q_f[n], s_f[n]), pp), label="n=$n", fill=true)
end
vline!(plt, [π/4], color=:black, label="π/4")
display(plt)
```

## The Bayesian approach

In the Bayesian approach, we start guessing the area of the quarter circle, or, more precisely, the probability that it be a certain value within a certain range. It is reasonable to assume it is a little over half the area of the unit circle and not too close to 1, with higher probability of being closer to the middle of these two values. We could use a normal distribution, but a better choice is a Beta distribution since it is conjugate to the likelihood, which is expected to be a Beta distribution $B(\alpha, \beta)$ with density $x^\alpha (1 - x)^\beta$, where $\alpha$ counts as the number of success draws (within the quarter circle) and $\beta$ as the number of failures (outside the quarter circle). 

So we choose as prior the distribution $Beta(\alpha, \beta)$ with something like $\alpha = 24$ and $\beta = 8$, in which case the mean is $\alpha / (\alpha + \beta) = 24/32 = 3/4 = 0.75$ and the variance is $αβ/((α + β)^2(α + β + 1)) = 192/33792 ≈ 0.00568$. These are our *hyperparameters.* Let us visualize this prior distribution

```@example find_pi
prior_distribution = Beta(24,8)

plt = plot(pp, pdf.(prior_distribution, pp), label=nothing, title="Density of the (prior) distribution $prior_distribution\nmean = $(round(mean(prior_distribution), sigdigits=4)); standard deviation = $(round(std(prior_distribution), sigdigits=5))", titlefont=10, fill=true, legend=:topleft)

vline!(plt, [mean(prior_distribution)], label="mean $(round(mean(prior_distribution), sigdigits=4))")

vline!(plt, mean(prior_distribution) .+ [-std(prior_distribution), +std(prior_distribution)], label="mean +/- std: $(round(mean(prior_distribution), sigdigits=4)) +/- $(round(std(prior_distribution), sigdigits=5))")

display(plt)
```

Just for the sake of illustration, we draw some sample from the prior

```@example find_pi
priordata = rand(prior_distribution, N)
```

Now we generate some "real" data. Since we know $\pi$ with high precision, we use that to generate the data, which are Bernoulli trials with probability $\pi/4$, and check the mean, which should be close to p_true.

```@example find_pi
p_true = π/4
```

```@example find_pi
data = rand(Bernoulli(p_true), N)

mean(data)
```

We update our prior in closed form, since the prior is a Beta distribution, which is a conjugate prior to the Bernoulli distribution. It is updated by simply counting the number of sucesses and the number of failures to the shape parameters $\alpha$ and $\beta$, respectively.

```@example find_pi
function update_belief(prior_distribution::Beta, data::AbstractArray{Bool})
    # Count the number of successes and failures.
    successes = sum(data)
    failures = length(data) - successes

    # Update our prior belief using the fact that Beta is conjugate distribution.
    return Beta(prior_distribution.α + successes, prior_distribution.β + failures)
end
```

In order to see the evolution of the posterior with respect to the added evidence, we first we update the prior with a small part of the data

```@example find_pi
Ns = div.(N, (1000, 100, 10, 1))
posterior_distributions = Dict(n => update_belief(prior_distribution, view(data, 1:n)) for n in Ns)
```

Now we visualize the posterior, which is the updated distribution.

```@example find_pi
plt = plot(title="Density of the posterior distributions Beta(α', β')", titlefont=10, legend=:topleft, xlims=(0.5, 1.0))
for n in Ns
    distr = posterior_distributions[n]
    plot!(plt, pp, pdf.(distr, pp), label="N=$n; α'=$(distr.α), β'=$(distr.β), mean=$(round(mean(distr), sigdigits=4))", fill=true, alpha=0.5)
end
vline!(plt, [π/4], label=nothing, color=:black)
```

## Performance of the two methods

It is not so relevant to compare the performance of the two methods above since this is a toy problem and not representative of the variety of situations that can be handled by either frequentist or Bayesian approaches. Drawing pseudo random numbers is pretty cheap and the frequentist approach above is much faster, even with the code not optimized for performance. The use of the Bayesian approach is more relevant in more complex cases and where sampling is more expensive. But more important than that is the perspective of the Bayesian approach of treating parameters as random variables and of their distributions as uncertainties, or quantifiers of our belief on their values.
