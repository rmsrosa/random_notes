# Modeling mortality tables

In this section, we attempt to fit the Gompertz-Makeham and the Heligman-Pollard models to a selected mortality table.

## The Gompertz-Makeham model

In the Gompertz-Makeham model, the *force of mortality* (akin to the hazard function) $\mu_x$ is given by

```math
    \mu_x = Ae^{Bx} + C,
```

for age $x$. The [force of mortality](https://en.wikipedia.org/wiki/Force_of_mortality) represents the *instantaneous* rate of mortality at a certain age $x$, in an annual basis.  It is closely related to the [mortality rate](https://en.wikipedia.org/wiki/Mortality_rate) $q_x$, which is the percentage of deaths in a population per year, which can be interpreted as the probability a person at an age $x$, in years, dies before reaching age $x+1$. Under certain assumptions, the two are related by

```math
    \mu_x = \frac{q_x}{1 - q_x}, \qquad q_x = \frac{\mu_x}{1 + \mu_x}.
```

The terms $A$ and $B$ are associated with the *Gompertz law* $Ae^{Bx}$, while the term $C$ is an additional term provided by *Makeham,* which combine to form the Gompertz-Makeham model.

## The Heligman-Pollard model

The Gompertz-Makeham model approximates reasonably well the force of mortality, especially the growth seen in adult years, but a better model is the Heligman-Pollard, that also approximates well the childhood and young groups and the eldery years. This model takes the form

```math
    \mu_x = A^{(x + B)^C} + D e^{-E \ln(x/F)^2} + \frac{GH^x}{1 + KGH^x},
```

for suitable coefficients $A, B, C, D, E, F, G, H, K$. It is common to see the Heligman-Pollard models with $K=0$, which is the main model suggested by the authors, but in the same paper they also discuss two extensions of the model, one of them being the one above.

We can clearly distinguish three terms in the model, with the first term, with parameters $A, B, C$, modeling the steep exponential decline in mortality in the early childhood years due in part to a relatively high degree of mortality in the newborn; the second term, with parameters $D, E, F$, representing the log-normal bump in mortality in the youth ages; and the last term, with parameters $G, H, K$, with the exponential growth at middle to older ages. Notice the last term follows the original Gompertz law (without the additional term due to Makeham).

## A mortality table

We start by loading the necessary packages

```@example mortality
using Distributions, Turing, StatsPlots
```

We illustrate the modeling process with the [United States 1997 mortality table](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiirPed8Kn8AhXyL7kGHeuiA5wQFnoECA8QAQ&url=http%3A%2F%2Fwww.epidemiolog.net%2Fstudymat%2FLifetable.xls&usg=AOvVaw2Dsu6y07w6bRNDIcvyDYQK). We removed the data from ages 0-1 and 100+ hoping for a better fit.

```@example mortality
x = collect(1:99)
# mortality rate
qx = [
    0.00055
    0.00036
    0.00029
    0.00023
    0.00021
    0.00020
    0.00019
    0.00017
    0.00015
    0.00014
    0.00014
    0.00019
    0.00028
    0.00041
    0.00055
    0.00068
    0.00078
    0.00085
    0.00089
    0.00093
    0.00098
    0.00101
    0.00101
    0.00101
    0.00100
    0.00099
    0.00100
    0.00103
    0.00108
    0.00114
    0.00119
    0.00126
    0.00133
    0.00140
    0.00149
    0.00157
    0.00167
    0.00178
    0.00192
    0.00206
    0.00222
    0.00239
    0.00257
    0.00278
    0.00300
    0.00325
    0.00352
    0.00380
    0.00411
    0.00444
    0.00482
    0.00524
    0.00571
    0.00623
    0.00685
    0.00755
    0.00833
    0.00916
    0.01005
    0.01101
    0.01208
    0.01321
    0.01439
    0.01560
    0.01679
    0.01802
    0.01948
    0.02127
    0.02338
    0.02565
    0.02799
    0.03043
    0.03297
    0.03563
    0.03843
    0.04147
    0.04494
    0.04904
    0.05385
    0.05938
    0.06555
    0.07241
    0.07990
    0.08812
    0.09653
    0.10556
    0.11539
    0.12616
    0.13802
    0.15085
    0.16429
    0.17813
    0.19250
    0.20764
    0.22354
    0.23999
    0.25653
    0.27295
    0.28915
]
# mx = - log.(1.0 .- qx) # force of mortality # roughly the same as below
mx = qx ./ (1.0 .- qx) # force of mortality
scatter(x, qx, yscale=:log10, legend=:topleft, label="qx")
scatter!(x, mx, yscale=:log10, legend=:topleft, label="mx")
```

## The Gompertz-Makeham model in Turing.jl

First we start by defining the function that characterizes the Gompertz-Makeham model
```@example mortality
function gompertz_makeham(x, p)
    A, B, C = p
    m = A * exp(B * x) + C
    return m
end
```

Now we define the compound probability model, assigning Beta distributions to the parameters. But, as mentioned before, the initial prior is very important. It is not easy to get a good fit. So, first, we approximate by "hand", with trial and error, to get the following approximate fit:

```@example mortality
let (A, B, C) = (0.00002, 0.10, 0.001)
    m = A * exp.(B * x) .+ C
    plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
    plot!(plt, x, m, label="Gompertz-Makeham hand-fit")
    scatter!(plt, x, mx, label="data")
end
```

With these values, we define the prior Beta distributions for the compound probability model with parameters that yield a mean near those values.

```@example mortality
@model function gompertz_makeham_model(x, m)
    A ~ Beta(2, 99998)
    B ~ Beta(2, 18)
    C ~ Beta(2, 1998)
    σ² ~ InverseGamma()
    σ = sqrt(σ²)
    p = (A, B, C)

    for i in eachindex(x)
        y = gompertz_makeham(x[i], p)
        m[i] ~ Normal(y, σ)
    end
end
```

Now we instantiate the Turing model

```@example mortality
model_gm = gompertz_makeham_model(x, mx)
```

and fit it:

```@example mortality
# chain_gm = sample(model_gm, HMC(0.05, 10), 40_000)
chain_gm = sample(model_gm, NUTS(0.65), 5_000)
```

Here is the result of the MCMC:

```@example mortality
plot(chain_gm)
```

We can see the mean values of the parameters as follows

```@example mortality
mean(chain_gm)
```

The mean fit is given by

```@example mortality
m_gm = [gompertz_makeham(xi, mean(chain_gm, [:A, :B, :C]).nt.mean) for xi in x]
```

and we plot it

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
plot!(plt, x, m_gm, label="Gompertz-Makeham fit")
scatter!(plt, x, mx, label="data")
```

It remains to compute the 95% credible interval,

```@example mortality
quantiles_gm = reduce(
    hcat,
    quantile(
        [
            gompertz_makeham(xi, (A, B, C)) for (A, B, C) in eachrow(view(chain_gm.value.data, :, 1:3, 1))
        ],
        [0.025, 0.975]
        )
    for xi in x
)
```

and plot it

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
plot!(plt, x, m_gm, ribbon=(m_gm .- view(quantiles_gm, 1, :), view(quantiles_gm, 2, :) .- m_gm), label="Gompertz-Makeham fit")
scatter!(plt, x, mx, label="data")
```

Notice how the function with the means of the parameters is outside the quantiles, which is based on the function values of the parameter samples. Let's check the ensemble.

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=nothing)
plot!(plt, x, m_gm, label="Bayesian fitted line", color=2)
for (A, B, C) in eachrow(view(chain_gm.value.data, :, 1:3, 1))
    plot!(plt, x, x -> gompertz_makeham(x, (A, B, C)), alpha=0.01, color=2, label=false)
end
scatter!(plt, x, mx, color=1)
```

Let's look at just a few samples to have a better look at the dependence of the function on the sampled values:[^off]

[^off]: How often do you see $x \mapsto f_{\mathrm{mean}(p)}(x)$ fall off the credible interval of the family $x \mapsto \{f_p(x)\}_p$, where $p$ is the set of parameters?

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=nothing)
plot!(plt, x, m_gm, label="Bayesian fitted line", color=2)
for (A, B, C) in eachrow(view(chain_gm.value.data, 1:50:500, 1:3, 1))
    plot!(plt, x, x -> gompertz_makeham(x, (A, B, C)), alpha=0.4, color=3, label=false)
end
scatter!(plt, x, mx, color=1)
```

## The Heligman-Pollard in Turing.jl

Now we consider the Heligman-Pollard model.

First we start by defining the function that characterizes the model:

```@example mortality
function heligman_pollard(x, p)
    A, B, C, D, E, F, G, H, K = p
    m = A^((x + B)^C) + D * exp(-E * log(x / F)^2) + (G * H^x) / (1 + K * G * H^x)
    return m
end
```

Now we define the compound probability model. As mentioned before, the initial prior is very important. We start with numbers of the order of those given in the original article by Heligman and Pollard. They considered a number of examples, but, as a starting point, we borrow only the data from the 1970-1972 period, separated by gender:

| Parameter | Males 1970-72 | Females 1970-1972 |
| --- | --- | --- |
| A | 0.00160 | 0.00142 |
| B | 0.00112 | 0.0350 |
| C | 0.1112 | 0.1345 |
| D | 0.00163 | 0.00038 |
| E | 16.71 | 21.86 |
| F | 20.03 | 18.27 |
| G | 0.0000502 | 0.0000507 |
| H | 1.1074 | 1.0937 |
| K | 2.416 | -2.800 |

The difference in the sign of $K$ between the male and female populations is justified by the difference in mortality for the elderly, although the data showed in the article both look more like in the description of the male mortality. Let's see how an approximation of that looks like, keeping a positive sign for $K$ because of this observation.

```@example mortality
let (A, B, C, D, E, F, G, H, K) = (0.0015, 0.018, 0.012, 0.001, 19.0, 19.0, 0.00005, 1.1, 1.0)
    m = [heligman_pollard(xi, (A, B, C, D, E, F, G, H, K)) for xi in x]
    plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
    plot!(plt, x, m, label="Heligman-Pollard hand-fit")
    scatter!(plt, x, mx, label="data")
end
```

Ok, that seems like a reasonable starting point. So we choose the following priors for each parameter:

| Parameter | prior | mean |
| --- | --- | --- |
| A | Beta(15, 9985) | 0.0015 |
| B | Beta(18, 982) | 0.018 |
| C | Beta(12, 988) | 0.012 |
| D | Beta(2, 1998) | 0.001 |
| E | Gamma(38, 0.5) | 19.0 |
| F | Gamma(38, 0.5) | 19.0 |
| G | Beta(5, 99995) | 0.00005 |
| H | Gamma(2, 0.5) | 1.0 |
| K | Gamma(2, 0.5) | 1.0 |

But these turned out not to be so good. We either look for better informative priors or use slightly less informative priors. We fiddle a little bit with the parameters to get the following improvement.

```@example mortality
let (A, B, C, D, E, F, G, H, K) = (0.0003, 0.02, 0.08, 0.0008, 15.0, 20.0, 0.00005, 1.1, 1.0)
    m = [heligman_pollard(xi, (A, B, C, D, E, F, G, H, K)) for xi in x]
    plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
    plot!(plt, x, m, label="Heligman-Pollard hand-fit")
    scatter!(plt, x, mx, label="data")
end
```

Based on that, we choose the following priors:

```@example mortality
@model function heligman_pollard_model(x, m)
    A ~ Beta(3, 9997) # Beta(1, 660) # Beta(15, 9985)
    B ~ Beta(2, 98) # Beta(1, 50) # Beta(18, 982)
    C ~ Beta(8, 92) # Beta(12, 988)
    D ~ Beta(8, 9992) # Beta(1, 999) # Beta(2, 1998)
    E ~ Gamma(30, 0.5) # Gamma(19, 1) # Gamma(38, 0.5)
    F ~ Gamma(40, 0.5) # Gamma(19, 1) # Gamma(38, 0.5)
    G ~ Beta(5, 99995) # Beta(1, 19999) # Beta(5, 99995)
    H ~ Gamma(2.2, 0.5) # Gamma(1, 1) # Gamma(2, 0.5)
    K ~ Gamma(2.0, 0.5) # Gamma(1, 1) # Gamma(2, 0.5)
    σ² ~ InverseGamma()
    σ = sqrt(σ²)

    for i in eachindex(x)
        y = A ^ ((x[i] + B) ^ C) + D * exp( - E * (log(x[i]) - log(F)) ^ 2) + G * H ^ (x[i]) # / (1 + K * G * H ^ (x[i]) )
        m[i] ~ Normal(y, σ)
    end
end
```

Now we instantiate the Heligman-Pollard Turing model

```@example mortality
model_hp = heligman_pollard_model(x, mx)
```

and fit it:

```@example mortality
# chain_hp = sample(model_hp, HMC(0.05, 10), 4_000) # Pure HMC didn't converge
# chain_hp = sample(model_hp, MH(), 5_000) # Metropolis-Hasting didn't converge
# chain_hp = sample(model_hp, Gibbs(MH(:A, :B, :C), MH(:D, :E, :F), HMC(0.65, 5, :G, :H, :K)), 5_000) # I am afraid I am not getting this right
chain_hp = sample(model_hp, NUTS(0.85), 5_000)
```

Here is the result of the MCMC:

```@example mortality
plot(chain_hp)
```

We can see the mean values of the parameters as follows

```@example mortality
mean(chain_hp)
```

The mean fit is given by

```@example mortality
m_hp = [heligman_pollard(xi, mean(chain_hp, [:A, :B, :C, :D, :E, :F, :G, :H, :K]).nt.mean) for xi in x]
```

and we plot it

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
plot!(plt, x, m_hp, label="Heligman-Pollard fit")
scatter!(plt, x, mx, label="data")
```

It remains to compute the 95% credible interval,

```@example mortality
quantiles_hp = reduce(
    hcat,
    quantile(
        [
            heligman_pollard(xi, p) for p in eachrow(view(chain_hp.value.data, :, 1:9, 1))
        ],
        [0.025, 0.975]
        )
    for xi in x
)
```

and plot it

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
plot!(plt, x, m_hp, ribbon=(m_hp .- view(quantiles_hp, 1, :), view(quantiles_hp, 2, :) .- m_hp), label="Heligman-Pollard fit")
scatter!(plt, x, mx, label="data")
```

Notice how the function with the means of the parameters is again outside the quantiles, which is based on the function values of the parameter samples. Let's check the ensemble with the first one thousand parameters values:

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=nothing)
plot!(plt, x, m_hp, label="Bayesian fitted line", color=2)
for p in eachrow(view(chain_hp.value.data, 1:1_000, 1:9, 1))
    plot!(plt, x, x -> heligman_pollard(x, p), alpha=0.01, color=2, label=false)
end
scatter!(plt, x, mx, color=1)
```

Let's look at just a few samples to have a better look at the dependence of the function on the sampled values:

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=nothing)
plot!(plt, x, m_hp, label="Bayesian fitted line", color=2)
for p in eachrow(view(chain_hp.value.data, 1:50:500, 1:9, 1))
    plot!(plt, x, x -> heligman_pollard(x, p), alpha=0.4, color=3, label=false)
end
scatter!(plt, x, mx, color=1)
```
