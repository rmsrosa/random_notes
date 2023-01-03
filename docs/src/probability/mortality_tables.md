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

for suitable coefficients $A, B, C, D, E, F, G, H, K$[^K].

[^K]: Wait, all Heligman-Pollard models I see don't have the denominator above, i.e. have $K=0$.

We can clearly distinguish three terms in the model, with the first term, with parameters $A, B, C$, modeling the steep exponential decline in mortality in the early childhood years, following a relatively high degree of mortality in the newborn; the second term, with parameters $D, E, F$, representing the log-normal growth in mortality in the middle ages; and the last term, with parameters $G, H, K$, with the exponential growth at older ages. Notice the last term follows the original Gompertz law (without the Makeham term).

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
# population
lx = [
    99_277
    99_222
    99_187
    99_158
    99_135
    99_114
    99_094
    99_076
    99_059
    99_044
    99_030
    99_016
    98_997
    98_970
    98_929
    98_875
    98_807
    98_730
    98_646
    98_559
    98_467
    98_371
    98_271
    98_172
    98_073
    97_975
    97_878
    97_780
    97_679
    97_574
    97_462
    97_346
    97_224
    97_094
    96_958
    96_814
    96_662
    96_501
    96_329
    96_144
    95_946
    95_733
    95_504
    95_259
    94_994
    94_709
    94_401
    94_069
    93_711
    93_326
    92_912
    92_464
    91_979
    91_454
    90_884
    90_262
    89_580
    88_834
    88_020
    87_136
    86_176
    85_135
    84_011
    82_802
    81_510
    80_142
    78_697
    77_164
    75_523
    73_757
    71_866
    69_854
    67_728
    65_495
    63_162
    60_734
    58_216
    55_600
    52_873
    50_026
    47_055
    43_971
    40_787
    37_528
    34_221
    30_918
    27_654
    24_463
    21_377
    18_426
    15_647
    13_076
    10_747
    8_678
    6_876
    5_339
    4_058
    3_017
    2_193
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

With these values, we define the prior Beta distributions for the compound probability model

```@example mortality
@model function gompertz_makeham_model(x, m)
    A ~ Beta(1, 50000)
    B ~ Beta(1, 10)
    C ~ Beta(1, 1000)
    σ² ~ InverseGamma()
    σ = sqrt(σ²)
    p = (A, B, C)

    for i in eachindex(x)
        # y = A * exp(B * x[i]) + C
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

The mean fit is given by

```@example mortality
m = [gompertz_makeham(xi, mean(chain_gm, [:A, :B, :C]).nt.mean) for xi in x]
```

and we plot it

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=:topleft)
plot!(plt, x, x -> gompertz_makeham(x, mean(chain_gm, [:A, :B, :C]).nt.mean), label="Gompertz-Makeham fit")
scatter!(plt, x, mx, label="data")
```

It remains to compute the 95% credible interval,

```@example mortality
quantiles = reduce(
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
plot!(plt, x, m, ribbon=(m .- view(quantiles, 1, :), view(quantiles, 2, :) .- m), label="Gompertz-Makeham fit")
scatter!(plt, x, mx, label="data")
```

Weird. Why is the function with the means of the parameters is outside que quantiles, which is based on the function values of the parameter samples?

Let's check the ensemble.

```@example mortality
plt = plot(yscale=:log10, title="Force of mortality", titlefont=10, xlabel="age", ylabel="force of mortality", legend=nothing)
plot!(plt, x, m, label="Bayesian fitted line", color=2)
for (A, B, C) in eachrow(view(chain_gm.value.data, :, 1:3, 1))
    plot!(plt, x, x -> gompertz_makeham(x, (A, B, C)), alpha=0.01, color=2, label=false)
end
scatter!(plt, x, mx, color=1)
```

Still weird.

## The Heligman-Pollard in Turing.jl

Here we define the Heligman-Pollard model.[^Fnormal]

[^Fnormal]:
    > How come they have $F \sim \mathrm{Normal}(\mu_F, \sigma_F^2)$ since they will take the log of it? Can't let negative values in. I changed it to a Beta distribution.

```@example mortality
@model function heligman_pollard(x, q)
    A ~ Beta()
    B ~ Beta()
    C ~ Beta()
    D ~ Beta()
    E ~ Gamma()
    F ~ Gamma()
    G ~ Beta()
    H ~ Gamma()
    σ² ~ InverseGamma()
    σ = sqrt(σ²)

    for i in eachindex(x)
        m = A ^ ((x[i] + B) ^ C) + D * exp( - E * (log(x[i]) - log(F)) ^ 2) + G * H ^ (x[i]) # / (1 + K * G * H ^ (x[i]) )
        y = m / (1 + m) 
        q[i] ~ Normal(y, σ)
    end
end
```
