# Modeling mortality tables

```@example mortality
using Distributions, Turing, StatsPlots
```

How come they have $F \sim Normal(\mu_F, \sigma_F^2)$ since they will take the log of it? Can't let negative values in. I changed it to a Beta distribution.

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
        η = A ^ ((x[i] + B) ^ C) + D * exp( - E * (log(x[i]) - log(F)) ^ 2) + G * H ^ (x[i]) # / (1 + K * G * H ^ (x[i]) )

        y = η / (1 + η) 
        q[i] ~ Normal(y, σ)
    end
end
```

```@example mortality
@model function gompertz_makeham(x, q)
    A ~ Beta()
    B ~ Beta()
    C ~ Beta()
    σ² ~ InverseGamma()
    σ = sqrt(σ²)

    for i in eachindex(x)
        η = A + B * C ^ (x[i])

        y = η / (1 + η) 
        q[i] ~ Normal(y, σ)
    end
end
```

```@example mortality
x = collect(0:100)
q = [
    0.003522
    0.000213
    0.000114
    0.000093
    0.000066
    0.000077
    0.000068
    0.000053
    0.000053
    0.000055
    0.000066
    0.000054
    0.000054
    0.000087
    0.000093
    0.000107
    0.000125
    0.000154
    0.000207
    0.000208
    0.000182
    0.000198
    0.000234
    0.000197
    0.000213
    0.000253
    0.000256
    0.000294
    0.000300
    0.000316
    0.000377
    0.000376
    0.000439
    0.000478
    0.000559
    0.000569
    0.000636
    0.000731
    0.000777
    0.000804
    0.000860
    0.000946
    0.001064
    0.001166
    0.001306
    0.001435
    0.001557
    0.001683
    0.001915
    0.002002
    0.002154
    0.002394
    0.002514
    0.002691
    0.002858
    0.003197
    0.003542
    0.003823
    0.004260
    0.004538
    0.005112
    0.005484
    0.006272
    0.006691
    0.007167
    0.007879
    0.008552
    0.009269
    0.010364
    0.011147
    0.012587
    0.013296
    0.015106
    0.016883
    0.019224
    0.021169
    0.023871
    0.027332
    0.030672
    0.035113
    0.038899
    0.044112
    0.049270
    0.056318
    0.064188
    0.072903
    0.083274
    0.094737
    0.106725
    0.120461
    0.135122
    0.152146
    0.169843
    0.188654
    0.206166
    0.229104
    0.252690
    0.276870
    0.299265
    0.320210
    0.3499
]
scatter(x, q, yscale=:log10, legend=:topleft)
```

```@example mortality
model = heligman_pollard(x, q)
model = gompertz_makeham(x, q)
```

```@example mortality
chain = sample(model, HMC(0.05, 10), 40_000)
# chain = sample(model, NUTS(0.65), 20_000)
```

```@example mortality
plot(chain)
```

```@example mortality
plot(x, mean(chain, :A) .+ mean(chain, :B) * mean(chain, :C) .^ x, yscale=:log10)
```
