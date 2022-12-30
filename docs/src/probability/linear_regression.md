# Linear Regression in several ways

The plan is to do a simple linear regression in julia, in several different ways. We'll use plain least squares [`Base.:\`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.:\\-Tuple{AbstractMatrix,%20AbstractVecOrMat}), the genereral linear model package [JuliaStats/GLM.jl](https://juliastats.org/GLM.jl/stable/) and the probabilistic programming package [Turing.jl](https://turing.ml/stable/).

```@example linear_reg
using Distributions, GLM, Turing, StatsPlots
```

## The test data

This is a simple test set. We just generate a synthetic sample with a bunch of points approximating a straight line. We actually create two tests, an unperturbed straight line and a perturbed one.

```@example linear_reg
num_points = 20
xx = range(0.0, 1.0, length=num_points)

intercept = 1.0
slope = 2.0
ε = 0.1

yy = intercept .+ slope * xx

yy_perturbed = yy .+ ε * randn(num_points)

plt = plot(title="Synthetic data", titlefont=10, ylims=(0.0, 1.1 * (intercept + slope)))
scatter!(plt, xx, yy, label="unperturbed sample")
scatter!(plt, xx, yy_perturbed, label="perturbed sample")
```

## Straightforward least squares

The least square solution $\hat x$ to a linear problem $Ax = b$ is the vector $\hat x$ that minimizes the sum of the squares of the residuals $b - Ax$, i.e.
```math
    \hat x = \argmin_{x} \|Ax - b\|^2
```
The solution is obtained by solving the *normal equation*
```math
    (A^t A)\hat x = A^t b
```

In julia, $\hat x$ can be found by using the function [`Base.:\`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.:\\-Tuple{AbstractMatrix,%20AbstractVecOrMat}), which is actually a polyalgorithm, meaning it uses different algorithms depending on the shape and type of $A$. 

1. When $A$ is an invertible square matrix, then $\hat x = A^{-1}b$ is the unique solution of $Ax = b$.
2. When $A$ has more rows than columns and the columns are linearly independent, then $\hat x = (A^tA)^{-1}A^tb$ is the unique least square approximation solution of the overdetermined system $Ax = b$.
3. When $A$ has more columns than rows and the rows are linearly independent, then $\hat x = A^t(AA^t)^{-1}b$ is the unique least norm solution of the underdetermined system $Ax = b$.
4. In all other cases, attempting to solve $A\b$ throws an error.

First we build the Vandermonde matrix.

```@example linear_reg
A = [ones(length(xx)) xx]
size(A)
```

Now we solve the least square problem with the unperturbed data, solving it explicitly with $\hat x = (A^tA)^{-1}A^tb$ and via $A \setminus b$, and checking both against the original slope and intercept.

```@example linear_reg
betahat = inv(transpose(A) * A) * transpose(A) * yy

betahat ≈ A \ yy ≈ [intercept, slope]
```

Now we solve it with the perturbed data

```@example linear_reg
betahat = inv(transpose(A) * A) * transpose(A) * yy_perturbed

betahat ≈ A \ yy_perturbed
```

```@example linear_reg
intercepthat, slopehat = betahat
```

```@example linear_reg
yy_hat = intercepthat .+ slopehat * xx 
```

```@example linear_reg
plt = plot(title="Synthetic data and least square fit", titlefont=10, ylims=(0.0, 1.1 * (intercept + slope)))
scatter!(plt, xx, yy, label="unperturbed sample")
scatter!(plt, xx, yy_perturbed, label="perturbed sample")
plot!(plt, xx, yy, label="unperturbed line")
plot!(plt, xx, yy_hat, label="fitted line")
```

## Bayesian linear regression with Turing.jl

```@example linear_reg
@model function regfit(x, y)
    σ² ~ InverseGamma()
    σ = sqrt(σ²)
    c ~ Normal(0.0, 10.0)
    m ~ Normal(0.0, 10.0)

    for i in eachindex(x)
        v = c + m * x[i]
        y[i] ~ Normal(v, σ)
    end
end
```

```@example linear_reg
n = 20
c = 2.0
m = 1.5
x = (1:n) ./ n
y = c .+ m * (x .+ 0.02 * randn(n))
scatter(x, y, legend=:topleft)
```

Let's use the Hamiltonian Monte Carlo method to infer the parameters of the model.

```@example linear_reg
model = regfit(x, y)

chain = sample(model, HMC(0.05, 10), 4_000)

plot(chain)
```
