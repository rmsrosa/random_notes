
# Generating a pair of eyes

This is a toy problem to play with generative models on a very simple and easy-to-train problem.

The plan is to have the model learn to generate a pair of numbers symmetric about the origin, neither too far apart, nor too close. We interpret this as a pair of eyes, since many applications concern image generation. Mathematically speaking, we want the model to learn a specific function of two variables. This is a prototype for a more general problem of finding the probability distribution of some really complicate feature on a gazillion-dimensional space.

We will use a few different approaches for illustration and learning purposes:

1. A classical optimization method of minimizing a cost functional expressing that one number is double the other;
1. A classical optimization method of minimizing a cost functional based on the data, i.e. a classical least squares objetive;
1. A GAN method, i.e. a Generative Adversarial Network with two neural networks competing in a minimax game;
1. A stochastic Langevin method
1. A score-based diffusion model

The data set for training is generated based on a simple probability with $x \sim -y \sim 1$. More precisely, we generate a bunch of samples from a two-dimensional Normal distribution
```math
\mathcal{N}\left( \left(\begin{matrix} \mu_X \\ \mu_Y \end{matrix}\right), \left(\begin{matrix} \sigma_X^2 & \rho\sigma_X\sigma_Y \\ \rho\sigma_X\sigma_Y & \sigma_Y^2 \end{matrix}\right) \right),
```
where we choose
```math
  \mu_X = -\mu_Y = \ell = 1.0
```
and
```math
  \sigma_X = \sigma_Y = \sigma = 0.25, \quad \rho = 0.8,
```
with $\ell$ playing the role of the mean of the half-distance between the eyes, $\sigma$ is the variance for the half-distance, and $\rho$ is the correlation between the distances of the two eyes to the origin, which indicates the horizontal location of the nose, so $\rho$ controls how symmetric the face is, with $\rho = 1$ being perfectly symmetric and $\rho = 0$ not enforcing any symmetry.

```@example geneyes
using Random
using Distributions
using Plots

rng = Xoshiro(123)

ell = 1.0
sigma = 0.1
rho = 0.98
pair_of_eyes_law = MvNormal([ell; -ell], [sigma^2 rho * sigma^2; rho * sigma^2 sigma^2])
```
Now we sample 100 pairs
```@example geneyes
data = rand(rng, pair_of_eyes_law, 100)
nothing # hide
```

```@example geneyes
xx = range(0, 2, length=100)
yy = range(-2, 0, length=100)

pltsurface = plot(title="Surface view of distribution", surface(xx, yy, (x, y) -> pdf(pair_of_eyes_law, [x, y])))
scatter!(pltsurface, data[1, :], data[2, :], pdf(pair_of_eyes_law, data), label="data")

pltheatmap = plot(title="Heatmap of distribution", heatmap(xx, yy, (x, y) -> pdf(pair_of_eyes_law, [x, y])))
scatter!(pltheatmap, data[1, :], data[2, :], label="data")

plot(pltheatmap, pltsurface, size=(800, 400))
```

Now we draw some "eyes"
```@example geneyes
plts = []
for j in 1:4
    pltface = plot(title="Eyes sample $j", xlims=(-2, 2), ylims=(-1, 1), showaxis=false, grid=false, legend=nothing)
    scatter!(pltface, data[1:2, j], 0.7*ones(2), markersize=20)
    plot!(pltface, [0.0, 0.0], [0.4, 0.8])
    push!(plts, pltface)
end
plot(plts..., size=(800, 800))
```

As for the model, image generation usually uses a a convolutional network or a U-Net, but since this is a toy model with only two variables, we just use a dense network with a single hidden layer.
