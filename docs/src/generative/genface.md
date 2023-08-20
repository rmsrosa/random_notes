# Generating a face

After learning the fundamentals of the theory, we now attempt to generate a face, or rather a sketch of a face, for a more fun model. Training a more realistic feature with a high-resolution image is extremely demaning computationally and expensive, so we settle with that.

We train a low dimensional image parametrized by a few features, resembling a scalable vetor graphics (SVG) encoding.
```math
  \mathbf{x} = (x_1, y_1, l_1, a_1, b_1, x_2, y_2, l_2, a_2, b_2, x_3, y_3, l_3, x_4, y_4, l_4, w_4, a_4, b_4, c_4, d_4) \in \mathbb{R}^{21},
```
where
1. The left eye is modeled as the combination of the graphs of the functions $y(x) = y_1 + a_1(x - x_1)^2 - a_1l_1^2$ and $y(x) = y_1 + b_1(x - x_1)^2 + b_1l_1^2$, for $x_1 - l_1 \leq x \leq x_1 + l_1$, which is a portions of parabolas centered at $(x_1, y_1)$, with half-width $l_1$ and concavities $a_1$ and $b_1$, respectively;
1. The right eye is modeled as the combination of the graphs of the functions $y(x) = y_2 + a_2(x - x_2)^2 - a_2l_2^2$ and $y(x) = y_2 + b_2(x - x_2)^2 - b_2l_2^2$, for $x_2 - l_2 \leq x \leq x_2 + l_2$, which are portions of parabolas centered at $(x_2, y_2)$, with half-width $l_2$ and concavities $a_2$ and $b_2$, respectively;
1. The nose is modeled as the triangle with vertices $(x_3 - w_3/2, y_3)$, $(x_3, y_3 + l_3)$, and $(x_3 + w_3/2, y_3)$, where $l_3$ is the height and $w_3$ is the half-width of the nose;
1. The mouth is modeled as the graph of a quartic $y(x) = y_4 + a_4(x - x_4) + b_4(x - x_4)^2 + c_4(x - x_4)^3 + d_4(x - x_4)^4.$

Loading the necessary packages and setting a random seed:
```@example genface
using Random
using Distributions
using Plots
using ComponentArrays

rng = Xoshiro(123)
```

Parameters for an example face:
```@example genface
(x_1, y_1, w_1, a_1, b_1) = (-1.0, 1.0, 0.3, 0.3, -0.5)
(x_2, y_2, w_2, a_2, b_2) = (1.0, 1.0, 0.3, 0.3, -0.5)
(x_3, y_3, w_3, h_3) = (0.0, 0.5, 0.3, 0.5)
(x_4, y_4, w_4, a_4, b_4, c_4, d_4) = (0.0, 0.1, 1.0, 0.0, 0.1, 0.0, 0.0)
nothing
```

We actually represent the state vector as a `ComponentVector`, com [jonniedie/ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl):
```@example genface
u = ComponentVector(
    lefteye = [x_1, y_1, w_1, a_1, b_1],
    righteye = [x_2, y_2, w_2, a_2, b_2],
    nose = [x_3, y_3, w_3, h_3],
    mouth = [x_4, y_4, w_4, a_4, b_4, c_4, d_4]
)
```

We define a few helper functions for drawing a face:
```@example genface
function eye(x, p, pos)
    x_0, y_0, w_0, a_0, b_0 = p
    c_0 = pos == :up ? a_0 : pos == :down ? b_0 : zero(a_0)
    y = y_0 + c_0 * (x - x_0)^2 - c_0 * w_0^2
    return y
end

function nosetop(x, p)
    x_0, y_0, w_0, h_0 = p
    y = y_0 + min(h_0 * (x - x_0 + w_0) / w_0, h_0 * (- x + x_0 + w_0) / w_0)
    return y
end

function mouth(x, p)
    x_0, y_0, w_0, a_0, b_0, c_0, d_0 = p
    y = y_0 + a_0 * (x - x_0) + b_0 * (x - x_0)^2 + c_0 * (x - x_0)^3 + d_0 * (x - x_0)^4
    return y
end

function baserange(p; length)
    x_0 = p[1]
    w_0 = p[3]
    xx = range(x_0 - w_0, x_0 + w_0, length=length)
    return xx
end

function drawface(u; kwargs...)

    c = :color in keys(kwargs) ? kwargs[:color] : :auto

    plt = plot(xlims=(-2, 2), ylims=(-0.2, 1.2), legend=false, grid=false, showaxis=false)

    xx = baserange(u.lefteye, length=20)
    plot!(plt, xx, x -> eye(x, u.lefteye, :up), color=c)
    plot!(plt, xx, x -> eye(x, u.lefteye, :down), color=c)

    xx = baserange(u.righteye, length=20)
    plot!(plt, xx, x -> eye(x, u.righteye, :up), color=c)
    plot!(plt, xx, x -> eye(x, u.righteye, :down), color=c)

    xx = baserange(u.nose, length=40)
    plot!(plt, xx, x -> nosetop(x, u.nose), color=c)
    x_3, y_3, w_3 = view(u.nose, 1:3)
    plot!(plt, [x_3 - w_3, x_3 + w_3], [y_3, y_3], color=c)

    xx = baserange(u.mouth, length=80)
    plot!(plt, xx, x -> mouth(x, u.mouth), color=c)

    return plt
end
```

```@example genface
plt = drawface(u, color=:black)
plot!(plt, title="Smiley face")
```

For testing, we generate faces with
```@example genface
function genface(rng)
    lefteye = [-1.0, 1.0, 0.3, 0.3, -0.3]
    righteye = [1.0, 1.0, 0.3, 0.3, -0.3]
    nose = [0.0, 0.5, 0.3, 0.5]
    mouth = [0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0]
    sigma = 0.2
    u = ComponentVector(
        lefteye = rand(rng, MvNormal(lefteye, sigma^2)),
        righteye = rand(rng, MvNormal(righteye, sigma^2)),
        nose = rand(rng, MvNormal(nose, sigma^2)),
        mouth = rand(rng, MvNormal(mouth, sigma^2))
    )
    return u
end
```

With that, we generate a sample of faces for the model to learn upon.
```@example genface
faces = [
    genface(rng) for _ in 1:200
]
nothing
```

Let us visualize how the first faces look like
```@example genface
plt = [
    drawface(face) for face in view(faces, 1:20)
]
plot(plt...)
```
