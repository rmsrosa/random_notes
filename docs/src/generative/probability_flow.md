# Probability flow ODEs for Itô diffusions

## Aim

The aim is to review the probability flow introduced by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) and generalized by [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) and [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

## Background

[Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) extended the **denoising diffusion probabilistic models (DDPM)** of [Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)](https://dl.acm.org/doi/10.5555/3045118.3045358) and [Ho, Jain, and Abbeel (2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) and the **(multiple denoising) score matching with (annealed) Langevin dynamics (SMLD)** of [Song and Ermon (2019)](https://dl.acm.org/doi/10.5555/3454287.3455354) to the continuous case. This lead to a noising schedule based on a stochastic differential equation of the form
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
for suitable choices of $f=f(t, x)$ and $G=G(t, x)$ (usually with $f(t, x) = -a(t)x$ linear in $x$ and with $G(t, x) = g(t)\mathbf{I}$ diagonal and only time dependent).

[Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) then showed that the probability density $p(t, x)$ of $X_t$ can also be obtained with the (random) ODE
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{1}{2} \nabla_y \cdot ( G(t, Y_t)G(t, Y_t)^{\mathrm{tr}} ) - \frac{1}{2} G(t, Y_t)G(t, Y_t)^{\mathrm{tr}}\nabla_y \log p(t, Y_t).
```

This **probability flow ODE,** as so they termed, was based on the work by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802), who derived this equation in the particular case of a time-independent drift term and a constant diagonal noise factor, i.e. with
```math
    f(t, x) = f(x), \qquad G(t, x) = \sigma \mathbf{I}.
```

Both the SDE for $\{X_t\}_t$ and the random ODE for $\{Y_t\}_t$ have a reverse-time counterpart, which is then used for sampling, provided the Stein score $\nabla \log p(t, x)$ has been properly modeled by a suitable neural network.

In theory, both formulations are equivalent to each other, in the sense of yielding the same probability density $p(t, x).$ In practice, however, their numerical implementations and their Monte Carlo approximations differ considerably. Sampling via the reverse probability flow ODE has some advantages such as the fact that the sample trajectories are smoother and easier to integrate numerically, allowing for higher order schemes with lower computational cost, and that there are less parameters to fiddle with. It is also easier to go back and forth with the ODE. Meanwhile, in some situations and with well chosen parameters, sampling with the reverse SDE seems to somehow generate better sample distributions.

Notice that we denoted the probability flow ODE with $\{Y_t\}_t$ to stress the fact that this is a different process from $\{X_t\}_t$ but with the remarkable fact that they have same probability density function $p(t, x).$

Later, [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html) considered a particular type of SDE and obtained a sort of probability flow SDE with two main characteristics: a desired specific variance schedule and a mixture of the original SDE and the probability flow ODE. In essence, the desired variance is simply a reparametrization of the terms of the equation, while the mixture of the SDE and the ODE is just a split-up of the way the diffusion term is handled. More precisely, for the passage from the SDE to the flow ODE, the whole diffusion term is rewritten as a flux. For the mixture, just part of it is rewritten.

## Probability flow (random) ODEs

The probability flow ODEs are actually ordinary differential equations with random initial conditions, which is a special form of a *Random ODE (RODE).* The main point is that they have the same probability distributions as their original stochastic versions. In what follows, we will build that up in different context.

### With a constant scalar diagonal noise amplitude

This is the original result given by [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802). The SDE is the Itô diffusion with a constant scalar diagonal noise term
```math
    \mathrm{d}X_t = f(X_t)\;\mathrm{d}t + \sigma\;\mathrm{d}W_t,
```
where the unknown $\{X_t\}_t$ is a vector valued process with values in $\mathbb{R}^d,$ $d\in\mathbb{R};$ $\{W_t\}_t$ is a vector valued process in the same event space $\mathbb{R}^d$ with components made of independent Wiener processes; the drift term is a vector field $f:\mathbb{R}^d \rightarrow \mathbb{R}^d;$ and $\sigma > 0$ is a constant noise amplitude factor.

In this case, given a initial probability distribution $p_0$ for the initial random variable $X_0,$ the probability distribution $p(t, x)$ for $X_t$ is the solution of the Fokker-Planck (Kolmogorov forward) equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \frac{\sigma^2}{2}\Delta_x p(t, x).
```

The idea is that the diffusion term can also be expressed as a divergence, namely
```math
    \frac{\sigma^2}{2}\Delta_x p(t, x) = \frac{\sigma^2}{2}\nabla_x \cdot \nabla_x p(t, x) = \nabla_x \cdot \left(\frac{\sigma^2}{2}\nabla_x p(t, x)\right).
```
Another key point is that the gradient above can be written as a multiple of the probability density, with the help of the Stein score function $\nabla_x\log p(t, x).$ Indeed, in the region where $p(t, x) > 0,$
```math
    \nabla_x p(t, x) = p(t, x) \frac{\nabla_x p(t, x)}{p(t, x)} = p(t, x) \nabla_x \log p(t, x).
```
Then, with the understanding that $s\log s$ vanishes for $s = 0$, which extends this function continuously to the region $s\geq 0,$ we can assume this relation holds everywhere. Thus, the Fokker-Planck equation takes the form
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \nabla_x \cdot (\frac{\sigma^2}{2}p(t, x) \nabla_x \log p(t, x)).
```
We now rewrite this equation as
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left(f(x) p(t, x) - \frac{\sigma^2}{2}p(t, x) \nabla_x \log p(t, x) \right) = 0.
```
Defining
```math
    \tilde f(t, x) = f(x) - \frac{\sigma^2}{2} \nabla_x \log p(t, x),
```
the Fokker-Planck equation becomes
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left(\tilde f(x) p(t, x) \right) = 0,
```
which can be viewed as the Liouville equation associated with the evolution of a distribution governed by the ODE, with no diffusion,
```math
    \mathrm{d}Y_t = \tilde f(t, Y_t)\;\mathrm{d}t,
```
i.e. just a Random ODE (more specifically an ODE with random initial data) of the form
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{\sigma^2}{2} \nabla_y \log p(t, Y_t).
```

This equation did not receive any special name in the original work [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802), but got the name **probability flow ODE** in the work [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), which we discuss next.

Before that, we remark that one difficulty with the probability flow ODE is that it requires knownledge of the Stein score function $\nabla_x \log p(t, x)$ of the supposedly unknown distribution. This is circumvented in [Maoutsa, Reich, Opper (2020)](https://doi.org/10.3390/e22080802) by using a gradient log density estimator obtained from samples of the forward diffusion process, i.e. from a maximum likelyhood estimation based on the evolution of the empiral distribution of $X_0.$

On the other hand, [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456) models the score function directly as a (trained) neural network.

### With a nonlinear scalar diagonal noise amplitude

In [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), the authors consider the more general SDE
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
where the diffusion factor is not a scalar diagonal anymore but is a  matrix-valued, time-dependent function $G:I\times \mathbb{R}^d \rightarrow \mathbb{R}^{d\times d},$ and with the drift term also time dependent, $f:I\times \mathbb{R}^d \rightarrow \mathbb{R}^d,$ on an interval of interest $I=[0, T].$

Just for clarity, let us consider first the case in which $G=G(t, x)$ is still diagonal but not a constant anymore, i.e.
```math
    G(t, x) = g(t, x)\mathbf{I},
```
so that
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + g(t, X_t)\;\mathrm{d}W_t,
```
In this case, the Fokker-Planck equation takes the form
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(x) p(t, x)) = \frac{1}{2}\Delta_x (g(t, x)^2 p(t, x)).
```
As before, the diffusion term can be written as
```math
    \begin{align*}
        \frac{1}{2}\Delta_x (g(t, x)^2 p(t, x)) & = \frac{1}{2}\nabla_x \cdot \nabla_x (g(t, x)^2 p(t, x)) \\
        & = \frac{1}{2}\nabla_x \cdot \left( \nabla_x g(t, x)^2 p(t, x) + g(t, x)^2 \nabla_x p(t, x) \right) \\
        & = \frac{1}{2}\nabla_x \cdot \left( \nabla_x g(t, x)^2 p(t, x) + g(t, x)^2 p(t, x) \nabla_x \log p(t, x) \right)
    \end{align*}
```
Thus, the Fokker-Planck equation becomes
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left( f(x) p(t, x) - \frac{1}{2} \nabla_x g(t, x)^2 p(t, x) - g(t, x)^2 p(t, x) \nabla_x \log p(t, x) \right) = 0,
```
which can also be viewed as the Fokker-Planck equation for the SDE with no diffusion
```math
    \mathrm{d}Y_t = \tilde f(t, Y_t)\;\mathrm{d}t,
```
with
```math
    \tilde f(t, x) = f(t, x) - \frac{1}{2} \nabla_x g(t, x)^2 - g(t, x)^2\nabla_x \log p(t, x),
```
which is actually a random ODE (more specifically an ODE with random initial condition),
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{1}{2} \nabla_y g(t, Y_t)^2 - g(t, Y_t)^2\nabla_y \log p(t, Y_t),
```
with the equation for $p(t, x)$ being the associated Liouville equation.

### With arbitrary drift and diffusion

Now we address the more general case considered in [Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2020)](https://arxiv.org/abs/2011.13456), with an SDE of the form
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
where the diffusion factor is now a matrix-valued, time-dependent function $G:I\times \mathbb{R}^d \rightarrow \mathbb{R}^{d\times d}.$

In this case, the Fokker-Planck equation takes the form
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(t, x) p(t, x)) = \frac{1}{2}\nabla_x^2 : \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x)\right).
```
Notice that the term within the parentheses, on the right hand side, is a tensor field which at each point $(t, x)$ yields a certain matrix $A(t, x) = (A_{ij}(t, x))_{i,j=1}^d.$ The differential operation $\nabla_x^2 :$ acting on such a tensor field is given by
```math
    \nabla_x^2 : A(t, x) = \sum_{i=1}^d\sum_{j=1}^d \frac{\partial^2}{\partial x_i\partial x_j} A_{ij}(t, x).
```

We may write everything in coordinates, starting with
```math
    X_t = (X_t^i)_{i=1}^d, \quad W_t = (W_t^i)_{i=1}^d, \quad f(t, x) = (f_i(t, x))_{i=1}^d, \quad G(t, X_t) = (G_{ij}(t, X_t))_{i, j=1}^d,
``` 
so that the SDE reads
```math
    \mathrm{d}X_t^i = f_i(t, X_t^1, \ldots, X_t^d)\;\mathrm{d}t + \sum_{j=1}^d G_{ij}(t, X_t^1, \ldots, X_t^d)\;\mathrm{d}W_t^j,
```
and the Fokker-Planck equation reads
```math
    \frac{\partial p}{\partial t} + \sum_{i=1}^d \frac{\partial}{\partial x_i} (f(t, x) p(t, x)) = \frac{1}{2}\sum_{i=1}^d \sum_{j=1}^d \frac{\partial^2}{\partial x_i \partial x_j} (G_{ik}(t, x)G_{jk}(t, x) p(t, x)).
```
Writing the divergence of a tensor field $A(t, x) = (A_{ij}(x))_{i,j=1}^d$ as the vector
```math
    \nabla_x \cdot A(t, x) = \left( \nabla_x \cdot A_{i\cdot}(t, x)\right)_{i=1}^d = \left( \sum_{j=1}^d \frac{\partial}{\partial x_j} A_{ij}(t, x)\right)_{i=1}^d,
```
we can write the Fokker-Planck equation in divergence form,
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(t, x) p(t, x)) = \frac{1}{2}\nabla_x \cdot \left(\nabla_x \cdot (G(t, x)G(t, x)^{\mathrm{tr}} p(t, x))\right).
```
As before, the diffusion term can be written as
```math
    \begin{align*}
        \frac{1}{2}\nabla_x \cdot & \left( \nabla_x \cdot (G(t, x)G(t, x)^{\mathrm{tr}} p(t, x)) \right) \\
        & = \frac{1}{2}\nabla_x \cdot \bigg( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) + G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x p(t, x) \bigg) \\
        & = \frac{1}{2}\nabla_x \cdot \bigg( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) + G(t, x)G(t, x)^{\mathrm{tr}}p(t, x)\nabla_x \log p(t, x) \bigg).
    \end{align*}
```
With that, the Fokker-Planck equation reads
```math
    \begin{align*}
        \frac{\partial p}{\partial t} + & \nabla_x \cdot (f(t, x) p(t, x)) \\
        & = \frac{1}{2}\nabla_x \cdot \left( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) + G(t, x)G(t, x)^{\mathrm{tr}}p(t, x)\nabla_x \log p(t, x) \right).
    \end{align*}
```

Rearranging it, we obtain the following equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left( \left( f(t, x) - \frac{1}{2} \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) - \frac{1}{2} G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x) \right) p(t, x) \right) = 0,
```
which can be viewed as the Liouville equation for the random ODE
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{1}{2} \nabla_y \cdot ( G(t, Y_t)G(t, Y_t)^{\mathrm{tr}} ) - \frac{1}{2} G(t, Y_t)G(t, Y_t)^{\mathrm{tr}}\nabla_y \log p(t, Y_t).
```
This is the **probability flow (random) ODE** of [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html) (except for the symbol $\{Y_t\}_t$ instead of $\{X_t\}_t$).

## Splitted-up probability flow SDE

The change from the Fokker-Planck equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot (f(t, x) p(t, x)) = \frac{1}{2}\nabla_x^2 : \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right).
```
for the SDE
```math
    \mathrm{d}X_t = f(t, X_t)\;\mathrm{d}t + G(t, X_t)\;\mathrm{d}W_t,
```
to the Liouville equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \left( \left( f(t, x) - \frac{1}{2} \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) - \frac{1}{2} G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x) \right) p(t, x) \right) = 0,
```
for the random ODE
```math
    \frac{\mathrm{d}Y_t}{\mathrm{d}t} = f(t, Y_t) - \frac{1}{2} \nabla_y \cdot ( G(t, Y_t)G(t, Y_t)^{\mathrm{tr}} ) - \frac{1}{2} G(t, Y_t)G(t, Y_t)^{\mathrm{tr}}\nabla_y \log p(t, Y_t),
```
boils down to expressing the diffusion term completely as a flux term:
```math
    \begin{align*}
        \frac{1}{2}\nabla_x^2 : & \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right) \\
        & = \frac{1}{2}\nabla_x \cdot \left( \left( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) + G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x) \right) p(t, x) \right).
    \end{align*}
```
As discussed in the Introduction, both formulations seem to have their advantages. So one idea is to split up the diffusion term and handle one part as the ODE flow and leave the other part as the SDE diffusion, in an attempt to leverage the advantages of both formulations. This is what we do next.

### For a general Itô diffusion

We may introduce a weight parameter, say $\theta(t),$ which can even be time dependent, and split up the diffusion term of the Fokker-Planck equation into
```math
   \begin{align*}
        \frac{1}{2}\nabla_x^2 : & \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right) \\
        & = \frac{1 - \theta(t)}{2}\nabla_x^2 : \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right) + \frac{\theta(t)}{2}\nabla_x^2 : \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right).
    \end{align*}
```
Rewriting only the first term as a flux we obtain
```math
    \begin{align*}
        \frac{1}{2}\nabla_x^2 : & \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right) \\
        & = \frac{1 - \theta(t)}{2}\nabla_x \cdot \left( \left( \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) + G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x) \right) p(t, x) \right) \\
        & \qquad +  \frac{\theta(t)}{2}\nabla_x^2 : \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right).
    \end{align*}
```
In this way, the Fokker-Planck equation becomes
```math
    \begin{align*}
        \frac{\partial p}{\partial t} + \nabla_x \cdot \bigg( \bigg( & f(t, x) - \frac{1-\theta(t)}{2} \nabla_x \cdot ( G(t, x)G(t, x)^{\mathrm{tr}} ) \\
        & \qquad \qquad - \frac{1-\theta(t)}{2} G(t, x)G(t, x)^{\mathrm{tr}}\nabla_x \log p(t, x) \bigg) p(t, x) \bigg) \\
        & \qquad \qquad \qquad \qquad = \frac{\theta(t)}{2}\nabla_x^2 : \left( (G(t, x)G(t, x)^{\mathrm{tr}}) p(t, x) \right).
    \end{align*}
```

The associated **splitted-up probability flow SDE** reads
```math
    \begin{align*}
        \mathrm{d}Y_t = \bigg( & f(t, Y_t) - \frac{1- \theta(t)}{2} \nabla_y \cdot ( G(t, Y_t)G(t, Y_t)^{\mathrm{tr}} ) \\
        & \qquad \qquad - \frac{1 - \theta(t)}{2} G(t, Y_t)G(t, Y_t)^{\mathrm{tr}}\nabla_y \log p(t, Y_t)\bigg)\;\mathrm{d}t + \sqrt{\theta(t)} G(t, Y_t)\;\mathrm{d}W_t.
    \end{align*}
```

As mentioned before, all these formulations are theoretically equivalent. But their practical applications differ.

### For an SDE with scalar time-dependent diagonal noise

In the case that
```math
    G(t, x) = g(t)\mathbf{I},
```
the splitted-up probability flow SDE reduces to
```math
    \begin{align*}
        \mathrm{d}Y_t = \bigg( & f(t, Y_t) - \frac{1 - \theta(t)}{2} g(t)^2 \nabla_y \log p(t, Y_t)\bigg)\;\mathrm{d}t + \sqrt{\theta(t)} g(t)\;\mathrm{d}W_t,
    \end{align*}
```
with the Fokker-Planck equation
```math
    \frac{\partial p}{\partial t} + \nabla_x \cdot \bigg( \bigg( f(t, x) - \frac{1-\theta(t)}{2} g(t)\nabla_x \log p(t, x) \bigg) p(t, x) \bigg) = \frac{\theta(t)g(t)^2}{2}\Delta_x p(t, x).
```

### Connection with the Karras et al probability flow SDE

If we set
```math
    f(t, x) = 0, \qquad g(t) = \sqrt{2\dot\sigma(t) \sigma(t)}
```
and choose
```math
    \theta(t) = \frac{2\beta(t)\sigma(t)^2}{g(t)^2},
```
we get
```math
    \begin{align*}
        - \frac{1 - \theta(t)}{2} g(t)^2 & = \frac{1}{2}(\theta(t) - 1) g(t)^2 = \frac{1}{2}\left( \frac{2\beta(t)\sigma(t)^2}{g(t)^2} - 1\right)g(t)^2 = \frac{1}{2}\left(2\beta(t)\sigma(t)^2 - g(t)^2\right) \\
        & = \frac{1}{2}\left( 2\beta(t)\sigma(t)^2 - 2\dot\sigma(t) \sigma(t)\right) = \beta(t)\sigma(t)^2 - \dot\sigma(t) \sigma(t)
    \end{align*}
```
and 
```math
    \sqrt{\theta(t)} g(t) = \sqrt{2\beta(t)} \sigma(t)
```
Thus, we transform the splitted-up probability flow SDE into the probability flow SDE of [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html),
```math
    \begin{align*}
        \mathrm{d}Y_t = \left( -\dot\sigma(t)\sigma(t) + \beta(t)\sigma(t)^2 \right) \nabla_y \log p(t, Y_t)\;\mathrm{d}t + \sqrt{2\beta(t)} \sigma(t)\;\mathrm{d}W_t,
    \end{align*}
```
having the same distribution as the SDE
```math
    \mathrm{d}X_t = \sqrt{2\dot\sigma(t)\sigma(t)}\;\mathrm{d}W_t,
```
which is associated with the Fokker-Planck equation
```math
    \frac{\partial p}{\partial t}  = \dot\sigma(t)\sigma(t)\Delta_x p(t, x).
```

Notice that now, instead of the free parameter $\theta(t)$, we have the free parameter $\beta(t)$ of [Karras, Aittala, Aila, Laine (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html).

The motivation for writing it in this way, with $\dot\sigma \sigma$ as the diffusion coefficient, is that the heat kernel becomes
```math
    G(\sigma(t)) = \frac{1}{(2\pi \sigma(t)^2)^{d/2}} e^{-\frac{1}{2}\frac{x^2}{\sigma(t)^2}},
```
which is the probability density of the Gaussian $\mathcal{N}(0, \sigma(t)^2\mathbf{I}).$ The distribution $p(t, x)$ is given by the convolution
```math
    p(t, \cdot) = G(\sigma(t)) \star p_0,
```
where $p_0=p_0(x)$ is the initial probability density of the distribution we are attempting to model. This is just a reparametrization of the process directly in terms of a desired variance $\sigma(t)^2.$

## References

1. [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", ICML'15: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, 2256-2265](https://dl.acm.org/doi/10.5555/3045118.3045358)
1. [Y. Song and S. Ermon (2019), "Generative modeling by estimating gradients of the data distribution", NIPS'19: Proceedings of the 33rd International Conference on Neural Information Processing Systems, no. 1067, 11918-11930](https://dl.acm.org/doi/10.5555/3454287.3455354)
1. [J. Ho, A. Jain, P. Abbeel (2020), "Denoising diffusion probabilistic models", in Advances in Neural Information Processing Systems 33, NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
1. [D. Maoutsa, S. Reich, M. Opper (2020), "Interacting particle solutions of Fokker-Planck equations through gradient-log-density estimation", Entropy, 22(8), 802, DOI: 10.3390/e22080802](https://doi.org/10.3390/e22080802)
1. [Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2020), "Score-based generative modeling through stochastic differential equations", arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
1. [T. Karras, M. Aittala, T. Aila, S. Laine (2022), Elucidating the design space of diffusion-based generative models, Advances in Neural Information Processing Systems 35 (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)