## Bayesian inference

In many situations, we expect some random variable to follow a given distribution but it is not certain what parameters actually define the distribution. For instance, we may have a coin that might be biased but we are unsure about how biased it is. Or we may expect some feature of a population to follow a normal distribution but it is not clear what are its mean and/or standard deviation. In those cases, it is useful to treat those parameters as random variables themselves, leading to what is known as a *compound distribution.*

Then, given a certain feature and a model, we may attempt to fit the model to the available data, which is refereed to a *statistical inference* problem. We are particularly interested here in *Bayesian inference,* which amounts to using Bayes' formula in the inference process, by *updating a prior* knowledge of the distribution according to the *evidence* given by the data.

In loose terms, suppose a compound distribution model has a parameter $\theta$, and we initially believe in a certain *prior* distribution $p(\theta)$ for this parameter. We then observe some data, or evidence, $E$ and *update* our belief according to Bayes' formula,

```math
    p(\theta | E) = \frac{p(E | \theta) p(\theta)}{p(E)}.
```

After updating, the *posteriors* may indicate better the most likely values for the parameters.

The [Bayesian inference on defect item](http://localhost:8000/probability/bayes.html#Bayesian-inference-on-defect-item) is an example where each $D_i$, $i = 4, 5, 6$, represent a probable *model* for the chosen dice, with the posteriors $p(D_i|E)$ revealing the most likely dice picked in the beginning of the problem. This is an example of a finite number of choices for the parameter. More often, the parameter belongs to a continuum, in either a finite- or infinite-dimensional space.

## A biased coin

Let's suppose we are told a coin is biased but we don't know its bias and don't even know whether it is biased towards heads or tails. We let $X$ be the random variable with the result of tossing the coin, which follows a Bernoulli distribution with say probability $\theta$ of assuming the value 1, representing heads, and $1-\theta$ of assuming the value 0, representing tails. Thus,

```math
    X \sim \mathrm{Bernoulli}(\theta).
```

The bias $\theta$ may assume any value between 0 and 1, so we consider it as a random variable denoted $\Theta$. We could assume an *uninformative prior,* with $\Theta$ uniformly distributed between 0 and 1, or an *informative prior,* assuming, more likely, just a slight bias, near 1/2. In the first case, we take $\Theta \sim \mathrm{Beta}(1, 1) = \mathrm{Uniform}(0, 1)$, while in the second case, we may assume $\Theta$ to be distributed like $\mathrm{Beta}(n, m)$, with $n, m \gg 1$. In any case, we suppose

```math
    \Theta \sim \mathrm{Beta}(\alpha, \beta),
```

for $\alpha, \beta > 0$. Recall the probability distribution function for $\Theta$ is

```math
    f_\Theta(\theta) = f_\Theta(\theta; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha - 1}(1 - \theta)^{\beta - 1},
```

where $\Gamma = \Gamma(z)$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

Now, suppose we toss the coin a number times and it lands heads $k$ times and tails $m$ times. So this is our evidence $E$. The random variable $X$ is discrete, while $\Theta$ is continuous. The posterior becomes

```math
    f_\Theta(\theta | E) = \frac{p(E | \theta) f_\Theta(\theta)}{p(E)}.
```

Since $E$ is heads $k$ times and tails $m$ times, we have $p(E | \theta) = \theta^k (1 - \theta)^{m}$. Computing $(E)$ is not a trivial task but we will see we do not need to compute it in this case. Indeed, up to a constant, we have

```math
    p(\theta | E) \propto \theta^k (1 - \theta)^{n-k} \theta^{\alpha - 1}(1 - \theta)^{\beta - 1} = \theta^{k + \alpha - 1}(1 - \theta)^{m + \beta - 1} \sim \mathrm{Beta}(\alpha + k, \beta + m).
```

Hence, updating the prior in this case simply amounts to adding the number of heads and the number of tails to the parameters of the beta distribution.

## Conjugate distributions

This property that multipling a Bernoulli distribution by a Beta prior yields a Beta posterior is an example of *conjugate distributions.* Conjugate distributions greatly facilitate the updating process in Bayesian statistics. There are a number of other conjugate prior distributions, but, in general, updating a prior is a much harder process and requires some fancy computational techniques.
