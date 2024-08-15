# Rejection sampling method

In the rejection sampling method, one considers a probability distribution $\mathbb{P}$ with density $p=p(x)$ under the following assumptions
1. We know a non-negative function $f(x)$ proportional do the density, with some (unknown) normalizing constant $Z > 0,$ i.e.
   ```math 
        p(x) = \frac{f(x)}{Z}. 
   ```
2. We know how to sample from another distribution $\mathbb{P}'$ with known density $q=q(x)$ which bounds a multiple of the function $f(x)$ with a known bound, i.e. for some known $M>0,$
```math
    f(x) \leq M q(x).
```

Under this setting, we obtain samples of $\mathbb{P}$ by sampling from $\mathbb{P}'$ and accepting or rejecting the candidate sample according to a specific criteria, and we repeat the process until a candidate is accepted. More precisely, here is the method.

1. Draw a sample $x'$ of $\mathbb{P}',$ which we call a *candidate sample;*
2. Compute the *acceptance ratio* $r(x'),$ where
```math
    r(x) = \frac{f(x)}{Mq(x)}.
```
3. Draw a sample $u$ from the uniform distribution $\operatorname{Uniform}(0, 1).$
4. If $u \leq r(x),$ accept the sample $x'$ as a sample of the desired distribution $\mathbb{P},$ otherwise, if $u > r(x),$ reject the sample $x'$ and repeat the process drawing a new candidate and so on.

