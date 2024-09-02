# Empirical supremum rejection method

A variation of the von Neumann rejection sampling, proposed by [Caffo, Booth, and Davison (2002)](https://doi.org/10.1093/biomet/89.4.745) (see also [Section 6.3.3 Empirical Supremum Rejection Sampling](https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html#empirical-supremum-rejection-sampling) of [Peng (2022)](https://bookdown.org/rdpeng/advstatcomp/)), relaxes the need to know a good bound for $f(x) \leq M q(x).$ Instead, one uses an adaptive guess that gets updated each time a sample is rejected and the acceptance rate is greater than one.

## Setting

The setting is similar to that of the rejection sampling method. One considers a univariate random variable $X$ with density $p=p(x)$ such that

1. We may not know the density $p(x)$ but we do know a non-negative function $f(x)$ proportional to the density, with some unknown normalizing constant $Z > 0,$ i.e.
   ```math 
        p(x) = \frac{f(x)}{Z}. 
   ```
2. We know how to sample from another random variable $X'$ with known density $q=q(x)$ which bounds a multiple of the function $f(x)$ with an unknown bound, i.e. for some unknown $M>0,$
```math
    f(x) \leq M q(x), \quad \forall x.
```

## The rejection sampling method

Under this setting, we obtain samples of $X$ by sampling from $X'$ and accepting or rejecting the candidate sample according to an acceptance ratio with an adaptive multiplicative factor, updating this factor if the ratio is larger than one, and repeating the process until a candidate is accepted, and for as many samples that we want. More precisely, here are the steps of the method.

1. Start with an educated guess $C$ to hopefully bound $f(x)$ by $Cq(x).$
1. Draw a sample $x'$ of $X',$ which we call a *candidate sample;*
2. Compute the *acceptance ratio* $r(x', C),$ where $r(x, C) = \frac{f(x)}{Cq(x)}.$
3. Draw a sample $u$ from the uniform distribution $\operatorname{Uniform}(0, 1).$
4. Accept/reject/update step:
    1. If $u \leq r(x'),$ accept the sample $x'$ as a sample $x=x'$ of the desired random variable $X;$
    2. Otherwise, if $u > r(x'),$ reject the sample $x',$ check whether $r(x') > 1,$ in which case the multiplicative factor $C$ is updated by $C=f(x')/q(x'),$ and then repeat the process drawing a new candidate and so on, until a candidate sample is accepted.
5. Repeat for as many samples as desired.

## References

1. [Brian S. Caffo, James G. Booth, A. C. Davison (2002), "Empirical supremum rejection sampling," Biometrika, Volume 89, Issue 4, 745–754](https://doi.org/10.1093/biomet/89.4.745)
2. John von Neumann, "Various techniques used in connection with random digits. Monte Carlo methods" *Nat. Bureau Standards,* 12 (1951), 36–38.
3. [Roger D. Peng, "Advanced Statistical Computing", online](https://bookdown.org/rdpeng/advstatcomp/)