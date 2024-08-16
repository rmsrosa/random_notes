# Overview of sampling methods

A fundamental tool in Statistics is to be able to generate samples of a known distribution. The way to generate samples depends on *how* the distribution is given. We will see here different examples:
1. We may use a *pseudo-random number generator (PRNG)* to generate samples of a *uniform distribution;*
2. The *integral transform method* to generate samples of a scalar distribution when we known the inverse $F^{-1}(x)$ of the CDF;
3. Different sorts of transformation methods to generate samples of a distribution out of other distributions, such as the *Box-Muller transform* to generate pairs of independent normally distributed random numbers;
4. The *Rejection sampling* method when we only know a multiple $f(x)$ of the PDF, i.e. $p(x) = f(x)/Z,$ for some unknown $Z$, and we known how to sample from another distribution with PDF $q=q(x)$ such that a bound of the form $f(x) \leq M q(x),$ for a given $M,$ is available;
5. *Markov-Chain-Monte-Carlo (MCMC)* methods to generate samples when we only know a multiple $f(x)$ or the energy $U(x)$ of the PDF, i.e. $p(x) = f(x)/Z$ or $p(x) = e^{-U(x)}/Z,$ for an unknown or hard-to-compute normalizing constant $Z,$ such as the *Gibbs sampler,* the *Metropolis algorithm,* the *Metropolis-Hastings method,* and the *Hamiltonian Monte-Carlo (HMC)* method;
6. *Langevin sampling* when we only know the Stein score $s(x) = \nabla\ln p(x)$ of the distribution.