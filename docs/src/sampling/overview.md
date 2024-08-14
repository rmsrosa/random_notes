# Sampling methods

A fundamental tool in Statistics is to be able to generate samples of a known distribution. The way to generate samples depends on *how* the distribution is given. We may know its cumulative distribution function (CDF) $F=F(x);$ its probability density function (PDF) $p=p(x);$ the shape of the PDF as a multiple $f(x)$ of it, i.e. $p(x) = f(x)/Z,$ with an unknown and hard-to-compute normalizing constant $Z;$ the "energy" $U(x)$ of the distribution, given by $p(x) = e^{-U(x)}/Z;$ or its Stein score $s(x) = \nabla \ln p(x) = - \nabla\ln U(x).$

We will see here different examples:
1. We may use a *pseudo-random number generator (PRNG)* to generate samples of a *uniform distribution;*
2. The *integral transform method* to generate samples of a scalar distribution when we known inverse $F^{-1}(x)$ of the CDF;
3. Diferent sorts of transformation methods to generate samples of a distribution out of other distributions, such as the *Box-Muller transform* to generate pairs of independent normally distributed random numbers;
4. Markov-Chain-Monte-Carlo methods to generate samples when we only know a multiple $f(x)$ or the energy $U(x)$ of the PDF;
5. Langevin sampling when we only know the Stein score of the distribution.