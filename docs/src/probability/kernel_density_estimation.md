# Kernel density estimation

Let us say we have a sample $\{x_n\}_n$ associated with a random variable $X$. Which statistical informations one can draw from it?

Certainly we can compute the *sample* mean, and the *sample* standard deviation, directly from the data, and so on. We can also draw its histogram, to have a more visual information of the underlying distribution. That histogram resembles a PDF, so one natural question is how well can we approximate the PDF from the data.

There are *parametric* ways to do that, which means we can assume a *parametrized model*, say a Beta distribution $B(\alpha, \beta)$ with shape parameters $\alpha$ and $\beta$, and *fit* the model to the data, say using maximum likelyhood estimation, and use the pdf of the fitted model. That's all good. Depending on the random variable, your model can become quite complex, though.

There are also *nonparametric* ways of obtaining an approximate PDF for the distribution, like the *kernel density estimation.*
