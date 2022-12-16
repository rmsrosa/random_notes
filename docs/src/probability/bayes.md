## Bayes theorem and applications

In this section, we state Bayes' theorem and discuss some of its applications.

### Bayes' Theorem

**Bayes' Theorem** concerns the probability of a given event, conditioned to another event, and can be stated as follows.

!!! note "Bayes' Theorem"
    Let $p$ be a probability distribution on a given sample space and let $A$ and $B$ be two events with $p(B) > 0$. Then
    ```math
        p(A|B) = \frac{p(B|A)p(A)}{p(B)}.
    ```

In other words, Bayes' Theorem says that the *posterior* conditional probability $p(A|B)$ of an event $A$, given the occurrence of another event $B$, equals the *likelihood* $p(B|A)$ of the second event $B$ given the first event $A$ times the *prior* $p(A)$ divided by the *marginal* $p(B)$. The *prior* here refers to the probability $p(A)$ of $A$ *before* observing the event $B$, while the *posterior* $p(A|B)$ refers to the probability of $A$ *after* the observation of the event $B$.

Bayes' theorem has many useful consequences (see e.g. ), but first let us sketch its proof.

!!! tip "Proof of Bayes's Theorem"
    When $P(A) = 0$, then $P(A|B) = 0$ and the result is trivial. When $p(A) > 0$, the result can be obtained from the conditional probability relations
    ```math
        p(A|B) = \frac{p(A\cap B)}{p(B)}, \qquad p(B|A) = \frac{p(B\cap A)}{p(A)},
    ```
    which imply
    ```math
        p(A|B)p(B) = p(A\cap B) = p(B\cap A) = p(B|A)p(A).
    ```
    Solving for $p(A|B)$ yields the desired result.

Very often, we are not given $p(B)$ directly, but we can use the law of total probability to find $p(B)$, according to a decomposition of the sample space $\Omega$, such as $\Omega = A \cup \neg A$, where $\neg A = \Omega \setminus A$ denotes the event complementary to $A$. This law has two forms, one in terms of joint probabilities and one in terms of conditional probabilities:

```math
    \begin{align*}
        p(B) & = p(B\cap A) + p(B \cap \neg A) \\
            & = p(B|A)p(A) + p(B|\neg A)p(\neg A).
    \end{align*}
```

This law also applies to decompositions of the sample space in terms of several disjoint events, i.e. $\Omega = \cup_i A_i$, with disjoint $p(A_i \cap A_j) = 0$, for $i\neq j$.

Using this decomposition, we can write the Bayes' formula as

!!! note "Extended version of Bayes' formula"
    Let $p$ be a probability distribution on a given sample space and let $A$ and $B$ be two events with $p(B) > 0$. Then
    ```math
        p(A|B) = \frac{p(B|A)p(A)}{p(B|A)p(A) + p(B|\neg A)p(\neg A)}.
    ```

## Monty Hall problem

The [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) is a classic probability puzzle. In a television show, a contender has to choose between three doors, with only one of them giving you a reward. You choose one at random and you have 1/3 chance of choosing the right one. But after you choose this one, the host of the show reveals one of the doors which do not have any reward and asks if you want to choose a different door or keep the same. It turns out that if you switch to the remaining door, your chances rise to 2/3. Indeed, if you choose the right one at first and change, you loose, so this is a 1/3 chance of failure. But if you choose either one of the wrong ones at first, with a 2/3 probability, then the remaining wrong one is discarded by the host and you get to chance to the right one, giving you a 2/3 chance of success.

Let us do this more formally. Suppose $R$ denotes the door with the reward and $W$ the other two doors. Let $X$ be the random variable denoting the player's choice. With a single choice, $p(X=R) = 1/3$.

Now suppose we make two choices, denoted by the random variables $X_1$ and $X_2$. In the first strategy, that the player doesn't change his choice, we have $X_2 = X_1$. In this case, by the law of total probability

```math
    p(X_2 = R) = P(X_2 = R|X_1 = R)p(X_1 = R) + p(X_2 = R|X_1 = W)p(X_1 = W).
```

If the player doesn't change his choice, then $p(X_2 = R|X_1 = R) = 1$ and $p(X_1 = R) = 1/3$, while $p(X_2 = R|X_1 = W) = 0$, so that $p(X_2 = R) = 1/3$.

Now, if the player changes his choice, then $p(X_2 = R|X_1 = R) = 0$, while $p(X_2 = R|X_1 = W) = 1$ and $p(X_1 = W) = 2/3$, so that $p(X_2 = R) = 2/3$.

Or, using Bayes' rule,

```math
???
```

## Screening test

There are many applications of Bayes' Theorem in Biomedicine. Let's say, for example, a certain test for a given endemic disease (or illegal drug use, etc.) has a 4% chance of false negative and 0.1% chance of false positive, and suppose that the disease occurs in 1% of the population.

If a certain person tests positive, what are their chances of really carrying the disease? This means we want to know the conditional probability of having the disease, given that it tested positive. Let's use the following notation for the relevant events:

*  $D$ denotes the event of having the disease;
*  $\neg D$ denotes the event of not having the disease;
*  $P$ denotes the event of testing positive;
*  $N$ denotes the event of testing negative.

The chances of a person who tested positive to have the disease can be expressed as the conditional probability $p(D|P)$. Using Bayes' theorem, this can be expressed as

```math
    p(D|P) = \frac{p(P|D)p(D)}{p(P)}.
```

According to the given information, the probability $p(P|D)$ of testing positive while having the disease is 96%, since the false negatives $p(N|D)$ amount to 4%. The probability $p(D)$ of having the disease among the general population is 1%. Finally, the probability of testing positive can be obtained from the law of total probability:

```math
    p(P) = p(P|D)p(D) + p(P|\neg D)p(\neg D) = 96\% \times 1\% + 0.1\% \times 99\% = 1.059\%.
```

Thus, according to Bayes' Theorem,

```math
    p(D|P) = \frac{p(P|D)p(D)}{p(P)} = \frac{96\% \times 1\%}{1.059\%} \approx 90.6\%.
```

Hence, the chances a person who tested positive has indeed this disease are of about 90%, which is reasonably high.

If, however, the false negatives were of the order of 5% and the false positives were of the order of 1%, then the chances $p(D|P)$ of a person who tested positive to indeed have the disease would be only of the order of 49%! Pretty low, right? Not quite reliable. [PSA tests](https://www.cancer.gov/types/prostate/psa-fact-sheet) are one example where this conditional probability is low, of the order of 25%.

