## Bayes theorem and applications

In this section, we state Bayes' theorem and discuss some of its applications.

### Bayes' Theorem

**Bayes' Theorem** concerns the probability of a given event, conditioned to another event, and can be stated as follows.

!!! note "Bayes' Theorem"
    Let $p$ be a probability distribution on a given sample space and let $A$ and $B$ be two events with $p(B) > 0$. Then
    ```math
        p(A|B) = \frac{p(B|A)p(A)}{p(B)}.
    ```

In plain words, Bayes' Theorem says that the conditional probability of a given event, based on the occurrence of another event, equals the likelihood of the second event given the first event times the probability of the first event.

Bayes' theorem has many useful consequences, but first let us sketch its proof.

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

## Monty Hall problem

The [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) is a classic probability puzzle. In a television show, a contender has to choose between three doors, with only one of them giving you a reward. You choose one and you have 1/3 chance of choosing the right one. But after you choose this one, the host of the show reveals one of the doors which do not have any reward and asks if you want to choose a different door or keep the same.

## Screening test

There are many applications of Bayes' Theorem in biomedicine. Let's say, for example, a certain test for a given endemic disease (or illegal drug use, etc.) has a 4% chance of false negative and 0.2% chance of false positive, and suppose that the disease occurs in 1% of the population.

If a certain person tests positive, what are their chances of really carrying the disease? This means we want to know the conditional probability of having the disease, given that it tested positive. Let's use the following notation for the relevant events:

*  $D$ denotes the event of having the disease;
*  $H$ denotes the event of not having the disease;
*  $P$ denotes the event of testing positive;
*  $N$ denotes the event of testing negative.

The chances of a person who tested positive to have the disease can be expressed as the conditional probability $p(D|P)$. Using Bayes' theorem, this can be expressed as

```math
    p(D|P) = \frac{p(P|D)p(D)}{p(P)}.
```

According to the given information, the probability $p(P|D)$ of testing positive while having the disease is 96%, since the false negatives $p(N|D)$ amount to 4%. The probability $p(D)$ of having the disease among the general population is 1%. Finally, the probability of testing positive can be obtained from the law of total probability

```math
    p(P) = p(P|D)p(D) + p(P|H)p(H) = 96\% \times 1\% + 0.2\% \times 99\% = 1.158\%.
```

Thus, according to Bayes' Theorem,

```math
    p(D|P) = \frac{p(P|D)p(D)}{p(P)} = \frac{96\% \times 1\%}{1.158\%} \approx 83\%.
```

Hence, the chances a person who tested positive has indeed this disease are of about 83%.
