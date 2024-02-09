# Bayes theorem and applications

In this section, we state Bayes' theorem and discuss some of its applications.

## Bayes' Theorem

**Bayes' Theorem** concerns the probability of a given event, conditioned to another event, and can be stated as follows.

!!! note "Bayes' Theorem"
    Let $p$ be a probability distribution on a given sample space and let $A$ and $B$ be two events with $p(B) > 0$. Then
    ```math
        p(A|B) = \frac{p(B|A)p(A)}{p(B)}.
    ```

In other words, Bayes' Theorem says that the *posterior* conditional probability $p(A|B)$ of an event $A$, given the occurrence of another event $B$, equals the *likelihood* $p(B|A)$ of the second event $B$ given the first event $A$ times the *prior* $p(A)$ divided by the *marginal* $p(B)$. The *prior* here refers to the probability $p(A)$ of $A$ *before* observing the event $B$, while the *posterior* $p(A|B)$ refers to the probability of $A$ *after* the observation of the event $B$.

Bayes' theorem has many useful consequences (see e.g. ), but first let us sketch its proof.

> **Proof of Bayes's Theorem.**
> 
> When $P(A) = 0$, then $P(A|B) = 0$ and the result is trivial. When $p(A) > 0$, the result can be obtained from the conditional probability relations
> ```math
>     p(A|B) = \frac{p(A\cap B)}{p(B)}, \qquad p(B|A) = \frac{p(B\cap A)}{p(A)},
> ```
> which imply
> ```math
>     p(A|B)p(B) = p(A\cap B) = p(B\cap A) = p(B|A)p(A).
> ```
> Solving for $p(A|B)$ yields the desired result. ■

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

## Bayesian inference on defect item

Suppose we have a collection of four six-faced dice, with two normal ones with faces numbered one to six, but one with two faces numbered five and none numbered six and one with three faces numbered four and none numbered five nor six. Let's call them dice types $D_{6}$, $D_{5}$ and $D_{4}$, respectively. A friend picks one of the dice at random and throws it repeatedly to find the numbers 3, 1, 4, 5, 1, 5, 2, 5, reading them aloud. What is the most likely type of die your friend picked?

Your prior is that a normal die is selected with probability 1/2,

```math
    p(D_{6}) = \frac{1}{2},
```

while the other two, with probability 1/4:

```math
    p(D_{5}) = p(D_{4}) = \frac{1}{4}.
```

Now, after learning about the *evidence* $E = (3, 1, 4, 5, 1, 5, 2, 5)$ of the numbers thrown by your friend, you update your *prior* with this evidence to find the *posteriors*

```math
    p(D_i | E) = \frac{p(E|D_i)p(D_i)}{p(E)}.
```

For each die, the likelyhood $p(E|D_i)$ is

```math
    p(E|D_i) = p(3|D_i)p(1|D_i)p(4|D_i)p(5|D_i)p(1|D_i)p(5|D_i)p(2|D_i)p(5|D_i)
```

Since $p(5|D_4) = 0$, $p(5|D_5) = 1/3$ and all remaining odds are $1/6$, we find

```math
    \begin{align*}
        p(E|D_4) & = 0, \\
        p(E|D_5) & = \left(\frac{1}{6}\right)^5\left(\frac{1}{3}\right)^3 = 8 \left(\frac{1}{6}\right)^8, \\
        p(E|D_6) & = \left(\frac{1}{6}\right)^8.
    \end{align*}
```

The probability of seeing this evidence is given by the law of total probability

```math
    p(E) = p(E|D_4)p(D_4) + p(E|D_5)p(D_5) + p(E|D_6)p(D_6) = 0 \times \frac{1}{4} + 8 \left(\frac{1}{6}\right)^8\frac{1}{4} + \left(\frac{1}{6}\right)^8\frac{1}{2}.
```

Hence,

```math
    p(E) = \frac{5}{2} \left(\frac{1}{6}\right)^8.
```

Therefore, the posteriors are

```math
    \begin{align*}
        p(D_4|E) & = 0, \\
        p(D_5|E) & = \frac{8 \times 1/4}{5/2} = 4/5, \\
        p(D_6|E) & = \frac{1 \times 1/2}{5/2} = 1/5.
    \end{align*}
```

Therefore, it is four times more likely that your friend picked the defective $D_5$-type die than the normal die, and, of course, the $D_4$-type die was surely not picked.

## Monty Hall problem

The [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) is a classic probability puzzle. In a television show, a contender has to choose between three doors, with only one of them giving you a reward. You choose one at random and you have 1/3 chance of choosing the right one. But after you choose this one, the host of the show reveals one of the doors which do not have any reward and asks if you want to choose a different door or keep the same. It turns out that if you switch to the remaining door, your chances rise to 2/3.

### Solution via probability tree

At first, you have 1/3 chance of choosing the right one and 2/3 chances of choosing a door without the reward. If you choose the right door and changes it after the host reveals an empty door, then you necessarily change to an empty door. This with a 1/3 chance. If you choose a door without reward and changes it after the host reveals an empty door, then you necessarily change it to the right door. This with a 2/3 chance. Hence, you have a 2/3 chance of success!

### Solving it via the law of total probability

Let us do this more formally. Suppose $R$ denotes the door with the reward. Let $X$ be the random variable denoting the player's choice. With a single choice, $p(X=R) = 1/3$.

Now suppose we make two choices, denoted by the random variables $X_1$ and $X_2$. In the first strategy, that the player doesn't change his choice, we have $X_2 = X_1$. In this case, we work with a probability conditioned to $X_2 = X_1$, and we simplify the notation to $p_1(E) = p(E|X_2 = X_1)$, for any possible event $E$. Then, by the law of total probability,

```math
    p_1(X_2 = R) = p_1(X_2 = R|X_1 = R)p_1(X_1 = R) + p_1(X_2 = R|X_1 \neq R)p_1(X_1 \neq R).
```

If the player doesn't change his choice, then $p_1(X_2 = R|X_1 = R) = 1$ and $p_1(X_1 = R) = 1/3$, while $p_1(X_2 = R|X_1 \neq R) = 0$, so that $p(X_2 = R|X_2 = X_1) = p_1(X_2 = R) = 1/3$.

Now, if the player changes his choice, then we work with the probability $p_2(E) = p(E|X_2 \neq X_1)$. In this case, $p_2(X_2 = R|X_1 = R) = 0$, while $p_2(X_2 = R|X_1 \neq R) = 1$ and $p_2(X_1 \neq R) = 2/3$, so that $p_2(X_2 = R) = 2/3$.

In this derivation, we did not make explicit the dependence on the choice of the host. We leave this as an exercise.

### Solving it via Bayes' rule

Suppose now you first pick door $X_1$, then the host picks door $H$, and next you choose door $X_2$. This means we have a random vector $(X_1, H, X_2)$ in a sample space with cardinality 27, meaning each choice can be any of the three doors. We are interested in the chances that $X_2$ is the door with the car, given that all chosen doors are different and that the host does not choose the door with the car. This corresponds to the strategy that the player changes the door. This can be written as the following conditional probability, where $R$ denotes the "right" door, with the car:

```math
    p(X_2 = R | X_2 \notin \{X_1, H\}, H \neq R, H \neq X_1)
```

Well, the condition of the show is that the host picks a different door than the first one chosen by the player, that that door is not the right one, and that the second door picked by the player is not the door chosen by the host. Under this rule, the player is free to stick we the first door, i.e. $X_2 = X_1$, or choose a different one, i.e. $X_2 \neq X_1$. In order to simplify the notation, we write the probability conditioned to the rules of the game as

```math
    \tilde p(E) = p(E | X_2 \neq H, H \neq X_1, H \neq R),
```

for any possible event $E$. Using Bayes' rule,

```math
    \tilde p(X_2 = R | X_2 \neq X_1) = \frac{\tilde p(X_2 \neq X_1 | X_2 = R)\tilde p(X_2 = R)}{\tilde p(X_2 \neq X_1)}.
```

Under the rules of the game,

```math
    \begin{align*}
        \tilde p(X_2 \neq X_1 | X_2 = R) & = 2/3; \\
        \tilde p(X_2 = R) & = 1/2 \\
        \tilde p(X_2 \neq X_1) & = 1/2
    \end{align*}
```

Hence,

```math
    \tilde p(X_2 = R | X_2 \neq X_1) = \frac{2/3 \times 1/2}{1/2} = \frac{2}{3}.
```

### Solving it via Bayes' rule by updating the prior

We can also use the classic interpretation of Bayes' rule as updating a prior distribution describing your chances of winning, according to new evidence revealed by the host. We'll do this in two ways, the first one, as it is usually presented, of updating the prior of having chosen correctly the first door $X_1$, and in a more natural way, of updating the prior of choosing correctly the last door opened $X_2$.

#### Updating the prior for $X_1 = R$

In this case, the player has, initially, one-third chance of choosing the right door:

```math
    p(X_1 = R) = \frac{1}{3}.
```

This is the player's *prior.* After that, the host reveals a door, showing it does not have a car, which we state as $H \neq R$. Now we want to update our *prior* to have a *posterior* probability

```math
    p(X_1 = R | H \neq R)
```

According to Bayes' rule,

```math
    p(X_1 = R | H \neq R) = \frac{p(H \neq R | X_1 = R) p(X_1 = R)}{p(H \neq R)}.
```

Well, the host always picks the door without the car, so both $p(H \neq R | X_1 = R)$ and $p(H \neq R)$ are equal to $1$. Meanwhile, your prior is $p(X_1 = R) = 1/3$. Thus,

```math
    p(X_1 = R | H \neq R) = \frac{1 \times 1/3}{1} = \frac{1}{3}.
```

This means your chances of having chosen the right door at first do not change after the evidence. On the other hand, after the host reveals their choice of door, you only have two alternatives: stick with $X_1$ or change to the only remaining door. This means

```math
    p(X_1 = R | H \neq R) + p(X_1 \neq R | H \neq R) = 1,
```

i.e.

```math
    p(X_1 \neq R | H \neq R) = 1 - \frac{1}{3} = \frac{2}{3}.
```

#### Updating the prior for $X_2 = R$.

A different way of solving this is to update directly the prior probability of selecting the right door $X_2 = R$, when switching your choice. Without the host opening a door, your chances are still one-third. This conditioned to the rules of the game, $H \neq X_1$, $H \neq R$, and of the strategy $X_2 \neq X_1$. In this case, despite the fact that the host chooses the door without the car, the prior does not use this knowledge and the player does not know which door was chosen by the host and, at first, can choose either one of the remaining two, with still a one-third chance of getting it right:

```math
    p(X_2 = R | X_2 \neq X_1, H \neq X_1, H \neq R) = \frac{1}{3}.
```

As before, in order to simplify the notation, we consider the probability conditioned to the rules of the game and the strategy but without including yet $X_2 \neq H$ since we will start with a prior that lacks this knowledge. Hence, we consider the conditional probability

```math
    \tilde p(E) = p(E | X_2 \neq X_1, H \neq X_1, H \neq R)
```

Thus, the prior reads

```math
    \tilde p(X_2 = R) = \frac{1}{3}.
```

Now, once the door $H$ is revealed, you choose $X_2 \neq H$ upon this new evidence. Hence, by Bayes' theorem,

```math
    \tilde p(X_2 = R | X_2 \neq H) = \frac{\tilde p(X_2 \neq H | X_2 = R) \tilde p(X_2 = R)}{\tilde p(X_2 \neq H)}.
```

The *likelihood* $\tilde p(X_2 \neq H | X_2 = R) = 1$ because surely $X_2\neq H$ when conditioned to $X_2 = R$ and $H \neq R$. The marginal $\tilde p(X_2 \neq H) = 1/2$ because when conditioned to $X_2 \neq X_1$ and $H \neq X_1$, there are two doors left for $X_2$, one equals to $H$ and the other different than $H$, hence one out of two possibilities of $X_2 \neq H$. Therefore,

```math
    \tilde p(X_2 = R | X_2 \neq H) = \frac{1 \times 1/3}{1/2} = \frac{2}{3}.
```
