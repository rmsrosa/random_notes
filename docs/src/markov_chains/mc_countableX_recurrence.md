# Recurrence in the countable-space case

Recurrence is a fundamental concept related to the existence of invariant measures. We explore such concept here, in the context of a countable state space.

## Setting

Here, we assume that $(X_n)_n$ is a time-homogeneous, discrete-time Markov chain with a countable state space. More precisely, we assume the indices are $n=0, 1, 2, \ldots,$ and that the space $\mathcal{X}$ is finite or countably infinite. The sample space is the probability space $(\Omega, \mathcal{F}, \mathbb{P}),$ where $\mathcal{F}$ is the $\sigma$-algebra on the set $\Omega$ and $\mathbb{P}$ is the probability distribution. The one-step transition distribution is denoted by $K(x, y) = \mathbb{P}(X_{n+1} = y | X_n = x),$ and is independent of $n=0, 1, \ldots,$ thanks to the time-homogeneous assumption. Similary, the $n$-step transition distribution is denoted $K_n(x, y) = \mathbb{P}(X_{k+n} = y | X_k = x),$ for $n=1, 2, \ldots,$ independently of $k=0, 1, \ldots.$

## Definitions

We start with some fundamental definitions.

### Return time

!!! note "Definition (return time)"
    Given a point $x\in\mathcal{X},$ we define the **return time** to $x$ by
    ```math
        \tau_x = \inf\left\{n\in\mathbb{N}\cup\{+\infty\}; \; n = \infty \textrm{ or } X_n = x\right\}.
    ```

By definition, $\tau_x$ is a random variable with values in $\mathbb{N}.$ We call it *return time,* but, in the definition itself, we do not condition it on $X_0 = x,$ so it is not always a "return" time, *per se;* the quantity
```math
    \mathbb{P}(\tau_x < \infty)
```
is just the *probability of reaching $x$ in a finite number of steps.* It is more like a "first arrival" time. Only when conditioned to $X_0 = x$ that
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x)
```
is indeed the *probability of returning to $x$ in a finite number of steps.* Meanwhile, for $y \neq x,$
```math
    \mathbb{P}(\tau_y < \infty | X_0 = x)
```
is the *probability of reaching $y$ from $x$ in a finite number of steps.* 

### Number of passages

Another useful quantity is the random variable denoting the *number of passages* through a given state $x\in\mathcal{X},$

!!! note "Definition (number of passages)"
    The **number of passages** through a given state $x\in\mathcal{X}$ is defined by
    ```math
        \eta_x = \#\{n\in\mathbb{N}; \;X_n = x\} = \sum_{n=1}^\infty \mathbb{1}_{\{X_n = x\}}.
    ```

By definition, $\eta_x$ is a random variable with nonnegative integer values. Notice we did not include the starting time $n=0$ in the definition. Some authors do, while others don't (e.g. [Robert and Casella (2004)](https://doi.org/10.1007/978-1-4757-4145-2) don't, while [Lawler (2006)](https://doi.org/10.1201/9781315273600) does).

### Relations between return time and number of visits

There are some important relations between return time and number of visits. Indeed, the first return time is finite if, and only if, the state is visited at least once. This is valid for each sample point. We can express this as
```math
    \tau_x(\omega) < \infty \quad \Longleftrightarrow \quad \eta_x(\omega) \geq 1, \quad \forall \omega \in \Omega.
```
As a consequence,
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x) = \mathbb{P}(\eta_x \geq 1 | X_0 = x).
```

The complement of that is
```math
    \mathbb{P}(\tau_x = \infty | X_0 = x) = \mathbb{P}(\eta_x = 0 | X_0 = x).
```

More generally, the chances of having multiple visits is a power of the return time. This follows from the Markov property of the chain, since once back to $x$ for the $m-1$ time, the chances of coming back again is the same as coming back for the first time. And the chances of not returning after the $m-1$ visit is the same as the chances of never arriving any time.

!!! note "Proposition (return time and number of visits)"
    Let $x\in\mathcal{X}$ and set
    ```math
        q = \mathbb{P}(\tau_x < \infty | X_0 = x).
    ```
    Then,
    ```math
        \mathbb{P}(\eta_x \geq m | X_0 = x) = q^m,
    ```
    and
    ```math
        \mathbb{P}(\eta_x = m | X_0 = x) = q^m(1 - q),
    ```
    for any $m = 0, 1, 2, \ldots.$

**Proof.** By definition, we have $\eta_x \geq 0,$ thus the equality $\mathbb{P}(\eta_x \geq m | X_0 = x) = q^m$ for $m = 0$ is trivial. Now, for $m\in\mathcal{N},$ we have
```math
    \begin{align*}
        \mathbb{P}(\eta_x \geq m | X_0 = x) & = \mathbb{P}\bigg( \exist n_1, \ldots, n_m\in \mathbb{N}, X_i = x, n_{m-1} < i \leq n_m \Leftrightarrow i = n_m \\
        & \hspace{1in} \bigg| X_j = x, 0 \leq j \leq n_{m-1} \Leftrightarrow j = 0, n_1, \ldots, n_{m-1}\bigg) \\
        & \quad \times \mathbb{P}\bigg( \exist n_1, \ldots, n_{m-1} \in \mathbb{N}, X_i = x, n_{m-2} < i \leq n_{m-1} \Leftrightarrow i = n_{m-1} \\
        & \hspace{1in} \bigg| X_j = x, 0 \leq j \leq n_{m-2} \Leftrightarrow j = 0, n_1, \ldots, n_{m-2}\bigg) \\
        & \quad \times \cdots \\
        & \quad \times \mathbb{P}\bigg( \exist n_1 \in \mathbb{N}, X_i = x, 0 < i \leq n_1 \Leftrightarrow i = n_1 \bigg| X_0 = x \bigg)
    \end{align*}
```

By the Markov property of the chain, only the most recent conditioned state is important, so that
```math
    \begin{align*}
        \mathbb{P}(\eta_x \geq m | X_0 = x) & = \mathbb{P}\left( \exist n_1, \ldots, n_m\in \mathbb{N}, X_i = x, 1\leq i \leq n_m \Leftrightarrow i = n_1, \ldots,n_m | X_0 = x\right) \\
        & = \mathbb{P}\bigg( \exist n_{m-1}, n_m\in \mathbb{N}, X_i = x, n_{m-1} < i \leq n_m \Leftrightarrow i = n_m \bigg| X_{n_{m-1}} = x\bigg) \\
        & \;\; \times \mathbb{P}\bigg( \exist n_{m-2}, n_{m-1} \in \mathbb{N}, X_i = x, n_{m-2} < i \leq n_{m-1} \Leftrightarrow i = n_{m-1} \bigg| X_{n_{m-2}} = x \bigg) \\
        & \;\; \times \cdots \\
        & \;\; \times \mathbb{P}\bigg( \exist n_1 \in \mathbb{N}, X_i = x, 0 < i \leq n_1 \Leftrightarrow i = n_1 \bigg| X_0 = x \bigg)
    \end{align*}
```

By the time-homogeneous property, we can shift each probability by $-n_{k-1}$ to see that only the time differences $d_k = n_{k} - n_{k-1}$ matters, for $k=1, \ldots, m,$ with all the events conditioned at the initial time $n_0 = 0,$, i.e.
```math
    \begin{align*}
        \mathbb{P}(\eta_x \geq m | X_0 = x) & = \mathbb{P}\bigg( \exist d_m \in \mathbb{N}, X_i = x, 0 < i \leq d_m \Leftrightarrow i = d_m \bigg| X_0 = x\bigg) \\
        & \;\; \times \mathbb{P}\bigg( \exist d_{m-1} \in \mathbb{N}, X_i = x, 0 < i \leq d_{m-1} \Leftrightarrow i = d_{m-1} \bigg| X_0 = x \bigg) \\
        & \;\; \times \cdots \\
        & \;\; \times \mathbb{P}\bigg( \exist d_1 \in \mathbb{N}, X_i = x, 0 < i \leq d_1 \Leftrightarrow i = d_1 \bigg| X_0 = x \bigg)
    \end{align*}
```
The difference is just a matter of notation, for which we can denote them all by $d,$ and write
```math
    \mathbb{P}(\eta_x \geq m | X_0 = x) = \mathbb{P}\bigg( \exist d \in \mathbb{N}, X_i = x, 0 < i \leq d \Leftrightarrow i = d \bigg| X_0 = x\bigg)^m.
```
The existence of one such $d$ is equivalent to $\eta_x \geq 1,$ which is equivalent to $\tau_x < \infty,$ se we can write
```math
    \mathbb{P}(\eta_x \geq m | X_0 = x) = \mathbb{P}(\tau_x < \infty | X_0 = x)^m = q^m,
```
for all $m\in\mathbb{N},$ which completes the proof of the first statement.

Now, for any $m=0, 1, 2, \ldots,$ the events $\tau_x = m$ and $\tau_x \geq m+1$ are independent, so that
```math
    \begin{align*}
        \mathbb{P}(\eta_x = m | X_0 = x) & = \mathbb{P}(\eta_x \geq m | X_0 = x) - \mathbb{P}(\eta_x \geq m + 1 | X_0 = x) \\
        & = q^m - q^{m+1} \\
        & = q^m (1 - q),
    \end{align*}
```
which completes the proof. □  

### Recurrence and Transience

As the name says it, recurrence refers to a property that occurs repeatedly in time. In the case of a state $x\in\mathcal{X},$ we are interested in knowning how often the state is observed, i.e. how often $X_n = x,$ with respect to $n\in\mathbb{N}.$

The idea is that a state that is not visited after some instant in time is, in some sense, *transient,* while others that get visited infinitely often in time are *recurrent.* But since this is a stochastic process, we must quantify *how* often this happens in a probabilistic way, i.e. with respect to the underlying probability measure. This can be measured by the probability of a state to be visited infinitely often in the chain, i.e.
```math
    \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x).
```
If the probability is one, we will almost surely observe this state infinitely many times. If the probability is zero, we will almost surely observe this state at most a finite number of times. What if the probability is in between zero and one? Could it be in a superpositioned state, i.e. in some instances it is visited infinitely often while in other instances it is visited only finitely-many times? Fortunately, this cannot happen. The probability is either zero or one, and we can definitely characterize it as recurrent or transient. This is a manifestation of the Kolmogorov zero-one law for the tail event 
```math
    \{X_n = \textrm{i.o}\} = \bigcap_{n\in\mathbb{N}}\bigcup_{m\geq n}\{X_m = x\}.
```

!!! note "Proposition (zero-one law for infinitely-many visits)"
    Consider a state $x\in\mathcal{X}.$ Then either
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1
    ```
    or
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 0
    ```

**Proof.** The idea is to use the Markov property of the chain, that if the probability of being visited once after the initial time is $q,$ then, the probability of having $m$ visits is $q^m,$ to deduce that, at the limit $m\rightarrow \infty,$ it is either zero or one, depending on whether $0 \leq q < 1$ or $q = 1.$ 
    
More precisely, we know that
```math
    \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = \mathbb{P}(\eta_x = \infty | X_0 = x).
```
This means
```math
    \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = \mathbb{P}(\eta_x \geq m, \;\forall m\in\mathbb{N} | X_0 = x).
```
But
```math
    \{\eta_x \geq m, \;\forall m\in\mathbb{N}, X_0 = x \} = \bigcap_{m\in\mathbb{N}}\{\eta_x \geq m, X_0 = x\},
```
with the intersection being of non-increasing sets, with respect to increasing $m\in\mathbb{N}.$ Thus, by the continuity of probability measures,
```math
    \mathbb{P}(\eta_x \geq m, \;\forall m\in\mathbb{N}, X_0 = x) = \lim_{m\rightarrow} \mathbb{P}(\eta_x \geq m, X_0 = x).
```
Similarly,
```math
    \mathbb{P}(\eta_x \geq m, \;\forall m\in\mathbb{N} | X_0 = x) = \lim_{m\rightarrow} \mathbb{P}(\eta_x \geq m | X_0 = x).
```
We have already seen that
```math
    \mathbb{P}(\eta_x \geq m | X_0 = x) = q^m,
```
where
```math
    q = \mathbb{P}(\tau_x < \infty | X_0 = x).
```
Thus,
```math
    \mathbb{P}(\eta_x \geq m, \;\forall m\in\mathbb{N} | X_0 = x) = \lim_{m\rightarrow} q^m.
```
Clearly,
```math
    \lim_{m\rightarrow} q^m = 0, \quad \textrm{if } 0 < q \leq 1,
```
and
```math
    \lim_{m\rightarrow} q^m = 1, \quad \textrm{if } q = 1.
```
Since $q$ is a probability, which can only assume values in the range $0\leq q \leq 1,$ the only possible limits are 0 and 1, proving the result. □

With this in mind, we have the following definitions.

!!! note "Definition (recurrent and transient states)"
    A state $x$ is called **recurrent** when
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1,
    ```
    and is called **transient** when
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 0,
    ```
    with no intermediate values being possible.

Equivalent definitions of recurrence can be made with the notions of number of passages and return time. In that regard, we have the following result, which we borrow, essentially, from [Lawler (2006)](https://doi.org/10.1201/9781315273600), except that we do not include the time $n=0$ in the number of passages, so the formula is slightly different.

!!! note "Theorem (characterizations of recurrent and transient states)"
    For any state $x\in\mathcal{X},$ we have
    ```math
        x \textrm{ is recurrent } \quad \Longleftrightarrow \quad \mathbb{P}(\tau_x < \infty | X_0 = x) = 1 \quad \Longleftrightarrow \quad \mathbb{E}[\eta_x | X_0 = x] = \infty,
    ```
    and
    ```math
        x \textrm{ is transient } \quad \Longleftrightarrow \quad \mathbb{P}(\tau_x < \infty | X_0 = x) < 1 \quad \Longleftrightarrow \quad \mathbb{E}[\eta_x | X_0 = x] < \infty.
    ```
    Moreover, we have the relation
    ```math
        \mathbb{E}[\eta_x | X_0 = x] = \frac{\mathbb{P}(\tau_x < \infty | X_0 = x)}{1 - \mathbb{P}(\tau_x < \infty | X_0 = x)},
    ```
    with the understanding that the left hand side is infinite when the probability in the right hand side is 1.
    
**Proof.** We have that $\tau_x = \infty$ iff $X_n$ never returns to $x.$ If $\tau_x < \infty,$ then it returns to $x$ in finite time at least once. If $\tau_x < \infty$ with probability one, then, with probability one, it will return again and again to $x,$ still with probability one, since the countable intersection of sets of full measure still has full measure. Thus,
```math
    \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1 \quad \Longleftrightarrow \quad \mathbb{P}(\tau_x < \infty | X_0 = x) = 1.
```
This proves that $x$ is recurrent if, and only if, $\mathbb{P}(\tau_x < \infty | X_0 = x) = 1,$ which is the first part of the equivalence in the recurrence case. The complement of that is precisely that $x$ is transient if, and only if, $\mathbb{P}(\tau_x < \infty | X_0 = x) < 1.$

For the remaining equivalences, let us suppose, first, that $x$ is transient. Then, as we have seen,
```math
    q = \mathbb{P}(\tau_x < \infty | X_0 = x) < 1.
```
We compute the expectation of the number of passages by
```math
    \begin{align*}
        \mathbb{E}[\eta_x | X_0 = x] & = \mathbb{E}\left[ \sum_{n=1}^\infty \mathbb{1}_{X_n = x} \bigg| X_0 = x\right] = \sum_{n=1}^\infty \mathbb{E}\left[\mathbb{1}_{X_n = x} \bigg| X_0 = x\right] \\
        & = \sum_{n=1}^\infty \mathbb{P}(X_n = x | X_0 = x) = \sum_{n=1}^\infty p_n(x, x).
    \end{align*}
```
We can also compute this in a different way. Since $\eta_x$ is always an integer, its expectation is given by
```math
    \mathbb{E}[\eta_x | X_0 = x] = \sum_{m=1}^\infty m \mathbb{P}(\eta_x = m | X_0 = x).
```
We need a way to calculate $\mathbb{P}(\eta_x = m | X_0 = x),$ for each integer $m\in\mathbb{N}.$ When $\eta_x = m,$ it means it returns to $x$ $m$ times and then it does not return anymore. This means that $\mathbb{P}(\eta_x = m | X_0 = x) = q^m.$ Thus,
```math
    \mathbb{E}[\eta_x | X_0 = x] = \sum_{m=1}^\infty m q^m.
```
Let $S = \sum_{m=1}^\infty m q^m,$ so that $qS = \sum_{m=1}^\infty m q^{m+1} = \sum_{m=2}^\infty (m-1)q^m,$ and hence
```math
    (1 - q)S = S - qS = q + \sum_{m=1}^\infty q^m = \sum_{m=1}^\infty q^m = \frac{q}{1 - q}.
```
Thus,
```math
    \mathbb{E}[\eta_x | X_0 = x] = \frac{\mathbb{P}(\tau_x < \infty | X_0 = x)}{1 - \mathbb{P}(\tau_x < \infty | X_0 = x)}.
```
(In [Lawler (2006)](https://doi.org/10.1201/9781315273600), the number of passages includes the initial time $n=1,$ so that the formula obtained is $1/(1-q),$ instead of $q/(1 - q).$)

When
```math
    q = \mathbb{P}(\tau_x < \infty | X_0 = x) = 1,
```
then
```math
    \mathbb{P}(\tau_x < \infty | X_0 = x) \geq r,
```
for any $0 < r < 1,$ and we get, similarly, that
```math
    \mathbb{E}[\eta_x | X_0 = x] \geq \sum_{m=1}^\infty m r^m = \frac{r}{(1 - r)} \rightarrow \infty,
```
as $r \rightarrow 1,$ so that $\mathbb{E}[\eta_x | X_0 = x] = \infty.$

This proves the identity between the expectation and the probability. In particular, the expectation is finite if, and only if, the probability is strictly less than one, which proves the remaining equivalences. □

These equivalences are true, in general, only in the countable case; see page 222 of [Robert and Casella (2004)](https://doi.org/10.1007/978-1-4757-4145-2).

Another equivalence with recurrence is the following.

!!! note "Proposition (characterizations of recurrent and transient states)"
    Given a state $x\in\mathcal{X},$ we have the equality
    ```math
        \mathbb{E}[\eta_x | X_0 = x] = \sum_{n=1}^\infty K_n(x, x)
    ```
    and, therefore, $x$ is recurrent, when
    ```math
        \sum_{n=1}^\infty K_n(x, x) = \infty,
    ```
    and $x$ is transient, when
    ```math
        \sum_{n=1}^\infty K_n(x, x) < \infty.
    ```

**Proof.** The identity is proved using the definition of $\eta_x.$ Indeed,
```math
    \begin{align*}
        \mathbb{E}[\eta_x | X_0 = x] & = \mathbb{E}\left[\sum_{n=1}^\infty \mathbb{1}_{\{X_n = x\}} \bigg| X_0 = x\right] \\
        & = \sum_{n=1}^\infty \mathbb{E}\left[\mathbb{1}_{\{X_n = x\}} \bigg| X_0 = x\right] \\
        & = \sum_{n=1}^\infty \mathbb{P}\left[X_n = x \bigg| X_0 = x\right] \\
        & = \sum_{n=1}^\infty K_n(x, x).
    \end{align*}
```
Now, the characterization of recurrence and transience of $x$ follow from this identity and from the corresponding characterizations in terms of the expectation $\mathbb{E}[\eta_x | X_0 = x].$ This completes the proof. □

For instance, we have seen that the random walk $X_{n+1} = X_n + B_n,$ where $B_n$ are Bernoulli i.i.d. with states $+1$ with probability $p$ and $-1$ with probability $1 - p,$ where $0 < p < 1.$ For any $x, y\in\mathcal{X}=\mathbb{Z}$ and $n\in\mathbb{N},$ the transition probability $K_n(x, y)$ is zero if $|y - x|$ and $n$ have different parity or when $n < |y - x|,$ and is given by the binomial distribution
```math
    p_n(m) = \begin{pmatrix} n \\ i \end{pmatrix} p^i(1-p)^j,
```
with
```math
    i = \frac{n + m}{2}, \quad j = n - i = \frac{n - m}{2}, \quad m = y - x,
```
when
```math
    |m| \leq n, \quad m, n \textrm{ with same parity}
```
Thus, we can write
```math
    K_n(x, y) = \begin{cases}
        \begin{pmatrix} n \\ \frac{n + y - x}{2} \end{pmatrix} p^{(n+y-x)/2}(1-p)^{(n + x - y)/2}, & |x - y| \leq n, \; x - y, n \textrm{ same parity}, \\
        0, & \textrm{otherwise}.
    \end{cases}
```
In particular, the probability of returning to $x$ is
```math
    K_n(x, x) = \begin{cases}
        \begin{pmatrix} n \\ \frac{n}{2} \end{pmatrix} p^{n/2}(1-p)^{n/2}, & n \textrm{ is even}, \\
        0, & \textrm{otherwise}.
    \end{cases}
```
We have
```math
    \{X_n = x \textrm{ infinitely often} | X_0 = x\} = \bigcap_{n \in\mathbb{N}}\bigcup_{m \geq n, m\in \mathbb{N}} \{X_m = x | X_0 = x\}.
```
Thus, by the Borel-Cantelli Lemma,
```math
    \mathbb{P}\left(X_n = x \textrm{ infinitely often} | X_0 = x\right) = 0,
```
since
```math
    \begin{align*}
        \sum_{n\in\mathbb{N}} \mathbb{P}\left(X_m = x | X_0 = x\right) & = \sum_{n\in\mathbb{N}, n \textrm{ even}} \begin{pmatrix} n \\ \frac{n}{2} \end{pmatrix} p^{n/2}(1-p)^{n/2} \\
        & = \sum_{n\in\mathbb{N}} \begin{pmatrix} 2n \\ n \end{pmatrix} p^n(1-p)^n = \sum_{n\in\mathbb{N}} \frac{(2n)!}{2(n!)} (p(1-p))^n.
    \end{align*}
```
Using Stirling's approximation
```math
    n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n, \quad (2n)! \approx \sqrt{4\pi n}\left(\frac{2n}{e}\right)^{2n},
```
we have
```math
    \frac{(2n)!}{2(n!)} \approx \frac{\sqrt{4\pi n}}{2\sqrt{2\pi n}}\left(\frac{2n}{e}\right)^{2n}\left(\frac{e}{n}\right)^n = \frac{\sqrt{2}}{2}\left(\frac{4n}{e}\right)^n,
```
and we find
```math
    \sum_{n\in\mathbb{N}} \mathbb{P}\left(X_m = x | X_0 = x\right) \approx \frac{\sqrt{2}}{2}\sum_{n\in\mathbb{N}} \left(\frac{4np(1-p)}{e}\right)^n = \infty.
```
Thus, the chain is recurrent.

### Recurrent chain

When every state is recurrent we say that the chain is recurrent.

!!! note "Definition (recurrent chain)"
    The Markov chain is called **recurrent** when every state is recurrent, i.e.
    ```math
        \mathbb{P}(X_n = x \textrm{ infinitely often} | X_0 = x) = 1, \quad \forall x\in\mathcal{X}.
    ```

## Existence of invariant distribution

Recurrence is a fundamental property associated with the existence of invariant measures. But the associated invariant measure may be finite or infinite. If it is finite, then we can normalize it and obtain a stationary probability distribution. The condition for finiteness of the invariant measure associated with a recurrent state $x$ is that the expectation of $\tau_x,$ conditioned to $X_0 = x,$ be finite. If such expectation is finite, then necessarily $\tau_x$ is finite almost surely and, hence, $x$ is recurrent. But a state can be recurrent without this expectation being finite. Some authors call the latter case *null recurrence,* meaning that the state is recurrent but it is not associated with a stationary probability distribution. Otherwise, it is called *positive recurrence.*

!!! note "Theorem (recurrence implies existence of invariant measure)"
    Suppose that $x\in\mathcal{X}$ is recurrent. Then
    ```math
        {\tilde P}_x(z) = \sum_{n=1}^\infty \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x) = \mathbb{E}\left[\sum_{n=1}^{\tau_x} \mathbb{1}_{X_n = y} \bigg| X_0 = x\right]
    ```
    defines a non-trivial invariant measure for the Markov chain, which may be finite or infinite, with
    ```math
        {\tilde P}_x(\mathcal{X}) = \mathbb{E}[\tau_{x} | X_0 = x].
    ```
    Thus, if, moreover,
    ```math
        \mathbb{E}[\tau_{x} | X_0 = x] < \infty,
    ```
    then this measure is finite and can be normalized to a stationary probability distribution
    ```math
        P_x(z) = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=1}^\infty \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x).
    ```

The proof below is adapted from Theorem 6.37 of [Robert and Casella (2004)](https://doi.org/10.1007/978-1-4757-4145-2). Notice we are not assuming irreducibility, so we are not claiming that this invariant distribution is unique. In fact, we may have several non-connected recurrent state, each associated with a different invariant measure. We'll later see the condition of *irreducibility* to avoid such non-connected states and find a *unique* invariant probability distribution.

**Proof.** First of all, since the space is discrete and each ${\tilde P}_x(z) \geq 0,$ for $z\in\mathcal{X},$ it follows that ${\tilde P}_x$ defines indeed a measure on $\mathcal{X}.$ Let us check that ${\tilde P}_x$ is in fact a nontrivial measure and that it is invariant. 

In order to show that it is non-trivial, we prove that ${\tilde P}_x(x) = 1.$ This will also be a crucial fact in the proof of invariance. This follows from
```math
    \begin{align*}
        {\tilde P}_x(x) & = \sum_{n=1}^\infty \mathbb{P}(X_n = x, \tau_{x} \geq n | X_0 = x) \\
        & = \sum_{n=1}^\infty \mathbb{P}(X_n = x, \tau_{x} = n | X_0 = x) \\
        & = \sum_{n=1}^\infty \mathbb{P}(\tau_{x} \geq n | X_0 = x).
    \end{align*}
```
Thanks to the recurrence assumption on $x,$ we have $\tau_x < \infty$ almost surely, when conditioned on $X_0 = x,$ which means $\tau_x = \infty | X_0 = x$ has zero measure. Thus, 
```math
    \begin{align*}
        {\tilde P}_x(x) & = \sum_{n=1}^\infty \mathbb{P}(n \leq \tau_{x} < \infty | X_0 = x) \\
        & = \mathbb{P}(\cup_{n\in\mathbb{N}} \{\tau_{x} = n | X_0 = x\}) \\
        & = \mathbb{P}(\tau_{x} < \infty | X_0 = x).
    \end{align*}
```
Using again the fact that $x$ is recurrent, we have $\mathbb{P}(\tau_{x} < \infty | X_0 = x) = 1,$ which means
```math
    {\tilde P}_x(x) = 1,
```
as we wanted.

Now, let us check that ${\tilde P}_x$ is invariant. We need to show that
```math
    \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) = {\tilde P}_x(z),
```
for every $z\in\mathcal{X}.$ For that, we write
```math
    \begin{align*}
        \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \sum_{y\in\mathcal{X}} K(y, z)\sum_{n=1}^\infty \mathbb{P}(X_n = y, \tau_{x} \geq n | X_0 = x) \\
        & = \sum_{n=1}^\infty \sum_{y\in\mathcal{X}} K(y, z) \mathbb{P}(X_n = y, \tau_{x} \geq n | X_0 = x)
    \end{align*}
```
We split the sum according to $y=x$ and $y\neq x,$ so that
```math
    \begin{align*}
        \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \sum_{n=1}^\infty K(x, z) \mathbb{P}(X_n = x, \tau_{x} \geq n | X_0 = x) \\
        & \qquad + \sum_{n=1}^\infty \sum_{y\neq x} K(y, z) \mathbb{P}(X_n = y, \tau_{x} \geq n | X_0 = x) \\
        & = K(x, z) {\tilde P}_x(x) + \sum_{n=1}^\infty \sum_{y\neq x} \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n | X_0 = x) \\
    \end{align*}
```
For $y\neq z,$ we have
```math
    \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n | X_0 = x) = \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n+1 | X_0 = x).
```
Thus,
```math
    \begin{align*}
        \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = K(x, z) {\tilde P}_x(x) + \sum_{n=1}^\infty \sum_{y\neq x} \mathbb{P}(X_{n+1} = z, X_n = y, \tau_{x} \geq n+1 | X_0 = x) \\
        & = K(x, z) {\tilde P}_x(x) + \sum_{n=1}^\infty \mathbb{P}(X_{n+1} = z, \tau_{x} \geq n+1 | X_0 = x) \\
        & = K(x, z) {\tilde P}_x(x) + \sum_{n=2}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x),
    \end{align*}
```
where in the last step we just reindexed the summation. Now we use that ${\tilde P}_x(x) = 1$ (as proved above) and that 
```math
    K(x, z) = \mathbb{P}(X_1 = z | X_0 = x) = \mathbb{P}(X_1 = z, \tau_{x} \geq 1 | X_0 = x)
```
(since $\tau_{x} \geq 1$ always), to obtain
```math
    \begin{align*}
        \sum_{y\in\mathcal{X}} K(y, z){\tilde P}_x(y) & = \mathbb{P}(X_1 = z, \tau_{x} \geq 1 | X_0 = x) + \sum_{n=2}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x) \\
        & = \sum_{n=1}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x) \\
        & = {\tilde P}_x(z),
    \end{align*}
```
proving the invariance.

Now we compute ${\tilde P}_x(\mathcal{X}).$ We have
```math
    \begin{align*}
        {\tilde P}_x(\mathcal{X}) & = \sum_{z\in\mathcal{X}} P_x(z) \\
        & = \sum_{z\in\mathcal{X}}\sum_{n=1}^\infty \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x) \\
        & = \sum_{n=1}^\infty \sum_{z\in\mathcal{X}} \mathbb{P}(X_n = z, \tau_{x} \geq n | X_0 = x) \\
        & = \sum_{n=1}^\infty \mathbb{P}( \tau_{x} \geq n | X_0 = x) \\
        & = \sum_{n=1}^\infty \sum_{m=n}^\infty \mathbb{P}(\tau_{x} = m | X_0 = x) \\
        & = \sum_{m=1}^\infty \sum_{n=1}^{m} \mathbb{P}(\tau_{x} = m | X_0 = x) \\
        & = \sum_{m=1}^\infty m \mathbb{P}(\tau_{x} = m | X_0 = x) \\
        & = \mathbb{E}[\tau_{x} | X_0 = x]
    \end{align*}
```
If we assumed that 
```math
    \mathbb{E}[\tau_{x} | X_0 = x] < \infty,
```
it follows that the measure ${\tilde P}$ is finite, so we can normalize ${\tilde P}$ by this expectation to obtain the invariant probability distribution
```math
    P(z) = \frac{1}{\mathbb{E}[\tau_{x} | X_0 = x]} \sum_{n=1}^\infty \mathbb{P}(X_{n} = z, \tau_{x} \geq n | X_0 = x).
```
This concludes the proof.□

**Remark.** Notice that ${\tilde P}_x(z)$ defines a measure regardless of $x$ being recurrent or not. And it is always non-trivial, in fact, because $\sum_{y\in \mathbb{Z}} K(x, z) = 1,$ there must exist some $z\in\mathcal{Z}$ for which $X_0 = x$ and $X_1 = z,$ with positive probability, and thus, since $\tau_x \geq 1$ always,
```math
    {\tilde P}_x(z) = \sum_{n=1}^\infty \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x) \geq \mathbb{P}(X_1 = z, \tau_x \geq 1 | X_0 = 1) = \mathbb{P}(X_1 = z | X_0 1) = K(x, z) > 0.
```
But this measure may not be invariant. The recurrence is needed to assure that ${\tilde P}_x$ is invariant.

**Remark.** The expression
```math
        {\tilde P}_x(z) = \sum_{n=1}^\infty \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x) = \mathbb{E}\left[\sum_{n=1}^{\tau_x} \mathbb{1}_{X_n = y} \bigg| X_0 = x\right]
```
for the invariant measure appears naturally when we assume that an invariant measure exists. Indeed, for an invariant measure ${\tilde P}$ and for any two states $x, z\in\mathcal{X},$ one can show, by recusively using that the measure is invariant and by splitting the corresponding summation into the state equal to $x$ and the states different from $x,$ that
```math
    {\tilde P}(z) \geq {\tilde P}_x(z){\tilde P}(x).
```
This inequality will be proved in the following pages. This is then used to prove the uniqueness (up to a multiplicative constant) of the invariant measure (local or global, depending on whether just the chain is reducible or irreducible). In any case, we see, from this calculation, that the expression for ${\tilde P}_x(z)$ appears naturally from the hypothesis of invariance alone. This expression measures how often the chain visits a certain state $z.$ When in statistical equilibrium, this is what we expect as how frequent the state is observed.