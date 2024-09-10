# Connected states, irreducibility and uniqueness of invariant measures

The notion of communicating states is a fundamental concept related to the uniqueness of an invariant measure, at least in a local scope. When every pair of states communicate with each other, then we have the notion of irreducibility, which extends the uniquenesse to a global scope.

## Setting

As before, we assume that $(X_n)_n$ is a time-homogeneous, discrete-time Markov chain with a countable state space. More precisely, we assume the indices are $n=0, 1, 2, \ldots,$ and that the space $\mathcal{X}$ is finite or countably infinite. The sample space is the probability space $(\Omega, \mathcal{F}, \mathbb{P}),$ where $\mathcal{F}$ is the $\sigma$-algebra on the set $\Omega$ and $\mathbb{P}$ is the probability distribution. The one-step transition distribution is denoted by $K(x, y) = \mathbb{P}(X_{n+1} = y | X_n = x),$ and is independent of $n=0, 1, \ldots,$ thanks to the time-homogeneous assumption. Similary, the $n$-step transition distribution is denoted $K_n(x, y) = \mathbb{P}(X_{k+n} = y | X_k = x),$ for $n=1, 2, \ldots,$ independently of $k=0, 1, \ldots.$

## Definitions

We start with some fundamental definitions.

### Communicating points

Markov chains are about the probability of states changing with time. If starting at some state, some of the other states might be more likely to be observed in the future than others, and some might never be observed. We distinguish them by the notion of communication.

!!! note "Definition (communicating points)"
    We say that **$x$ leads to $y$** when there exists a nonnegative integer $n=n(x, y)$ such that $K_n(x, y) > 0.$ When $x$ leads to $y$, we write $x \rightarrow y.$ If $x$ does not lead to $y,$ we write $x \not\rightarrow y.$ When $x$ leads to $y$ and $y$ leads to $x,$ we say that these states **communicate** with each other, and we write $x \leftrightarrow y.$

**Remark.** Notice that this definition automatically gives that $x \leftrightarrow x,$ for every state $x,$ even if $x$ is transient and the chain that starts at $x$ never returns to $x.$ Because of this, some authors defines $x\rightarrow y$ restricting $n(x, y)$ to be a *positive* integer. As a side effect, the communication property looses the reflexivity property which has to be enforced in the definition of an equivalence relation, so we can properly decompose the space into equivalence classes of communicating states.

### Equivalence class of communicating states

The relation of mutual communication $x \leftrightarrow y$ is an equivalence class.

!!! note "Fact (communication is an equivalence relation)"
    The relation $x \leftrightarrow y$ is an equivalence relation which we term **communication relation.** The **communication class** of a state $x$ is denoted $[x].$

### Closed communication class

In particular, the state space can be decomposed into one or more communication classes. But a communication class may not carry an invariant measure. The chain may "leak" to other classes. More precisely, we may have one state in one class leading to another state in a different class. It is important to distinguish when this happens or not. For that, we have the following definition.

!!! note "Definition (Closed communication class)"
    A communication class $C$ is called **closed** when for every $x\in C$ and every $z\in\mathcal{X}$ such that $x\rightarrow z,$ we also have $z\in C.$ In other words, if $x\in C$ and $z\in \mathcal{X}\setminus C,$ then $x \not\rightarrow z.$

## Local uniqueness of invariant measures

First we have the following result, without any special assumption, but which will be key, for the uniqueness, when the state $x$ is assumed to be recurrent and with positive probability for the assumed invariant measure.

!!! note "Lemma (lower bound on an invariant measure)"
    Suppose ${\tilde P}$ is an invariant measure for the Markov chain. Then, for any two states $x, z\in\mathcal{X},$
    ```math
        {\tilde P}(z) \geq {\tilde P}_x(z){\tilde P}(x),
    ```
    where
    ```math
        {\tilde P}_x(z) = \sum_{n=1}^\infty \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x).
    ```

**Proof.** If ${\tilde P}$ is a given invariant measure, then, using this invariance,
```math
    {\tilde P}(z) = \sum_{y_1} K(y_1, z){\tilde P}(y_1).
```
Splitting the summation into $y_1=x$ and $y_1\neq x,$ we have
```math
    {\tilde P}(z) = K(x, z){\tilde P}(x) + \sum_{y_1\neq x} K(y_1, z){\tilde P}(y).
```
Using again the invariance for the term ${\tilde P}(y)$ inside the summation and spliting again the summation, we have
```math
    \begin{align*}
        {\tilde P}(z) & = K(x, z){\tilde P}(x) + \sum_{y_1\neq x} K(y_1, z)\left( \sum_{y_2} K(y_2, y_1){\tilde P}(y_2) \right) \\
        & = K(x, z){\tilde P}(x) + \sum_{y_1\neq x} K(x, y_1)K(y_1, z){\tilde P}(x) + \sum_{y_1\neq x}\sum_{y_2\neq x} K(y_2, y_1)K(y_1, z){\tilde P}(y_2).
    \end{align*}
```
By induction, we obtain
```math
    \begin{align*}
        {\tilde P}(z) & = K(x, z){\tilde P}(x) \\
        & \quad + \sum_{y_1\neq x} K(x, y_1)K(y_1, z){\tilde P}(x) \\
        & \quad + \sum_{y_1\neq x}\sum_{y_2\neq x} K(x, y_1)K(y_1, z){\tilde P}(x) \\
        & \quad + \cdots \\
        & \quad + \sum_{y_1\neq x}\cdots \sum_{y_{k-1}\neq x} K(x, y_{k-1})K(y_{k-2}, y_{k-1})\cdots K(y_1, z){\tilde P}(x) \\
        & \quad + \sum_{y_1\neq x}\cdots \sum_{y_k\neq x} K(y_k, y_{k-1})K(y_{k-1}, y_{k-2})\cdots K(y_1, z){\tilde P}(x),
    \end{align*}
```
for every $k\in\mathbb{N}.$ Negleting the last term at each iteration $k,$ we find that
```math
    \begin{align*}
        {\tilde P}(z) & \geq \bigg(K(x, z) + \sum_{y_1\neq x} K(x, y_1)K(y_1, z) + \cdots \\
        & \qquad + \sum_{y_1\neq x}\cdots \sum_{y_{k-1}\neq x} K(x, y_{k-1})K(y_{k-2}, y_{k-1})\cdots K(y_1, z)\bigg){\tilde P}(x).
    \end{align*}
```
Now notice that, for $k=1,$
```math
    K(x, z) = \mathbb{P}(X_1 = z | X_0 = x) = \mathbb{P}(X_1 = z, \tau_x \geq 1 | X_0 = x),
```
for $k=2,$
```math
    \sum_{y_1\neq x} K(x, y_1)K(y_1, z) = \mathbb{P}(X_2 = z, X_1 \neq x | X_0 = x) = \mathbb{P}(X_2 = z, \tau_x \geq 2 | X_0 = x),
```
and, more generally, for any $k\in\mathbb{N},$
```math
    \begin{align*}
        \sum_{y_1\neq x}\cdots \sum_{y_{k-1}\neq x} K(x, y_{k-1})K(y_{k-2}, y_{k-1})\cdots K(y_1, z) & = \mathbb{P}(X_k = z, X_{k-1}\neq x, \ldots, X_1 \neq x | X_0 = x) \\
        & = \mathbb{P}(X_k = z, \tau_x \geq k | X_0 = x).
    \end{align*}
```
The summation of all such $k\in\mathbb{N}$ terms is precisely ${\tilde P}_x(z),$ and we find
```math
    {\tilde P}(z) \geq {\tilde P}_x(z){\tilde P}(x).
```
This concludes the proof. □

In the result above, we do not need to assume that $x$ is recurrent. If it is not, then both terms on the right hand side may vanish and the inequality is vacuous. However, if $x$ is recurrent and has positive measure ${\tilde P}(x) > 0$ with respect to the invariant measure, then we deduce that ${\tilde P}$ must be a multiple of ${\tilde P}_x,$ meaning uniqueness up to a multiplicative constant, at least locally among all communicating states to $x.$

In order to establish such result, let us first show that, if a state $x$ is recurrent, then the associated invariant measure ${\tilde P}_x$ is positive over the communication class $[x]$ and vanishes on the complement of $[x].$

!!! note "Proposition"
    Suppose $x$ is a recurrent state and that $[x]$ is a closed communication class. Then
    ```math
        {\tilde P}_x(z) > 0 \quad \Longleftrightarrow z \in [x].
    ```

**Proof.** If $z = x,$ then the result is immediate since ${\tilde P}_x(x) = 1.$ If $z\neq x$ but $z\in [x],$ then there exists $n\in\mathbb{N}$ such that $K_n(x, z) > 0.$ Since ${\tilde P}_x$ is invariant, we have
```math
    {\tilde P}_x(z) = \sum_{y\in\mathcal{X}} K_n(y, z) {\tilde P}_x(y) \geq K(x, z){\tilde P}_x(x)
```
Estimating from below the summation over $y\in\mathcal{x}$ by the summand at $y=x$ and using that $K_n(x, z) > 0$ and ${\tilde P}_x(x) = 1,$
```math
    {\tilde P}_x(z) = \sum_{y\in\mathcal{X}} K_n(y, z) {\tilde P}_x(y) \geq K_n(x, z){\tilde P}_x(x) = K_n(x, z) > 0,
```
which proves that $z\in [x]$ implies ${\tilde P}_x(z).$ Now for the converse. Suppose ${\tilde P}_x(z) > 0.$ If $z = x,$ then $z\in [x]$ and we are done. If $z\neq x,$ we have, by the definition of this measure as
```math
    {\tilde P}_x(z) = \sum_{n=1}^\infty \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x),
```
that, for some $n\in\mathbb{N},$
```math
    \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x) > 0.
```
In particular,
```math
    K_n(x, z) = \mathbb{P}(X_n = z | X_0 = x) \geq \mathbb{P}(X_n = z, n \leq \tau_{x} | X_0 = x) > 0,
```
which means that $x \rightarrow z.$ Now, since $[x]$ is closed, this means that $z\in [x].$ This completes the characterization of $z\in [x]$ as ${\tilde P}_x(z) > 0$.  □

!!! note "Theorem (local uniqueness up to a multiplicative constant)"
    Assume that $x$ is a recurrent state with closed communication class $[x].$ Suppose that ${\tilde P}$ is an invariant measure for the Markov chain which is carried by $[x]$ and with ${\tilde P}(x) > 0.$ Then, for any state $z \in [x],$ both
    ```math
        {\tilde P}_x(z), {\tilde P}(z) > 0
    ```
    and
    ```math
        \frac{{\tilde P}(z)}{{\tilde P}_x(z)} \geq \frac{{\tilde P}(x)}{{\tilde P}_x(x)},
    ```
    which implies that ${\tilde P}$ and ${\tilde P}_x$ are proportional i.e. there exists $C > 0$ such that
    ```math
        \frac{{\tilde P}(z)}{{\tilde P}_x(z)} = C, \quad \forall z\in [x].
    ```

**Proof.** It follows from the lemma on the lower bound on an invariant measure that
```math
    {\tilde P}(z) \geq {\tilde P}_x(z){\tilde P}(x),
```
for all $z\in\mathcal{X}.$ Since $x$ is a recurrent state, it follows that ${\tilde P}_x$ is also an invariant measure and, as seen in the previous proposition, since $[x]$ is closed,
```math
    {\tilde P}_x(z) > 0, \quad \forall z\in [x].
```
We also know that
```math
    {\tilde P}_x(x) = \mathbb{P}(\tau_{x} < \infty | X_0 = x) = 1.
```
Thus, we obtain the inequality
```math
    \frac{{\tilde P}(z)}{{\tilde P}_x(z)} \geq \frac{{\tilde P}(x)}{{\tilde P}_x(x)}, \quad \forall z\in [x].
```
Since ${\tilde P}(x) > 0,$ this implies that
```math
    {\tilde P}(z) > 0, \quad \forall z\in [x].
```
This proves the first claim that ${\tilde P}$ is also positive on the communication class $[x].$

Now, using that ${\tilde P}$ is invariant and is carried by $[x],$ we have
```math
    {\tilde P}(x) = \sum_z K_n(z, x){\tilde P}(z) = \sum_{z\in [x]} K_n(z, x){\tilde P}(z).
```
Using the previous inequality and the fact that ${\tilde P}_x$ is also invariant and carried by $[x],$ we have, for any $n\in\mathbb{N},$
```math
    \begin{align*}
        {\tilde P}(x) & = \sum_{z\in [x]} K_n(z, x){\tilde P}(z) \\
            & \geq \frac{{\tilde P}(x)}{{\tilde P}_x(x)} \sum_{z\in [x]} K_n(z, x){\tilde P}_x(z) \\
            & = \frac{{\tilde P}(x)}{{\tilde P}_x(x)} {\tilde P}_x(x) \\
            & = {\tilde P}(x).
    \end{align*}
```
This means that the inequality in the middle is actually an equality, i.e.
```math
    \sum_{z\in [x]} K_n(z, x){\tilde P}(z) = \frac{{\tilde P}(x)}{{\tilde P}_x(x)} \sum_{z\in [x]} K_n(z, x){\tilde P}_x(z).
```
Since for each $z \in [x]$ we have
```math
    {\tilde P}(z) \geq \frac{{\tilde P}(x)}{{\tilde P}_x(x)} {\tilde P}_x(z)
```
and the summation of these terms is equal, this means that each summand must be equal, i.e.
```math
    {\tilde P}(z) = \frac{{\tilde P}(x)}{{\tilde P}_x(x)} {\tilde P}_x(z), \quad \forall z\in [x].
```
Dividing by ${\tilde P}_x(z),$ which we know is positive over $[x],$ yields the second claim.

Finally, considering the constant
```math
    C = \frac{{\tilde P}(x)}{{\tilde P}_x(x)},
```
yields the claim that ${\tilde P}$ is proportional to ${\tilde P}_x,$ completing the proof. □

!!! note "Corollary"
    Suppose the chain is recurrent and irreducible. Then there is only one invariant measure, up to a multiplicative constant.

**Proof.** Since the chain is assumed to be recurrent, any $x\in\mathcal{X}$ is recurrent, which means ${\tilde P}_x$ is a nontrivial invariant measure, for any $x\in\mathcal{X}.$ Since the chain is irreducible, the whole space is a single communication class and is also closed because the chain has nowhere else to go. Thus, any invariant measure is a multiple of any of the ${\tilde P}_x,$ $x\in\mathcal{X},$ and any pair ${\tilde P}_{x}$ and ${\tilde P}_y$ is one a constant multiple of the other.

!!! note "Corollary (Kac's Theorem)"
    Suppose the chain is irreducible and has a stationary probability distribution ${\tilde P}.$ Then
    ```math
        {\tilde P}(x) = \frac{1}{\mathbb{E}[\tau_x | X_n = x]}.
    ```

**Proof.** ...

**Remark.** If we consider the chain
```math
    X_{n+1} = \begin{cases} 
        X_n + 1, & n \neq -1, 0, \\
        X_n + 2, & n = -1, \\
        X_n, & n = 0.
    \end{cases}
```
Then, the only recurrent state is $X_n = 0.$ The associated stationary probability distribution is the Dirac Delta at $X=0.$ On the other hand, the counting measure is also invariant and has measure 1 at any point, including the recurrent point $X=0,$ but this measure is not proportional to the Dirac Delta, so it does not suffice to assume that ${\tilde P}$ is an invariant measure with ${\tilde P}(x) > 0$ at a recurrent point $x.$ One must assume that ${\tilde P}$ is carried by the equivalence class of $x.$