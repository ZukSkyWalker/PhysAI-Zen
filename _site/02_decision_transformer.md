# Chapter 02: Decision Transformer – Offline RL as Discrete Path Integral

> Offline reinforcement learning recast as sequence modeling: trajectories are "paths" in state-action space, and the Transformer learns a distribution over trajectories weighted by cumulative reward — exactly analogous to a path integral with action $S = -\sum R_t$.

---

## Physics Background

### Path Integrals in Quantum Mechanics

In Feynman's formulation, the transition amplitude from state $x_0$ at $t=0$ to $x_T$ at time $T$ is:

$$
\langle x_T | e^{-iHT/\hbar} | x_0 \rangle = \int \mathcal{D}x(t) \, \exp\Big(\frac{i}{\hbar} S[x(t)]\Big)
$$

where the **action** is:

$$
S[x(t)] = \int_0^T dt \, \Big[\frac{m}{2}\dot{x}^2 - V(x)\Big]
$$

In the **Euclidean (imaginary-time)** version ($t \to -i\tau$), this becomes:

$$
Z = \int \mathcal{D}x(\tau) \, \exp\big(-S_E[x(\tau)]\big)
$$

with $S_E = \int d\tau \, [\frac{m}{2}\dot{x}^2 + V(x)]$. Now it looks like **statistical mechanics**: paths with lower action have exponentially higher weight.

### Discrete Path Integrals

For a system evolving in discrete time steps $t = 0, 1, \dots, T$:

$$
Z = \sum_{\text{all paths } x_0, x_1, \dots, x_T} \exp\big(-S_{\text{discrete}}[x_0, \dots, x_T]\big)
$$

where the discrete action might be:

$$
S_{\text{discrete}} = \sum_{t=0}^{T-1} \Big[\frac{(x_{t+1} - x_t)^2}{2\Delta t} + V(x_t) \Delta t\Big]
$$

**Key idea**: The system "explores all possible paths" but exponentially favors paths with low action.

### Mapping Physics to RL

In the discrete action above, we can identify two components:

1. **Kinetic term** (dynamics): $\frac{(x_{t+1} - x_t)^2}{2\Delta t}$ penalizes large jumps between consecutive states
2. **Potential term** (cost): $V(x_t) \Delta t$ assigns a cost to visiting state $x_t$

In **reinforcement learning**, the analogous structure is:

$$
S_{\text{RL}}[\tau] = \sum_{t=0}^{T-1} \Big[\underbrace{-\log p(s_{t+1} | s_t, a_t)}_{\text{dynamics constraint}} + \underbrace{(-r_t)}_{\text{negative reward}}\Big]
$$

**Correspondence**:
- **Dynamics** $p(s_{t+1} | s_t, a_t)$ ↔ **Constrained paths**: The MDP transition probabilities act like a "kinetic term" that constrains which state transitions are likely. Deterministic dynamics ($s_{t+1} = f(s_t, a_t)$) correspond to zero kinetic energy—the path is fully constrained.
  
- **Reward** $r_t = R(s_t, a_t)$ ↔ **Negative potential**: Reward plays the role of negative potential energy $-V(x_t)$. High reward = low action, making the trajectory more probable.

This gives the **trajectory distribution**:

$$
p(\tau) = p(s_0) \prod_{t=0}^{T-1} p(s_{t+1} | s_t, a_t) \cdot \pi(a_t | s_t) \cdot \exp\Big(\frac{1}{\alpha} r_t\Big)
$$

When we marginalize over the dynamics (which are fixed by the environment), we recover the MaxEnt RL objective where the policy $\pi(a|s)$ implicitly learns to sample high-reward trajectories.

---

## RL as Path Integral Over Trajectories

### Reinforcement Learning Setup

An agent interacts with an environment over discrete time steps:
- **State**: $s_t \in \mathcal{S}$
- **Action**: $a_t \in \mathcal{A}$
- **Reward**: $r_t = R(s_t, a_t)$
- **Dynamics**: $s_{t+1} \sim p(s' | s_t, a_t)$

A **trajectory** is a sequence:

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T, a_T, r_T)
$$

The **return** (cumulative reward) is:

$$
G(\tau) = \sum_{t=0}^{T} \gamma^t r_t
$$

where $\gamma \in [0,1]$ is the discount factor.

### Maximum Entropy RL (brief recap from Chapter 03)

In MaxEnt RL, we seek a policy $\pi(a|s)$ that maximizes:

$$
J(\pi) = \mathbb{E}_\tau \Big[\sum_t r_t\Big] + \alpha H(\pi)
$$

The optimal policy has the form:

$$
\pi^*(a|s) \propto \exp\Big(\frac{1}{\alpha} Q^*(s, a)\Big)
$$

This is a **Boltzmann policy** with "temperature" $\alpha$.

### Trajectory Distribution as Path Integral

We can define a **distribution over trajectories** weighted by return:

$$
p(\tau) \propto \exp\Big(\frac{1}{\alpha} G(\tau)\Big)
$$

Rewrite $G(\tau) = \sum_t \gamma^t r_t$ as a "negative action":

$$
S[\tau] := -\sum_t \gamma^t r_t
$$

Then:

$$
p(\tau) = \frac{1}{Z} \exp\big(-S[\tau] / \alpha\big)
$$

where $Z = \sum_{\tau} \exp(-S[\tau]/\alpha)$ is the trajectory partition function.

**This is exactly the discrete path integral formulation**: trajectories with higher reward (lower "action" $S$) have exponentially higher probability.

---

## Algorithm: Decision Transformer

**Key insight** (Chen et al., NeurIPS 2021):  
Instead of learning a value function $Q(s,a)$ or policy $\pi(a|s)$, directly model the **conditional distribution**:

$$
p(a_t \mid s_0, a_0, r_0, \dots, s_t, \hat{R}_t)
$$

where $\hat{R}_t$ is the **desired return-to-go** at time $t$.

At test time:
1. Specify a high target return $\hat{R}_0$ (e.g., the maximum return seen in the dataset)
2. Sample actions autoregressively:
   $$
   a_t \sim p_\theta(a_t \mid s_{0:t}, a_{0:t-1}, r_{0:t-1}, \hat{R}_t)
   $$
3. Update return-to-go: $\hat{R}_{t+1} = \hat{R}_t - r_t$

**Why this works**: The model learns that "if I'm in state $s$ and want to achieve return $\hat{R}$, I should take action $a$ that historically led to $\hat{R}$ from $s$."

### Architecture

Treat the trajectory as a sequence of **interleaved tokens**:

$$
(\hat{R}_0, s_0, a_0, \hat{R}_1, s_1, a_1, \dots, \hat{R}_T, s_T, a_T)
$$

Use a **causal Transformer** with:
- **Token embeddings**:
  - $\hat{R}_t$: scalar → linear layer → $\mathbb{R}^{d}$
  - $s_t$: vector or image → embedding layer → $\mathbb{R}^{d}$
  - $a_t$: discrete or continuous → embedding layer → $\mathbb{R}^{d}$
- **Positional embeddings**: learned embeddings for timestep $t$
- **Causal attention**: tokens at time $t$ can only attend to $t' \leq t$

Output: predict $a_t$ given all tokens up to $(s_t, \hat{R}_t)$.

**Training objective**: supervised learning on offline trajectories

$$
\max_\theta \sum_{t=0}^{T} \log p_\theta(a_t \mid \hat{R}_{0:t}, s_{0:t}, a_{0:t-1})
$$

where $\hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ is computed from the recorded trajectory.



---

## ML Narrative: Trajectory Modeling vs. Dynamic Programming

### Traditional RL: Value Functions

Most RL algorithms (Q-learning, Actor-Critic, PPO) learn:
- **Value function**: $V^\pi(s) = \mathbb{E}_\pi[\sum_{t} \gamma^t r_t \mid s_0 = s]$
- **Q-function**: $Q^\pi(s, a) = \mathbb{E}_\pi[\sum_{t} \gamma^t r_t \mid s_0 = s, a_0 = a]$

These satisfy the **Bellman equation** (dynamic programming):

$$
Q^\pi(s, a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p(\cdot|s,a)} [V^\pi(s')]
$$

This is a **local, recursive** update: the value at $s$ depends only on the immediate reward and the value of the next state.

### Decision Transformer: Sequence Modeling

Instead, Decision Transformer:
1. Treats the **entire trajectory** as a sequence
2. Uses **global attention** over all timesteps
3. Learns the conditional distribution $p(a_t \mid \text{history}, \hat{R}_t)$

**Advantages**:
- No bootstrapping errors (no Bellman backup)
- No off-policy correction needed (just supervised learning)
- Naturally handles long-horizon dependencies (via attention)
- Can be trained on **offline data** without ever interacting with the environment

**Tradeoffs**:
- Requires large, diverse offline datasets
- Doesn't improve beyond the best trajectory in the dataset
- Less sample-efficient than model-based RL (but more stable)

---

## Connection to Path Integrals

### Why "Path Integral"?

In the trajectory distribution

$$
p(\tau) \propto \exp\Big(\frac{1}{\alpha} \sum_t \gamma^t r_t\Big),
$$

the sum $\sum_t \gamma^t r_t$ plays the role of a **discrete action**. The Decision Transformer implicitly learns this distribution by:
1. Conditioning on return-to-go $\hat{R}_t$ (≈ "boundary condition" in path integrals)
2. Modeling $p(a_t | s_t, \hat{R}_t, \text{past})$ (≈ "path propagator")

At inference, by setting a high $\hat{R}_0$, we **bias the sampling toward high-reward paths**, just as in a path integral, we can bias toward paths with low action by adjusting boundary conditions or sources.

### Analogy Table

| Path Integral (Physics) | Decision Transformer (RL) |
|-------------------------|---------------------------|
| Path $x(t)$ | Trajectory $\tau = (s_t, a_t, r_t)$ |
| Action $S[x(t)] = \int L(x, \dot{x}) dt$ | Negative return $S[\tau] = -\sum_t \gamma^t r_t$ |
| Kinetic term $\frac{m}{2}\dot{x}^2$ | Dynamics constraint $-\log p(s_{t+1} \mid s_t, a_t)$ |
| Potential $V(x)$ | Negative reward $-r_t$ |
| Weight $\exp(-S[x]/\hbar)$ | Probability $\exp(\sum_t r_t / \alpha)$ |
| Partition function $Z = \int \mathcal{D}x \, e^{-S[x]}$ | Trajectory partition function $Z = \sum_\tau e^{G(\tau)/\alpha}$ |
| Boundary condition $x(0), x(T)$ | Desired return $\hat{R}_0$, initial state $s_0$ |
| Propagator $K(x_T, x_0)$ | Trajectory distribution $p(\tau \mid s_0, \hat{R}_0)$ |

**Key insight**: The Decision Transformer is a **discrete, learned approximation** to the trajectory propagator. By conditioning on return-to-go $\hat{R}_t$, it learns which actions lead to high-reward paths, analogous to how a path integral weights paths by their action.

Note that in the RL setting, the dynamics $p(s_{t+1} | s_t, a_t)$ are given by the environment and constrain which trajectories are possible (like a "kinetic term" restricting path smoothness), while the policy chooses actions to maximize reward (minimize "potential energy" $-r_t$).

---

## Key Takeaways

1. **RL = Sequence Modeling**: By treating trajectories as sequences and conditioning on return-to-go, we bypass the need for value functions and Bellman equations.

2. **Path Integral View**: The trajectory distribution $p(\tau) \propto \exp(G(\tau)/\alpha)$ is mathematically identical to a discrete path integral with action $S = -G(\tau)$.

3. **Offline RL Made Simple**: Decision Transformer reduces offline RL to supervised learning — no off-policy corrections, no bootstrapping, just maximum likelihood.

4. **Attention = Long-Range Dependencies**: Unlike Markovian methods (Q-learning), the Transformer can discover and exploit correlations across long time horizons.

5. **Physics → ML**: The path integral formulation, developed for quantum mechanics, provides a unifying language for understanding both equilibrium statistical mechanics (Chapter 01: Ising) and sequential decision-making (this chapter).

---

## Further Reading

- **Original paper**: Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling* (NeurIPS 2021)
- **Path integrals in RL**: Kappen, *Path integrals and symmetry breaking for optimal control theory* (2005)
- **MaxEnt RL**: Levine, *Reinforcement Learning and Control as Probabilistic Inference* (arXiv:1805.00909)
- **Physics connection**: Feynman & Hibbs, *Quantum Mechanics and Path Integrals*

