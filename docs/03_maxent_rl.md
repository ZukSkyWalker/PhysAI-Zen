# Chapter 03: Maximum-Entropy RL – Langevin Dynamics + Maximum Caliber

> Soft Actor-Critic (SAC) and maximum-entropy reinforcement learning are not ad-hoc tricks — they are the **unique, principled solution** to control under uncertainty, derived from Langevin dynamics and Jaynes' maximum caliber principle.

---

## Physical Background

### Langevin Dynamics

Consider a particle moving in a potential $V(x)$ at temperature $T$, subject to **overdamped** (high-friction) dynamics:

$$
\dot{x} = -\nabla V(x) + \sqrt{2 D} \, \eta(t)
$$

where:
- $\nabla V(x)$: deterministic force (gradient descent on potential)
- $\eta(t)$: white noise, $\langle \eta(t) \eta(t') \rangle = \delta(t - t')$
- $D = k_B T / \gamma$: diffusion coefficient (Einstein relation), where $\gamma$ is friction

**Key result**: The stationary distribution is the **Boltzmann distribution**:

$$
p_{\text{eq}}(x) = \frac{1}{Z} \exp\Big(-\frac{V(x)}{k_B T}\Big)
$$

where $Z = \int dx \, \exp(-V(x)/k_B T)$.

**Interpretation**: The system **explores the energy landscape** via thermal fluctuations, but spends exponentially more time in low-energy regions. The temperature $T$ controls the exploration-exploitation tradeoff:
- High $T$: wide exploration (flat distribution)
- Low $T$: exploitation (concentrate at global minimum)

### Fokker-Planck Equation

The evolution of the probability density $p(x, t)$ under Langevin dynamics is governed by the **Fokker-Planck equation**:

$$
\frac{\partial p}{\partial t} = \nabla \cdot \Big[\nabla V(x) \, p + D \, \nabla p\Big]
$$

At equilibrium ($\partial p / \partial t = 0$), this yields the Boltzmann distribution.

---

## Maximum Caliber: Entropy Over Paths

### Jaynes' Maximum Entropy Principle (Static)

Given constraints $\mathbb{E}_p[f_k(x)] = c_k$, the **least-biased** distribution is:

$$
p^*(x) = \arg\max_p H(p) = \arg\max_p \Big(-\int p(x) \log p(x) \, dx\Big)
$$

subject to the constraints. The solution is the **exponential family**:

$$
p^*(x) = \frac{1}{Z} \exp\Big(-\sum_k \lambda_k f_k(x)\Big)
$$

### Maximum Caliber (Dynamic)

For **trajectories** $x(t)$ over time $[0, T]$, we maximize entropy over **path distributions** $P[x(t)]$:

$$
P^*[x(t)] = \arg\max_P \mathcal{H}[P]
$$

subject to trajectory-level constraints, e.g.:

$$
\mathbb{E}_P\Big[\int_0^T L(x(t), \dot{x}(t)) \, dt\Big] = \bar{L}
$$

The solution (Jaynes, 1980) is:

$$
P^*[x(t)] \propto \exp\Big(-\int_0^T L(x(t), \dot{x}(t)) \, dt\Big)
$$

This is exactly the **path-integral weight** from statistical mechanics!

**In RL context**: Replace $L$ with negative reward $-R$, and maximize caliber subject to a constraint on expected cumulative reward. The result is a policy that **maximizes both reward and entropy**.

---

## Maximum-Entropy Reinforcement Learning

### Standard RL Objective

Find a policy $\pi(a|s)$ that maximizes expected return:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \Big[\sum_{t=0}^{T} \gamma^t r_t\Big]
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ is a trajectory sampled by following $\pi$.

**Problem**: This often leads to:
- **Brittle policies**: deterministic, overfit to specific environment dynamics
- **Poor exploration**: gets stuck in local optima
- **Lack of robustness**: fails under perturbations

### MaxEnt RL Objective

Add an **entropy regularization** term:

$$
J_{\text{MaxEnt}}(\pi) = \mathbb{E}_{\tau \sim \pi} \Big[\sum_{t=0}^{T} \gamma^t \big(r_t + \alpha H(\pi(\cdot | s_t))\big)\Big]
$$

where:
- $H(\pi(\cdot | s)) = -\sum_a \pi(a|s) \log \pi(a|s)$: entropy of the policy at state $s$
- $\alpha > 0$: temperature parameter (controls exploration)

**Intuition**: Among all policies achieving the same expected reward, prefer the **most random** one (maximum entropy). This ensures:
- **Exploration**: the policy doesn't collapse to a single action
- **Robustness**: spread probability mass over multiple good actions
- **Transfer learning**: learned skills are more general

### Connection to Langevin Dynamics

Rewrite the value function for MaxEnt RL:

$$
V^\pi(s) = \mathbb{E}_{\pi} \Big[\sum_{t=0}^{\infty} \gamma^t (r_t + \alpha H(\pi(\cdot | s_t))) \Big| s_0 = s\Big]
$$

Define a "free energy" (Helmholtz analogy):

$$
F^\pi(s) = -V^\pi(s) = -\mathbb{E}[R] - \alpha H(\pi)
$$

The optimal policy minimizes $F^\pi(s)$ — this is exactly the **variational principle** in statistical mechanics.

In continuous time, the policy update can be written as a **gradient flow**:

$$
\frac{\partial \pi}{\partial t} \propto -\nabla_\pi F(\pi) = \nabla_\pi \Big(\mathbb{E}_\pi[R] + \alpha H(\pi)\Big)
$$

This is **Langevin dynamics in policy space**, where:
- "Energy" $\leftrightarrow$ negative reward $-R$
- "Temperature" $\leftrightarrow$ entropy coefficient $\alpha$

---

## Algorithm: Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is the **state-of-the-art** continuous-control algorithm based on MaxEnt RL.

### Soft Bellman Equation

Define the **soft Q-function**:

$$
Q^*(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s' \sim p(\cdot|s,a)} \Big[V^*(s')\Big]
$$

where the **soft value function** is:

$$
V^*(s) = \mathbb{E}_{a \sim \pi^*(\cdot|s)} \Big[Q^*(s, a) - \alpha \log \pi^*(a|s)\Big]
$$

Equivalently:

$$
V^*(s) = \alpha \log \int \exp\Big(\frac{1}{\alpha} Q^*(s, a)\Big) da
$$

This is a **soft maximum** (LogSumExp). As $\alpha \to 0$, it reduces to the standard $V^*(s) = \max_a Q^*(s, a)$.

### Optimal Policy

The optimal policy in MaxEnt RL is:

$$
\pi^*(a|s) = \frac{1}{Z(s)} \exp\Big(\frac{1}{\alpha} Q^*(s, a)\Big)
$$

where $Z(s) = \int \exp(Q^*(s,a)/\alpha) da$ is the partition function.

**This is exactly the Boltzmann distribution** over actions, with:
- "Energy" $= -Q^*(s, a)$
- "Temperature" $= \alpha$

### SAC Algorithm

SAC maintains:
1. **Soft Q-networks**: $Q_\phi(s, a)$ (two networks for stability, take minimum)
2. **Policy network**: $\pi_\theta(a|s)$ (Gaussian for continuous actions)
3. **Target networks**: $Q_{\phi'}$ (slowly updated via Polyak averaging)

**Training loop**:

**1. Critic update** (minimize soft Bellman error):

$$
L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \Big[\big(Q_\phi(s,a) - y\big)^2\Big]
$$

where the target is:

$$
y = r + \gamma \Big(\min_{i=1,2} Q_{\phi_i'}(s', a') - \alpha \log \pi_\theta(a'|s')\Big), \quad a' \sim \pi_\theta(\cdot | s')
$$

**2. Actor update** (maximize expected soft Q-value):

$$
L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(\cdot|s)} \Big[\alpha \log \pi_\theta(a|s) - Q_\phi(s, a)\Big]
$$

This is equivalent to minimizing the KL divergence:

$$
\text{KL}\Big(\pi_\theta(\cdot|s) \,\Big\|\, \frac{1}{Z(s)} \exp\big(Q_\phi(s, \cdot)/\alpha\big)\Big)
$$

**3. Temperature auto-tuning** (optional):

Adjust $\alpha$ to maintain a target entropy $\bar{H}$:

$$
L_\alpha = -\mathbb{E}_{a \sim \pi_\theta} \Big[\alpha \big(\log \pi_\theta(a|s) + \bar{H}\big)\Big]
$$

### Implementation Sketch

```python
class SAC:
    def __init__(self, state_dim, action_dim, alpha=0.2, gamma=0.99, tau=0.005):
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        self.policy = GaussianPolicy(state_dim, action_dim)
        
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau  # Polyak averaging coefficient
        
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # Soft Bellman target
            target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # --- Actor update ---
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # --- Soft update of target networks ---
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
    
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## ML Narrative: Why MaxEnt Wins

### Comparison with Standard RL

| Standard RL | MaxEnt RL (SAC) |
|-------------|-----------------|
| Policy: deterministic or $\epsilon$-greedy | Policy: stochastic Boltzmann |
| Exploration: external (e.g., noise injection) | Exploration: intrinsic (entropy bonus) |
| Bellman equation: $Q(s,a) = r + \gamma \max_{a'} Q(s',a')$ | Soft Bellman: $Q(s,a) = r + \gamma \, \mathbb{E}[\text{softmax}_\alpha Q(s',\cdot)]$ |
| Overestimates value (max bias) | Less biased (soft max) |
| Brittle to environment changes | Robust (multi-modal policy) |

### Empirical Advantages

1. **Sample efficiency**: SAC matches or beats PPO/DDPG on most continuous-control benchmarks
2. **Stability**: Entropy regularization prevents policy collapse, smooths optimization
3. **Off-policy**: Can reuse old data (unlike on-policy methods like PPO)
4. **Automatic exploration**: No need to manually tune exploration noise

### Connection to Physics

The **Boltzmann policy**

$$
\pi^*(a|s) \propto \exp\Big(\frac{Q^*(s,a)}{\alpha}\Big)
$$

is not a heuristic — it's the **unique solution** to:

$$
\max_\pi \, \mathbb{E}_\pi[Q(s, \cdot)] + \alpha H(\pi)
$$

This is the same variational problem solved by the **canonical ensemble** in statistical mechanics:

$$
\max_p \, \mathbb{E}_p[E] - \frac{1}{\beta} H(p) \quad \Rightarrow \quad p^*(x) \propto e^{-\beta E(x)}
$$

In other words: **SAC is not "inspired by" physics; it *is* physics applied to control**.

---

## Implementation Checklist

### In `src/rl.py`:
- [ ] `SAC`: main class (critic, actor, target networks)
- [ ] `GaussianPolicy`: squashed Gaussian policy for continuous actions
- [ ] `QNetwork`: soft Q-function
- [ ] `train_sac()`: training loop with replay buffer
- [ ] `evaluate_policy()`: rollout for evaluation

### In `src/langevin.py`:
- [ ] `langevin_update()`: direct Langevin dynamics in policy space (pedagogical)
- [ ] `compute_policy_gradient()`: ∇ (reward + entropy) for policy
- [ ] `visualize_policy_landscape()`: plot policy as function of temperature

### In `src/viz.py`:
- [ ] `plot_sac_learning_curve()`: reward + entropy over training
- [ ] `plot_boltzmann_policy()`: visualize $\pi(a|s)$ at different $\alpha$
- [ ] `compare_sac_vs_ddpg()`: side-by-side comparison

---

## Key Takeaways

1. **MaxEnt RL = Langevin Dynamics**: The policy update in SAC is mathematically equivalent to running Langevin dynamics in policy space, where reward is "negative energy" and $\alpha$ is temperature.

2. **Boltzmann Policy is Optimal**: The exponential form $\pi(a|s) \propto \exp(Q(s,a)/\alpha)$ is not arbitrary — it's the unique solution to maximizing expected return under an entropy constraint.

3. **Physics Provides the Principle**: Just as the canonical ensemble in stat mech is derived from maximum entropy, MaxEnt RL is derived from maximum caliber over trajectories.

4. **Temperature Controls Exploration**: The parameter $\alpha$ is not a "hyperparameter" — it's the thermodynamic temperature of the policy. High $\alpha$ = more exploration; low $\alpha$ = more exploitation.

5. **Soft Bellman = Smooth Value**: The soft Bellman equation $V(s) = \alpha \log \int e^{Q(s,a)/\alpha} da$ is a temperature-smoothed version of $V(s) = \max_a Q(s,a)$, reducing overestimation bias.

6. **Unified View**: From Ising spins (Chapter 01) to trajectories (Chapter 02) to policies (Chapter 03), the **Boltzmann distribution** is the universal language connecting statistical physics and machine learning.

---

## Further Reading

- **Original SAC paper**: Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL* (ICML 2018)
- **MaxEnt RL framework**: Levine, *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review* (arXiv:1805.00909)
- **Maximum Caliber**: Jaynes, *The Minimum Entropy Production Principle* (1980); Pressé et al., *Principles of maximum entropy and maximum caliber in statistical physics* (RMP 2013)
- **Langevin RL**: Nachum et al., *Bridging the Gap Between Value and Policy Based RL* (NeurIPS 2017)
- **Thermodynamics of RL**: Ortega & Braun, *Thermodynamics as a theory of decision-making with information-processing costs* (2013)

