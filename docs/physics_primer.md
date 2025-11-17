# Transformer and Reinforcement Learning In a Nutshell

> 我至今觉得 deep learning 里面很多“高级概念”，本质都没跑出我大学热统课那一个学期的所学。  
> This note is for physicists who secretly suspect that transformers are just partition-function machines.

I’ll keep the **main body in English**, but I’ll sprinkle some **Chinese side comments** where the physics intuition really matters.

To help you not over- or under-trust any analogy, each section has three layers:

- **Physics** – what the equation really says in statistical / quantum mechanics  
- **ML narrative** – how this shows up in transformers / RL  

---

## 1. Partition function & Boltzmann distribution (soul of probabilistic modeling)

**Physics**  
Consider a system in thermal equilibrium at temperature $T$, with all possible microstates $s \in \Omega$ and energy function $E[s]$.

Partition function (canonical ensemble) at temperature $T$:

$$
Z(T) = \sum_{s \in \Omega} \exp\!\Big(-\frac{E[s]}{k_B T}\Big)
$$

where $k_B$ is the Boltzmann constant.

The Boltzmann distribution for the equilibrium state at temperature $T$:

$$
p_T(s) = \frac{1}{Z(T)} \exp\!\Big(-\frac{E[s]}{k_B T}\Big)
$$

This is the **first-principles definition of a probability distribution** under energy and temperature:

- Lower energy $\Rightarrow$ exponentially higher probability  
- The partition function $Z$ is just the **normalization constant**

**ML narrative**  
You can treat a token sequence $\sigma = (\text{token}_1, \dots, \text{token}_n)$ as a “configuration” and define an **energy**

$$
E_\theta(\sigma) := -\log p_\theta(\sigma).
$$

Then a maximum-likelihood model is implicitly saying:

$$
p_\theta(\sigma) \propto \exp\big(-E_\theta(\sigma)\big).
$$

At the level of a **single attention head**, the softmax over scores

$$
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_k \exp(s_{ik})}
$$

is literally a **local Boltzmann distribution** over “which key to attend to,” with:

- “energy” $\approx -s_{ij}$  
- “inverse temperature” absorbed into the scale of $s_{ij}$ (e.g. $1/\sqrt{d}$)


---

## 2. Nadaraya–Watson kernel regression = Proto-attention

**Physics**  
Given data $\{(x_i, y_i)\}_{i=1}^n$, the Nadaraya–Watson (NW) estimator predicts

$$
\hat{y}(x_*) = 
\frac{\sum_{i=1}^n K_h(x_*, x_i)\, y_i}{\sum_{i=1}^n K_h(x_*, x_i)}.
$$

For a Gaussian kernel with bandwidth $h$,

$$
K_h(x_*, x_i) = \exp\Big(-\frac{\|x_* - x_i\|^2}{2h^2}\Big).
$$

Rewriting this in “Boltzmann form”, define

- $E_i = \|x_* - x_i\|^2$  
-  $k_BT = 2h^2$  
- **partition function**
  $$
  Z(x_*, T) = \sum_{i=1}^n \exp\Big(-\frac{E_i}{k_B T}\Big).
  $$

Then

$$
\hat{y}(x_*) = \sum_{i=1}^n 
\underbrace{\frac{\exp(-E_i / T)}{Z(x_*, T)}}_{p_i(x_*)}\, y_i.
$$

So the prediction is literally a **Boltzmann-weighted ensemble average** of the labels $y_i$ in a “heat bath” centered at $x_*$ with temperature $T = h^2$:

- Closer points $\Rightarrow$ smaller $E_i$ $\Rightarrow$ larger Boltzmann factor $\exp(-E_i/T)$ $\Rightarrow$ larger weight  
- Large $h$ (high $T$) $\Rightarrow$ very smooth, almost global averaging  
- Small $h$ (low $T$) $\Rightarrow$ almost nearest-neighbor interpolation

**ML narrative**  
Now map this directly to attention:

- Query $q$ plays the role of $x_*$  
- Keys $k_i$ play the role of training inputs $x_i$  
- Values $v_i$ play the role of labels $y_i$

Replace the Gaussian kernel with a dot-product kernel:

$$
K(q, k_i) = \exp\Big(\frac{q^\top k_i}{\sqrt{d}}\Big),
$$

and rename $y_i \to v_i$. Then

$$
\text{Attention}(q, K, V) 
= \frac{\sum_i \exp\big(q^\top k_i / \sqrt{d}\big)\, v_i}
       {\sum_i \exp\big(q^\top k_i / \sqrt{d}\big)}.
$$

This is **exactly** the same functional form as Nadaraya–Watson with an exponential kernel: a similarity-based Boltzmann average of the $v_i$.

In other words: **attention is Nadaraya–Watson on steroids**:

- replaces a fixed Gaussian kernel by a **learned dot-product kernel**  
- uses the $\sqrt{d}$ factor to control the “effective temperature” and stabilize gradients  
- runs many queries and many heads in parallel

Vaswani et al. (2017) essentially **re-discovered the 1964 NW idea** in a high-dimensional, learnable, massively parallel form.

---

## 3. Langevin dynamics – the engine behind sampling & MaxEnt RL

**Physics**  
Overdamped Langevin dynamics:

$$
\dot{x} = -\nabla E(x) + \sqrt{2\beta^{-1}}\,\eta(t),
$$

with $\eta(t)$ standard white noise. Under reasonable conditions, the stationary distribution is exactly the Boltzmann distribution

$$
p^*(x) \propto e^{-\beta E(x)}.
$$

So Langevin is **gradient descent + thermal noise** that asymptotically samples the target distribution.

**ML narrative**

- Many modern MCMC algorithms (MALA, ULA, Langevin-based samplers) are discretizations or variants of this dynamics.  

**ML view (RL / MaxEnt).**  
In MaxEnt RL, you optimize a policy ($\pi$) for

$$
J(\pi) = \mathbb{E}_\pi[R] + \alpha H(\pi).
$$

You can heuristically write a **gradient flow in policy space** as

$$
\dot{\pi} \propto -\nabla_\pi\Big(\mathbb{E}_\pi[-R] + \beta^{-1} H(\pi)\Big),
$$

which has the same structure: energy term + temperature-weighted entropy term. This ties SAC / Soft Q-Learning to **maximum entropy + Langevin-like dynamics in function space**.

**PPO / TRPO aside.**  
Their “trust region” constraints (KL not too large between successive policies) behave like a **step-size control**: don’t jump too far so that a local quadratic approximation of the objective remains valid.  
This is reminiscent of requiring the discrete Langevin step not to be too large so that the second-order expansion of $E(x)$ remains meaningful.



---

## 4. Maximum entropy / maximum caliber – moral justification of modern ML

**Physics / information theory**  
Jaynes’ maximum entropy principle:

Given constraints on expectations $\mathbb{E}_p[f_k(x)] = c_k$, choose the distribution with maximal entropy

$$
p^* = \arg\max_p H(p)
\quad \text{s.t.} \quad 
\mathbb{E}_p[f_k(x)] = c_k.
$$

Lagrangian duality yields

$$
p^*(x) = \frac{1}{Z(\lambda)} 
\exp\Big(-\sum_k \lambda_k f_k(x)\Big).
$$

This is the **exponential family**: energy = linear combination of constraints, plus a partition function.

For trajectories (paths), the natural generalization is the **maximum caliber** principle: maximize entropy over path distributions under trajectory-level constraints.

**ML view.**

- A linear layer + softmax at the output of a neural network is literally an exponential family model.  
- Cross-entropy training = maximum likelihood = consistent with maximum entropy under the chosen features.
- MaxEnt RL (e.g. SAC) optimizes

$$
J(\pi) = \mathbb{E}[R] + \alpha H(\pi),
$$

  which can be viewed as a **maximum caliber** principle on the space of trajectories.

**Comment**
- softmax + cross-entropy 完全可以从最大熵原理里推出来； 
- 把 reward 和 entropy 同时放进目标函数，就是在“路径空间”上做最大口径。  



---

## Quick reference table

| AI concept              | Physics counterpart                         | Key formula / idea                        | Strictness |
|-------------------------|---------------------------------------------|-------------------------------------------|-----------|
| Softmax Attention       | Boltzmann factor + local partition function | $\alpha_{ij} = \exp(q\cdot k)/Z$        | ★★★★☆ (local) |
| Transformer (AR LM)     | Boltzmann distribution over sequences       | $p_\theta(\text{seq}) \propto e^{-E_\theta(\text{seq})}$ | ★★★★☆ |
| Rotary Embedding        | Phase rotation, $U(1)$–like invariance    | $\exp(i\, m\cdot n\cdot \theta)$        | ★★☆☆☆ |
| Label Smoothing         | Finite-temperature correction               | $p \to (1-\varepsilon)p + \varepsilon\,\text{uniform}$ | ★★★★☆ |
| MaxEnt RL (SAC)         | Langevin + maximum caliber in policy space  | $J = R + \alpha H(\pi)$                 | ★★★★☆ |
| PPO Clip / TRPO         | Trust region ≈ “relativistic speed limit”   | avoid leaving the quadratic regime        | ★★☆☆☆ (metaphor) |

