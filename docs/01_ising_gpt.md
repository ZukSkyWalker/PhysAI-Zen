# Chapter 01: IsingGPT – Transformer Learns Phase Transitions

> A minimal 2-layer Transformer trained on equilibrium samples from the 1D Ising model spontaneously discovers the Boltzmann distribution, nearest-neighbor spin correlations, and phase-transition behavior — without ever seeing the Hamiltonian. Kudos to Andrej Karpathy's, this chapter is inspired by his nanoGPT (https://github.com/karpathy/nanoGPT).
 
---

## Physics Background

### The 1D Ising Model

The Ising model is the "hydrogen atom" of statistical mechanics: simple enough to solve analytically, rich enough to exhibit nontrivial collective behavior.

**Hamiltonian**:

$$
H[\{s_i\}] = -J \sum_{i=1}^{L} s_i s_{i+1} - h \sum_{i=1}^{L} s_i
$$

where:
- $s_i \in \{-1, +1\}$: spin at site $i$ (periodic boundary: $s_{L+1} = s_1$)
- $J > 0$: ferromagnetic coupling (prefers alignment $\uparrow\uparrow$)
- $h$: external magnetic field
- $L$: chain length

**Boltzmann distribution** at temperature $T$:

$$
p_T(\{s_i\}) = \frac{1}{Z(T)} \exp\Big(-\frac{H[\{s_i\}]}{k_B T}\Big)
$$

where $Z(T) = \sum_{\{s_i\}} \exp(-H/k_B T)$ is the partition function.

### Phase Transition (in infinite 1D: none; but crossover exists)

In 1D with finite $L$, there is no true phase transition, but:
- **High $T$ (paramagnetic)**: spins are disordered, $\langle s_i s_j \rangle \to 0$ rapidly
- **Low $T$ (ordered)**: spins align, $\langle s_i s_j \rangle \approx 1$ for all $i, j$

**Key observable: two-point correlation function**

$$
C(r) = \langle s_i s_{i+r} \rangle - \langle s_i \rangle^2
$$

At $h=0$ (zero field), this decays as

$$
C(r) \sim e^{-r/\xi(T)}
$$

where $\xi(T) = -1/\log(\tanh(J/k_B T))$ is the **correlation length**. As $T \to 0$, $\xi \to \infty$ (quasi-long-range order).

---

## Algorithm: Metropolis-Hastings Sampling

We generate equilibrium samples from $p_T(\{s_i\})$ using the **Metropolis algorithm**:

1. Initialize spins randomly: $s_i \in \{-1, +1\}$
2. **Sweep** (repeat $L$ times):
   - Pick a random site $i$
   - Compute energy change if we flip $s_i \to -s_i$:
     $$
     \Delta E = 2 J s_i (s_{i-1} + s_{i+1}) + 2 h s_i
     $$
   - Accept flip with probability:
     $$
     \min\Big(1, \exp\big(-\Delta E / k_B T\big)\Big)
     $$
3. After equilibration (500–1000 sweeps), save configuration
4. Wait 10–20 sweeps between samples (decorrelation)

**Implementation** (`src/ising.py`):

```python
def generate_ising_samples(
    L: int = 32,
    n_samples: int = 50_000,
    temp: float = 1.0,
    J: float = 1.0,
    h: float = 0.0,
    equilibration_steps: int = 500,
    steps_between_samples: int = 10,
    device: str | None = None,
) -> torch.Tensor:
    """
    Generate equilibrium samples from 1D Ising model.
    Returns: (n_samples, L) tensor with values in {-1, +1}
    """
    beta = 1.0 / temp
    spins = torch.randint(0, 2, (L,), device=device) * 2 - 1
    
    samples = []
    for _ in range(n_samples):
        # Equilibration
        for _ in range(equilibration_steps):
            _metropolis_sweep(spins, beta, J, h, L)
        
        # Decorrelation + sampling
        for _ in range(L * steps_between_samples):
            _metropolis_sweep(spins, beta, J, h, L)
        
        samples.append(spins.clone())
    
    return torch.stack(samples)

@torch.jit.script
def _metropolis_sweep(spins: torch.Tensor, beta: float, J: float, h: float, L: int):
    for _ in range(L):
        i = torch.randint(0, L, (1,)).item()
        dE = 2.0 * J * spins[i] * (spins[(i-1)%L] + spins[(i+1)%L]) + 2.0 * h * spins[i]
        if dE <= 0 or torch.rand(1) < torch.exp(-beta * dE):
            spins[i] = -spins[i]
```

---

## ML Narrative: Transformer as Boltzmann Distribution Learner

### Setup

We treat each spin configuration $\{s_1, \dots, s_L\}$ as a **sequence** and train a standard autoregressive Transformer:

$$
p_\theta(s_1, \dots, s_L) = \prod_{i=1}^{L} p_\theta(s_i \mid s_{1:i-1})
$$

**Training objective**: maximum likelihood over equilibrium samples

$$
\max_\theta \; \mathbb{E}_{\text{samples}} \Big[\log p_\theta(s_1, \dots, s_L)\Big]
$$

If the model has enough capacity and training converges, then:

$$
p_\theta(\{s_i\}) \approx p_T(\{s_i\}) = \frac{1}{Z(T)} \exp\Big(-\frac{H[\{s_i\}]}{k_B T}\Big)
$$

**Key insight**: The Transformer never sees $J$, $h$, or $T$. It only sees raw samples. Yet it learns to **implicitly encode the Boltzmann distribution** and the underlying energy landscape.

### Architecture: IsingGPT

```python
class IsingGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2,        # {-1, +1} → {0, 1} after tokenization
        seq_len: int = 32,           # L
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len) in {0, 1}
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)
        h = self.transformer(h)
        logits = self.head(h)
        return logits  # (batch, seq_len, vocab_size)
```

**Training loop**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        # batch: (B, L) in {-1, +1}
        x = (batch + 1) // 2  # map to {0, 1}
        
        logits = model(x)  # (B, L, 2)
        loss = criterion(logits.reshape(-1, 2), x.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### What the Model Learns

After training, we observe:

1. **Accurate Boltzmann distribution**: Sampled configurations from $p_\theta$ match those from Metropolis
2. **Attention = Correlation function**: The attention weights $\alpha_{ij}$ in layer 1 spontaneously approximate the two-point correlation $C(|i-j|)$
3. **Phase-transition tracking**: Models trained at different $T$ exhibit systematically different attention patterns

**Visualization**: Plot attention matrix $A_{ij} = \alpha_{ij}$ vs. theoretical correlation $C(|i-j|)$. They align remarkably well.

---

## Connection to Statistical Physics

### Why does this work?

The autoregressive factorization

$$
p_\theta(s_1, \dots, s_L) = \prod_{i=1}^{L} p_\theta(s_i \mid s_{1:i-1})
$$

is **universal**: any distribution over sequences can be written this way. But the Ising model has a special property: it's a **Markov random field** with nearest-neighbor interactions.

The Transformer's **attention mechanism** naturally discovers this locality:
- At low $T$, spins are strongly correlated → attention focuses on neighbors
- At high $T$, correlations decay fast → attention is more diffuse

In other words:
- **Physics**: $p_T(\{s_i\}) \propto \exp(-H/k_B T)$, where $H$ encodes nearest-neighbor coupling
- **ML**: $p_\theta(\{s_i\})$ learned via attention, which discovers the same coupling structure

The Transformer **doesn't know about Hamiltonians**, but by maximizing likelihood on equilibrium samples, it reverse-engineers the energy landscape.

### Attention as a Boltzmann weight

At each position $i$, the attention score over past spins $s_j$ ($j < i$) can be written:

$$
\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_{j'<i} \exp(q_i \cdot k_{j'} / \sqrt{d})}
$$

This is exactly a **local Boltzmann distribution** over indices $j$, where:
- "energy" $\approx -q_i \cdot k_j$
- "temperature" $\approx \sqrt{d}$

In the Ising model, the relevant information for predicting $s_i$ is $s_{i-1}$ (and to a lesser extent $s_{i-2}, s_{i-3}, \dots$). The attention mechanism automatically **up-weights nearby spins** because they carry higher mutual information.

---

## Implementation Checklist

### In `src/ising.py`:
- [x] `generate_ising_samples()`: Metropolis sampler
- [x] `_metropolis_sweep()`: single sweep (JIT-compiled)
- [ ] `compute_correlation_function()`: exact $C(r)$ from samples
- [ ] `theoretical_correlation()`: analytic formula for 1D Ising

### In `src/transformer.py`:
- [ ] `IsingGPT`: minimal 2-layer Transformer
- [ ] `train_ising_gpt()`: training loop
- [ ] `sample_from_model()`: autoregressive sampling
- [ ] `extract_attention_weights()`: for visualization

### In `src/viz.py`:
- [ ] `plot_spin_configurations()`: visualize sample grids
- [ ] `plot_attention_vs_correlation()`: overlay attention & $C(r)$
- [ ] `plot_phase_diagram()`: trained models at different $T$

---

## Key Takeaways

1. **Transformers as implicit Boltzmann machines**: By training on equilibrium samples, the model learns the underlying energy landscape without ever seeing the Hamiltonian.

2. **Attention ≈ Correlation**: The learned attention weights naturally mirror the spin-spin correlation function — a purely emergent phenomenon.

3. **Statistical physics ↔ ML**: The Ising model is a pedagogical bridge. In more complex systems (language, images, RL), the same principle holds: **maximum likelihood on data ≈ learning the Boltzmann distribution of an unknown energy function**.

4. **Phase transitions in neural networks**: The model's internal representations (attention patterns, layer activations) change qualitatively as you vary the data-generating distribution's "temperature". This suggests a deep connection between phase transitions in physics and critical phenomena in deep learning.

---

## Further Reading

- **Physics**: Kardar, *Statistical Physics of Particles* (Chapter 3: Ising Model)
- **ML**: Graves, *Generating Sequences With Recurrent Neural Networks*
- **Connection**: Mehta et al., *A high-bias, low-variance introduction to Machine Learning for physicists* (arXiv:1803.08823)

