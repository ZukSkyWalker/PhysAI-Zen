# PhysAI-Zen

> **See Modern AI through the Lens of Statistical Physics, Path Integrals, and Quantum Information**

This is not another "Illustrated Transformer".  
This is the version Richard Feynman would have written if he were alive in 2025.

We re-derive Transformers and Reinforcement Learning from first principles — using only the language of partition functions, Langevin dynamics, maximum entropy, and path integrals.

---

## Repository Structure

```
PhysAI-Zen/
├── docs/
│   └── physics_primer.md          # Theory: Boltzmann → Attention → MaxEnt RL
├── notebooks/
│   ├── 01_ising_gpt.ipynb         # Transformer learns 1D Ising phase transitions
│   ├── 02_decision_transformer.ipynb
│   └── 03_maxent_rl.ipynb
├── src/
│   ├── attention.py               # Attention mechanisms & kernel functions
│   ├── ising.py                   # Ising model sampling (Metropolis-Hastings)
│   ├── transformer.py             # Transformer building blocks
│   ├── rl.py                      # RL algorithms (SAC, Decision Transformer)
│   └── viz.py                     # Visualization utilities
└── requirements.txt
```

---

## Design Philosophy

**Notebooks = Demos**: Each notebook is a self-contained story with heavy commentary, visualizations, and "aha moments". Run in < 2 minutes on M4 Mac.

**src/ = Reusable Functions**: All core algorithms live in Python modules. Notebooks import from `src/` and focus on pedagogy, not implementation.

**Theory First**: Read `docs/physics_primer.md` before diving into notebooks. It maps every ML concept to statistical physics rigorously.

---

## Chapter-by-Chapter Index

| Chapter | Title | Notebook | Functions in `src/` | Status |
|---------|-------|----------|---------------------|--------|
| 01 | IsingGPT: Transformer Learns Phase Transitions | `notebooks/01_ising_gpt.ipynb` | `ising.py`: `generate_ising_samples()`, `transformer.py`: `IsingGPT` | In Progress |
| 02 | Decision Transformer = Trajectory Path Integral | `notebooks/02_decision_transformer.ipynb` | `rl.py`: `DecisionTransformer`, `trajectory_sampler()` | Planned |
| 03 | MaxEnt RL = Langevin + Maximum Caliber | `notebooks/03_maxent_rl.ipynb` | `rl.py`: `SAC`, `langevin_policy_update()` | Planned |

---

## Quick Start

### 1. Setup Virtual Environment

I strongly recommend you use `uv` instead of conda.

```bash
# 1. Install uv (blazingly fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone & enter repo
git clone https://github.com/yourname/PhysAI-Zen.git
cd PhysAI-Zen

# 3. Create virtual environment
uv venv

# 4. Activate
source .venv/bin/activate  # bash/zsh

# 5. Install dependencies
uv pip install -r requirements.txt

# 6. Launch JupyterLab
jupyter lab
```

### 2. Read the Theory

Start with `docs/physics_primer.md` — it's a 15-minute read that will completely reshape how you think about transformers and RL.

### 3. Run the Demos

Each notebook is self-contained and runs in < 2 minutes:

| Demo | What you will see | Notebook |
|------|-------------------|----------|
| **IsingGPT** | A 2-layer Transformer learns the exact Boltzmann distribution of the 1D Ising model from raw samples → attention heads spontaneously discover nearest-neighbor correlations → phase-transition-like behavior emerges | `notebooks/01_ising_gpt.ipynb` |
| **Attention = Correlation Function** | Visualization of attention matrices at different temperatures → identical to spin-spin correlation functions C(r) | same notebook |
| **Decision Transformer** | Offline RL as discrete path integral over trajectories | `notebooks/02_decision_transformer.ipynb` (planned) |
| **SAC as Langevin** | Soft Actor-Critic rewritten as physically correct Langevin sampling | `notebooks/03_maxent_rl.ipynb` (planned) |

---

## What Makes This Different?

- **No handwaving**: Every "≈" in the theory is backed by a rigorous equation
- **Runnable in seconds**: All demos work on a MacBook, no GPU needed
- **Physics-first**: If you know Boltzmann distributions and path integrals, you already know transformers
- **Production-ready code**: `src/` modules are clean, typed, and reusable

---

## License

MIT
