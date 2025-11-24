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
│   ├── 01_ising_gpt.md            # Theory: Transformer learns Boltzmann distribution
│   ├── 02_decision_transformer.md # Theory: Offline RL as path integrals
│   ├── 03_maxent_rl.md            # Theory: MaxEnt RL & Langevin dynamics
│   ├── physics_primer.md          # Physics foundations
│   └── plots/                     # Generated visualizations
├── notebooks/
│   ├── 01_ising_model.ipynb       # Demo: Transformer learns 1D Ising model
│   ├── 02_Decision_Transformer.ipynb  # Demo: Trajectory transformers
│   └── 03_MaxEnt_RL.ipynb         # Demo: SAC implementation
├── src/physai/
│   ├── ising_data.py              # Ising model data generation
│   ├── ising_transformer.py       # Transformer architecture
│   ├── ising_diagnostics.py       # Validation & correlation functions
│   └── training.py                # Training utilities
└── requirements.txt
```

---

## Design Philosophy

**Notebooks = Demos**: Each notebook is a self-contained story with heavy commentary, visualizations, and "aha moments". Most demos run in < 5 minutes on modern hardware.

**src/physai/ = Reusable Package**: Core algorithms live in a clean Python package. Notebooks import from `physai` and focus on pedagogy, not implementation.

**Theory First**: Each chapter has both a theory document (in `docs/`) and an executable notebook. The theory provides rigorous derivations; the notebook provides hands-on experimentation.

---

## Chapter-by-Chapter Index

| Chapter | Title | Notebook | Documentation | Status |
|---------|-------|----------|---------------|--------|
| 01 | IsingGPT: Transformer Learns Phase Transitions | `01_ising_model.ipynb` | `docs/01_ising_gpt.md` | ✅ Complete |
| 02 | Decision Transformer = Trajectory Path Integral | `02_Decision_Transformer.ipynb` | `docs/02_decision_transformer.md` | 🚧 Theory complete, demo in progress |
| 03 | MaxEnt RL = Langevin + Maximum Caliber | `03_MaxEnt_RL.ipynb` | `docs/03_maxent_rl.md` | ✅ Complete |

---

## Quick Start

### 1. Setup Virtual Environment

I strongly recommend you use `uv` instead of conda.

```bash
# 1. Install uv (blazingly fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone & enter repo
git clone https://github.com/ZukSkyWalker/PhysAI-Zen.git
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
| **IsingGPT** | A 2-layer Transformer learns the exact Boltzmann distribution of the 1D Ising model from raw samples → attention heads spontaneously discover nearest-neighbor correlations → phase-transition-like behavior emerges | `01_ising_model.ipynb` |
| **Attention = Correlation Function** | Visualization of attention matrices at different temperatures → identical to spin-spin correlation functions C(r) | same notebook |
| **Decision Transformer** | Offline RL as discrete path integral over trajectories | `02_Decision_Transformer.ipynb` |
| **MaxEnt RL & SAC** | Complete implementation of Soft Actor-Critic with Langevin dynamics, Boltzmann policies, and training on Pendulum-v1 | `03_MaxEnt_RL.ipynb` |

---

## What Makes This Different?

- **No handwaving**: Every "≈" in the theory is backed by a rigorous equation
- **Runnable in seconds**: All demos work on a MacBook, no GPU needed
- **Physics-first**: If you know Boltzmann distributions and path integrals, you already know transformers
- **Production-ready code**: `src/` modules are clean, typed, and reusable

---

## Copyright Notice

This repository is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). You are free to share and adapt this work, but you must give appropriate credit, provide a link to the license, and indicate if changes were made. If you remix, transform, or build upon this material, you must distribute your contributions under the same license.
