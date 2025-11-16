# PhysAI-Zen

> Understand the Universe from the 1st Principles

## 🚀 Quick Setup


### 1. Install uv (Fast Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 2. Create Virtual Environment

```bash
# Create .venv in project root
uv venv

# Activate environment
source .venv/bin/activate  # bash/zsh
```

### 3. Install Dependencies

```bash
# Install all packages from requirements.txt
uv pip install -r requirements.txt

# Upgrade pip (optional but recommended)
uv pip install --upgrade pip
```

### 4. Setup Jupyter Kernel (Optional)

```bash
# Install ipykernel for Jupyter
uv pip install ipykernel
python -m ipykernel install --user --name=physai-zen
```

## 🛠️ Troubleshooting

### PyTorch Installation Issues
If you encounter PyTorch wheel issues:
```bash
# Force CPU-only version (fallback)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Flash Attention (Optional Performance Boost)
For faster transformer training, install flash-attn manually:
```bash
# Install system dependencies
brew install cmake ninja

# Install flash-attn (may take time to compile)
uv pip install flash-attn --no-build-isolation
```

### Mujoco Support (Advanced RL)
For full gymnasium support with Mujoco:
```bash
# Install Mujoco
brew install mujoco

# Then install gymnasium with mujoco
uv pip install "gymnasium[mujoco]"
```

## 🎯 Getting Started

After setup:
```bash
# Activate environment
source .venv/bin/activate

# Run Jupyter
jupyter lab

# Or run training scripts
python scripts/train_ising_gpt.py
```

## 📦 Included Libraries

- **Core**: PyTorch (with MPS acceleration), TorchVision, TorchAudio
- **Transformers**: Einops, Jaxtyping
- **Physics**: NumPy, SciPy, Matplotlib, Seaborn
- **RL**: Gymnasium, Stable-Baselines3, Weights & Biases
- **Quantum**: QuTiP, opt-einsum
- **Dev Tools**: JupyterLab, Rich (beautiful printing)
