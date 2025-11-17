# Cell: Dataset Preparation – From Spins to Tokens

import torch
from torch.utils.data import Dataset, DataLoader

def generate_ising_samples(
    L: int = 32,
    n_samples: int = 10_000,
    temp: float = 1.0,
    J: float = 1.0,
    h: float = 0.0,
    equilibration_steps: int = 500,      # 500 sweeps ≈ enough for equilibration
    steps_between_samples: int = 5,      # 5 sweeps between samples for decorrelation
    device: str | None = None,
) -> torch.Tensor:
    """
    Generate equilibrium samples from the 1D Ising model with periodic boundaries.

    For quick tests: n_samples=500, for training: n_samples=50_000.
    
    Returns: (n_samples, L) tensor with spins ∈ {-1, +1}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    
    beta = 1.0 / temp
    spins = torch.randint(0, 2, (L,), device=device, dtype=torch.float32) * 2 - 1
    
    # Initial equilibration (only once)
    for _ in range(equilibration_steps):
        _metropolis_sweep_vectorized(spins, beta, J, h)
    
    # Pre-allocate sample tensor
    samples = torch.empty((n_samples, L), dtype=spins.dtype, device=device)
    
    for i in range(n_samples):
        # Decorrelation steps between samples
        for _ in range(steps_between_samples):
            _metropolis_sweep_vectorized(spins, beta, J, h)
        
        samples[i] = spins
    
    # Global spin flip for zero-field case (vectorized over all samples)
    if h == 0.0:
        flip_mask = torch.rand(n_samples, device=device) < 0.5
        samples[flip_mask] = -samples[flip_mask]
    
    return samples


def _metropolis_sweep_vectorized(spins: torch.Tensor, beta: float, J: float, h: float):
    """
    Vectorized Metropolis sweep using checkerboard decomposition.
    Updates even sites, then odd sites in parallel.
    """
    L = len(spins)
    
    for parity in [0, 1]:
        sites = torch.arange(parity, L, 2, device=spins.device)
        
        if len(sites) == 0:
            continue
        
        left = torch.roll(spins, shifts=1, dims=0)[sites]
        right = torch.roll(spins, shifts=-1, dims=0)[sites]
        
        dE = 2.0 * spins[sites] * (J * (left + right) + h)
        
        accept_prob = torch.exp(-beta * dE)
        accept_prob = torch.clamp(accept_prob, max=1.0)
        
        rand = torch.rand(len(sites), device=spins.device)
        
        flip_mask = rand < accept_prob
        spins[sites[flip_mask]] *= -1


class IsingDataset(Dataset):
    def __init__(self, spins: torch.Tensor):
        """
        Convert ±1 spins → {0, 1} tokens for vocabulary size 2.
        Each sample becomes an autoregressive sequence: predict next spin from previous ones.
        """
        self.tokens = ((spins + 1) // 2).long()   # ±1 → 0/1
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        seq = self.tokens[idx]
        return seq[:-1], seq[1:]              # x = input, y = target (shifted by 1)


if __name__ == "__main__":
    # Create a temperature-mixed dataset (the secret sauce for robust learning)
    data = torch.cat([
        generate_ising_samples(temp=0.5, n_samples=25_000),
        generate_ising_samples(temp=1.0, n_samples=25_000),
        generate_ising_samples(temp=3.0, n_samples=25_000),
    ], dim=0)

    dataset = IsingDataset(data)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(dataset):,}")
    print(f"Vocabulary size: {dataset.tokens.max().item() + 1}")
    print(f"Example chain (tokens): {dataset[0][0]}")