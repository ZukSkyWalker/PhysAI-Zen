"""
Diagnostic tools for validating Ising model samples.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_magnetization(samples: torch.Tensor) -> torch.Tensor:
    """
    Compute average magnetization per sample.
    For equilibrium at h=0, should be ~0 due to Z₂ symmetry.
    
    Args:
        samples: (n_samples, L) tensor with spins ∈ {-1, +1}
    
    Returns:
        (n_samples,) tensor of magnetizations
    """
    return samples.mean(dim=1)


def compute_correlation_function(samples: torch.Tensor) -> torch.Tensor:
    """
    Compute spin-spin correlation function C(r) = ⟨s_i s_{i+r}⟩.
    
    Args:
        samples: (n_samples, L) tensor
    
    Returns:
        (L,) tensor: correlation as function of distance
    """
    L = samples.shape[1]
    corr = torch.zeros(L)
    
    for r in range(L):
        s_0 = samples[:, :]
        s_r = torch.roll(samples, shifts=r, dims=1)
        corr[r] = (s_0 * s_r).mean()
    
    return corr


def theoretical_correlation_1d(L: int, temp: float, J: float = 1.0) -> torch.Tensor:
    """
    Exact correlation function for 1D Ising model (transfer matrix solution).
    
    C(r) = tanh(β J)^r
    
    Args:
        L: chain length
        temp: temperature
        J: coupling constant
    
    Returns:
        (L,) tensor: theoretical correlation
    """
    beta = 1.0 / temp
    r = torch.arange(L, dtype=torch.float32)
    corr = torch.tanh(torch.tensor(beta * J)) ** r
    
    # Roll to center at r=0
    corr = torch.roll(corr, shifts=-L//2)
    return corr


def compute_autocorrelation_time(samples: torch.Tensor) -> float:
    """
    Estimate autocorrelation time from magnetization time series.
    If samples are well-decorrelated, this should be O(1).
    
    Args:
        samples: (n_samples, L) tensor
    
    Returns:
        Autocorrelation time (in units of samples)
    """
    mag = samples.mean(dim=1).cpu().numpy()  # (n_samples,)
    
    # Compute autocorrelation
    mag_centered = mag - mag.mean()
    acf = np.correlate(mag_centered, mag_centered, mode='full')
    acf = acf[len(acf)//2:]  # Keep only positive lags
    acf = acf / acf[0]  # Normalize
    
    # Integrated autocorrelation time: τ = 1 + 2 Σ_{t=1}^∞ C(t)
    # Truncate when ACF drops below 0.1
    cutoff = np.where(acf < 0.1)[0]
    if len(cutoff) > 0:
        cutoff = min(cutoff[0], len(acf) // 10)
    else:
        cutoff = len(acf) // 10
    
    tau = 1.0 + 2.0 * np.sum(acf[1:cutoff])
    return tau


def validate_ising_samples(
    samples: torch.Tensor,
    temp: float,
    J: float = 1.0,
    h: float = 0.0,
    plot: bool = True
) -> dict:
    """
    Comprehensive validation of Ising samples.
    
    Args:
        samples: (n_samples, L) tensor
        temp: temperature used for sampling
        J: coupling constant
        h: external field
        plot: whether to show diagnostic plots
    
    Returns:
        dict with validation metrics
    """
    n_samples, L = samples.shape
    
    # 1. Check magnetization distribution
    mag = compute_magnetization(samples)
    mag_mean = mag.mean().item()
    mag_std = mag.std().item()
    
    # 2. Compute correlation function
    corr_empirical = compute_correlation_function(samples)
    corr_theory = theoretical_correlation_1d(L, temp, J)
    
    # 3. Correlation length (fit exponential decay)
    r = torch.arange(L // 2, dtype=torch.float32)
    corr_positive = corr_empirical[:L//2]
    
    # Fit C(r) ~ exp(-r/ξ)
    log_corr = torch.log(torch.clamp(corr_positive, min=1e-8))
    slope, _ = np.polyfit(r.numpy(), log_corr.numpy(), deg=1)
    xi_empirical = -1.0 / slope if slope < 0 else float('inf')
    
    # Theoretical correlation length: ξ = -1 / log(tanh(βJ))
    xi_theory = -1.0 / np.log(np.tanh(1.0 / (temp * J))) if temp > 0 else float('inf')
    
    # 4. Autocorrelation time
    tau = compute_autocorrelation_time(samples)
    
    # 5. Check for Z₂ symmetry breaking (should see both ±1 domains equally at h=0)
    if h == 0.0:
        up_samples = (mag > 0.5).sum().item()
        down_samples = (mag < -0.5).sum().item()
        symmetry_ratio = min(up_samples, down_samples) / max(up_samples, down_samples + 1)
    else:
        symmetry_ratio = None
    
    results = {
        'n_samples': n_samples,
        'L': L,
        'temp': temp,
        'magnetization_mean': mag_mean,
        'magnetization_std': mag_std,
        'correlation_length_empirical': xi_empirical,
        'correlation_length_theory': xi_theory,
        'autocorrelation_time': tau,
        'Z2_symmetry_ratio': symmetry_ratio,
    }
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Magnetization histogram
        ax = axes[0, 0]
        ax.hist(mag.cpu().numpy(), bins=50, alpha=0.7, edgecolor='k')
        ax.axvline(0, color='r', linestyle='--', label='Expected (h=0)')
        ax.set_xlabel('Magnetization', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Magnetization Distribution\n⟨m⟩ = {mag_mean:.3f} ± {mag_std:.3f}', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Correlation function
        ax = axes[0, 1]
        r_plot = np.arange(-L//2, L//2)
        ax.plot(r_plot, corr_empirical.cpu().numpy(), 'o-', label='Empirical', markersize=5, lw=2)
        ax.plot(r_plot, corr_theory.cpu().numpy(), '--', label='Theory', lw=2)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xlabel('Distance r', fontsize=12)
        ax.set_ylabel(r'$\langle s_0 s_r \rangle$', fontsize=12)
        ax.set_title(f'Spin-Spin Correlation\nξ_emp = {xi_empirical:.2f}, ξ_theory = {xi_theory:.2f}', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Sample visualization
        ax = axes[1, 0]
        im = ax.imshow(
            samples[:20].cpu().numpy(),
            cmap='RdBu_r',
            aspect='auto',
            vmin=-1, vmax=1,
            interpolation='nearest'
        )
        ax.set_xlabel('Site', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        ax.set_title(f'First 20 Samples (T={temp:.2f})', fontsize=13)
        plt.colorbar(im, ax=ax, label='Spin')
        
        # 4. Autocorrelation
        ax = axes[1, 1]
        mag_np = mag.cpu().numpy()
        mag_centered = mag_np - mag_np.mean()
        acf = np.correlate(mag_centered, mag_centered, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        
        lags = np.arange(min(len(acf), 200))
        ax.plot(lags, acf[:len(lags)], 'o-', markersize=3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axhline(np.exp(-1), color='r', linestyle='--', label='1/e threshold')
        ax.set_xlabel('Lag (samples)', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_title(f'Magnetization Autocorrelation\nτ = {tau:.2f}', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results


def print_validation_report(results: dict):
    """Pretty print validation results."""
    print("=" * 60)
    print("ISING MODEL SAMPLE VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset: {results['n_samples']:,} samples × L={results['L']}")
    print(f"Temperature: T = {results['temp']:.2f}")
    print()
    print("Magnetization:")
    print(f"  ⟨m⟩ = {results['magnetization_mean']:.4f} (expect ~0 for h=0)")
    print(f"  σ(m) = {results['magnetization_std']:.4f}")
    print()
    print("Correlation Length:")
    print(f"  Empirical:    ξ = {results['correlation_length_empirical']:.2f}")
    print(f"  Theoretical:  ξ = {results['correlation_length_theory']:.2f}")
    print(f"  Relative error: {abs(results['correlation_length_empirical'] - results['correlation_length_theory']) / results['correlation_length_theory'] * 100:.1f}%")
    print()
    print("Decorrelation:")
    print(f"  Autocorrelation time: τ = {results['autocorrelation_time']:.2f}")
    print(f"  Status: {'✓ Good' if results['autocorrelation_time'] < 5 else '⚠ May need more decorrelation'}")
    print()
    if results['Z2_symmetry_ratio'] is not None:
        print("Z₂ Symmetry (h=0):")
        print(f"  Up/down domain ratio: {results['Z2_symmetry_ratio']:.3f}")
        print(f"  Status: {'✓ Symmetric' if results['Z2_symmetry_ratio'] > 0.4 else '⚠ Asymmetric (may need more equilibration)'}")
    print("=" * 60)

