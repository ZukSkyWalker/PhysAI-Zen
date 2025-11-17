# physai/__init__.py
from .ising import generate_ising_samples
# from .attention import plot_attention_as_correlation
# from .transformer import SimpleTransformer, RotaryEmbedding
# from .rl import SACAgent
# from .viz import plot_phase_transition, plot_attention_heatmap

__version__ = "0.1.0"
__all__ = [
    "generate_ising_samples",
    "SimpleTransformer",
]