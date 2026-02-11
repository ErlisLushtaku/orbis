"""Named constants for the SAE module, eliminating magic numbers."""

from typing import Final

# Numerical stability guard for division-by-zero in metric computations
EPSILON: Final[float] = 1e-8

DEFAULT_VAL_SPLIT: Final[float] = 0.1
DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_SAE_BATCH_SIZE: Final[int] = 4096

CACHED_BATCH_SIZE: Final[int] = 13824
SHARD_SHUFFLE_SIZE: Final[int] = 100
CACHE_SAVE_INTERVAL: Final[int] = 100

HISTOGRAM_BINS: Final[int] = 50
PLOT_DPI: Final[int] = 150

BYTES_PER_FLOAT16: Final[int] = 2
BYTES_PER_FLOAT32: Final[int] = 4

# Orbis ViT spatial grid (for 288x512 input, patch_size=1)
ORBIS_GRID_H: Final[int] = 18
ORBIS_GRID_W: Final[int] = 32

# Minimum samples required for reliable correlation
MIN_CORRELATION_SAMPLES: Final[int] = 50
