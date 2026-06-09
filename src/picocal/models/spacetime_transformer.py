"""Space-time kernel transformer (the proposed, novel model).

Treats PicoCal's ~15 ps timing as a spatial coordinate via a space-time
positional encoding, with linear-complexity kernel attention so it can scale
to high-occupancy events. Implemented in Phase 2 (Weeks 5-10).
"""
from __future__ import annotations

try:
    from torch import nn

    _HAS_TORCH = True
except Exception:  # torch not installed in lint-only environments
    _HAS_TORCH = False


if _HAS_TORCH:

    class SpaceTimeTransformer(nn.Module):
        """Kernel transformer with space-time positional encoding.

        Args:
            in_features: Number of per-cell input features (x, y, z, energy, t).
            d_model: Transformer hidden dimension.
            use_timing: If False, drops the timing coordinate (for the
                timing-ablation study in Phase 2).
        """

        def __init__(
            self,
            in_features: int = 5,
            d_model: int = 128,
            use_timing: bool = True,
        ) -> None:
            super().__init__()
            self.use_timing = use_timing
            self.embed = nn.Linear(in_features, d_model)
            # TODO(Phase 2): space-time positional encoding + kernel attention
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):  # noqa: D102
            raise NotImplementedError("Implement forward pass in Phase 2.")
