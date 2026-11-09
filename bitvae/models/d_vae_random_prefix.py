"""
Standard random-prefix Stage-1 tokenizer for the controlled comparison.

This file intentionally keeps the existing Infinity/BitVAE AutoEncoder and
MultiScaleBSQ unchanged.  The ONLY training-policy change is the prefix
sampling rule used by the main reconstruction branch:

    k ~ Uniform({1, ..., K})
    z_decode = sum_{i=1}^k R_i

where K is the number of scales available for the current dynamic-resolution
sample.

Not enabled by this model:
- Infinity stochastic scale dropout (the bash does not enable it)
- ARPC-style SRD
- BAT's selected truncation set / p_full sampling
- MSD/DINO (the bash sets dino_weight=0)
- PBSC (Stage-2 method)
"""

import torch

from bitvae.models.d_vae_prior import AutoEncoder as InfinityAutoEncoder


class AutoEncoder(InfinityAutoEncoder):
    """Infinity AutoEncoder with standard uniform random-prefix training."""

    def __init__(self, args):
        # Build the existing model without changing Encoder / Decoder /
        # MultiScaleBSQ / losses / optimizer-facing parameters.
        super().__init__(args)

        # Force the existing main prefix-reconstruction path to be active.
        # The base forward already:
        #   1) obtains per-scale full-resolution residual contributions,
        #   2) constructs cumulative prefix latents,
        #   3) calls _sample_prefix_scale(),
        #   4) sends the sampled prefix to the same decoder/loss branch.
        self.coarse_prefix_sample = True

        # The base forward sanitizes this list against the actual K of the
        # current sample.  Supplying a sufficiently long 1-based sequence
        # therefore yields exactly [1, ..., K] for dynamic-resolution training.
        self.coarse_prefix_scales = list(range(1, 257))

    def _sample_prefix_scale(
        self,
        candidate_scales,
        full_k: int,
        device,
        full_prob: float = 0.5,
    ):
        """Sample a contiguous prefix uniformly from all available scales.

        `candidate_scales` and `full_prob` are accepted only to preserve the
        base-class call signature.  Standard random-prefix training uses:

            P(k) = 1 / K,  k in {1, ..., K}

        including the full-scale prefix k=K with the same probability 1/K.
        """
        K = int(full_k)
        if K < 1:
            raise ValueError(f"full_k must be >= 1, got {K}")

        probs = torch.full(
            (K,),
            1.0 / float(K),
            device=device,
            dtype=torch.float32,
        )

        sampled_k = int(
            torch.randint(
                low=1,
                high=K + 1,
                size=(1,),
                device=device,
            ).item()
        )
        return sampled_k, probs

    @staticmethod
    def add_model_specific_args(parent_parser):
        # No new hyperparameters are introduced for this baseline.
        # Reuse the exact existing Infinity-style argument set.
        return InfinityAutoEncoder.add_model_specific_args(parent_parser)
