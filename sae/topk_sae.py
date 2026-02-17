"""
Top-K Sparse Autoencoder for Orbis world model activations.

Training recipe based on SAELens/OpenAI (Gao et al., "Scaling and Evaluating
Sparse Autoencoders"):
- Tied encoder-decoder initialization (W_enc = W_dec.T) with small decoder norms
- rescale_acts_by_decoder_norm: top-k selection invariant to decoder row norms
- Standard MSE loss (sum over features, mean over batch)
- Auxiliary TopK loss for dead-feature revival
- Cosine annealing with warmup LR schedule, fp32, Adam optimizer
- Geometric median b_dec initialization
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometric_median import compute_geometric_median
from .lr_scheduler import get_scheduler
from .utils.constants import EPSILON
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopKSAEConfig:
    """Configuration for TopK SAE."""

    d_in: int
    expansion_factor: int = 16
    k: int = 64

    dead_feature_window: int = 1000
    aux_loss_coefficient: float = 1.0
    decoder_init_norm: Optional[float] = 0.1
    rescale_acts_by_decoder_norm: bool = True
    normalize_activations: str = "none"  # "none" or "layer_norm"
    b_dec_init_method: str = "geometric_median"  # "geometric_median", "mean", "zeros"

    # Kept for checkpoint backward compatibility; not used in training
    use_ghost_grads: bool = False

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion_factor


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder.

    Architecture:
        Encoder: Linear(d_in -> d_sae) + ReLU + Top-K selection
        Decoder: Linear(d_sae -> d_in)

    Initialization (from Anthropic April 2024 / SAELens):
        W_dec: Kaiming uniform, rows normalized to decoder_init_norm (default 0.1)
        W_enc: W_dec.T (tied initialization)
        b_enc, b_dec: zeros (b_dec later set via geometric median)

    When rescale_acts_by_decoder_norm=True, top-k selection is invariant to
    decoder row norms without explicit per-step normalization or gradient
    orthogonalization. Norms can be folded post-training via fold_W_dec_norm.

    Loss: standard MSE + auxiliary TopK loss for dead features.
    """

    def __init__(self, config: TopKSAEConfig):
        super().__init__()
        self.config = config
        self.d_in = config.d_in
        self.d_sae = config.d_sae
        self.k = config.k

        self.W_enc = nn.Parameter(torch.empty(self.d_in, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_in))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        self._init_weights()

    def _init_weights(self):
        # W_dec: Kaiming uniform, then normalize rows to decoder_init_norm
        nn.init.kaiming_uniform_(self.W_dec)
        if self.config.decoder_init_norm is not None:
            with torch.no_grad():
                self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)
                self.W_dec.data *= self.config.decoder_init_norm
        # W_enc: tied to W_dec.T (encoder-decoder start in same subspace)
        self.W_enc.data = self.W_dec.data.T.clone().detach().contiguous()

    # ------------------------------------------------------------------
    # Decoder norm utilities (for checkpoint conversion / inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize each decoder row to unit L2 norm."""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Project out the gradient component parallel to W_dec rows."""
        if self.W_dec.grad is None:
            return
        parallel_component = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel_component * self.W_dec.data

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Fold decoder norms into encoder weights and bias for inference.

        After folding, decoder rows become unit-norm and the rescaling is
        permanently absorbed into W_enc and b_enc, so encode/decode no longer
        need runtime norm computation.

        Only valid when rescale_acts_by_decoder_norm=True.
        """
        if not self.config.rescale_acts_by_decoder_norm:
            raise ValueError(
                "fold_W_dec_norm is only valid when rescale_acts_by_decoder_norm=True"
            )
        W_dec_norm = self.W_dec.norm(dim=-1).clamp(min=1e-8)
        self.b_enc.data *= W_dec_norm
        W_dec_norms = W_dec_norm.unsqueeze(1)
        self.W_dec.data /= W_dec_norms
        self.W_enc.data *= W_dec_norms.T

    # ------------------------------------------------------------------
    # b_dec initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_b_dec(self, all_activations: torch.Tensor):
        """Initialize b_dec from a batch of training activations."""
        if self.config.b_dec_init_method == "geometric_median":
            self._initialize_b_dec_geometric_median(all_activations)
        elif self.config.b_dec_init_method == "mean":
            self._initialize_b_dec_mean(all_activations)
        elif self.config.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unknown b_dec_init_method: {self.config.b_dec_init_method}")

    @torch.no_grad()
    def _initialize_b_dec_geometric_median(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()

        result = compute_geometric_median(all_activations.cpu(), maxiter=100)
        median = result.median

        previous_distances = torch.norm(all_activations.cpu() - previous_b_dec, dim=-1)
        new_distances = torch.norm(all_activations.cpu() - median, dim=-1)

        logger.info("Initializing b_dec with geometric median of activations")
        logger.info(f"  Previous median distance: {previous_distances.median().item():.4f}")
        logger.info(f"  New median distance:      {new_distances.median().item():.4f}")
        logger.info(f"  Termination: {result.termination}")

        self.b_dec.data = median.to(dtype=self.b_dec.dtype, device=self.b_dec.device)

    @torch.no_grad()
    def _initialize_b_dec_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        mean = all_activations.mean(dim=0).cpu()

        previous_distances = torch.norm(all_activations.cpu() - previous_b_dec, dim=-1)
        new_distances = torch.norm(all_activations.cpu() - mean, dim=-1)

        logger.info("Initializing b_dec with mean of activations")
        logger.info(f"  Previous median distance: {previous_distances.median().item():.4f}")
        logger.info(f"  New median distance:      {new_distances.median().item():.4f}")

        self.b_dec.data = mean.to(dtype=self.b_dec.dtype, device=self.b_dec.device)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _compute_mse_loss(self, x: torch.Tensor, sae_out: torch.Tensor) -> torch.Tensor:
        """Standard MSE: sum over feature dim, mean over batch."""
        return F.mse_loss(sae_out, x, reduction="none").sum(dim=-1).mean()

    def _compute_aux_topk_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary TopK loss for dead-feature revival.

        From Gao et al. "Scaling and Evaluating Sparse Autoencoders" (Appendix B.1).
        Dead features get their own top-k selection over the residual, so they
        don't compete with alive features.
        """
        num_dead = int(dead_neuron_mask.sum())
        if num_dead == 0:
            return torch.tensor(0.0, device=x.device)

        residual = (x - sae_out).detach()

        # k_aux = d_in // 2 (heuristic from Appendix B.1)
        k_aux = self.d_in // 2
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Select top-k_aux among dead features only (alive set to -inf)
        auxk_latents = torch.where(
            dead_neuron_mask[None], hidden_pre, torch.tensor(-torch.inf, device=x.device)
        )
        auxk_topk = auxk_latents.topk(k_aux, sorted=False)

        auxk_acts = torch.zeros_like(hidden_pre)
        auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)

        recons = self.decode(auxk_acts)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()

        return self.config.aux_loss_coefficient * scale * auxk_loss

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input activations to sparse feature activations.

        When rescale_acts_by_decoder_norm=True, pre-activations are scaled
        by decoder row norms so that top-k selection is invariant to decoder
        norm magnitude.

        Returns:
            Tuple of (sparse_acts, hidden_pre).
        """
        if self.config.normalize_activations == "layer_norm":
            x = F.layer_norm(x, (self.d_in,))

        x_centered = x - self.b_dec
        hidden_pre = x_centered @ self.W_enc + self.b_enc

        if self.config.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        post_relu = F.relu(hidden_pre)
        topk_values, topk_indices = post_relu.topk(self.k, dim=-1, sorted=False)

        sparse_acts = torch.zeros_like(hidden_pre)
        sparse_acts.scatter_(dim=-1, index=topk_indices, src=topk_values)

        return sparse_acts, hidden_pre

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse feature activations back to input space.

        When rescale_acts_by_decoder_norm=True, feature activations are
        divided by decoder row norms to undo the encode-time rescaling.
        """
        if self.config.rescale_acts_by_decoder_norm:
            feature_acts = feature_acts / self.W_dec.norm(dim=-1)
        return feature_acts @ self.W_dec + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, decode, and compute losses.

        Returns:
            Tuple of (reconstruction, sparse_acts, loss, mse_loss, aux_loss).
        """
        sparse_acts, hidden_pre = self.encode(x)
        reconstruction = self.decode(sparse_acts)

        mse_loss = self._compute_mse_loss(x, reconstruction)
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.training and dead_neuron_mask is not None:
            aux_loss = self._compute_aux_topk_loss(
                x, reconstruction, hidden_pre, dead_neuron_mask
            )

        loss = mse_loss + aux_loss
        return reconstruction, sparse_acts, loss, mse_loss, aux_loss

    # ------------------------------------------------------------------
    # Inference / persistence
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations without gradient tracking."""
        sparse_acts, _ = self.encode(x)
        return sparse_acts

    def save(self, path: str):
        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "TopKSAE":
        checkpoint = torch.load(path, map_location=device or "cpu")
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        if device:
            model = model.to(device)
        return model

    def __repr__(self) -> str:
        return (
            f"TopKSAE(d_in={self.d_in}, d_sae={self.d_sae}, k={self.k}, "
            f"expansion={self.config.expansion_factor}x, "
            f"aux_coeff={self.config.aux_loss_coefficient}, "
            f"rescale_dec_norm={self.config.rescale_acts_by_decoder_norm})"
        )


class TopKSAETrainer:
    """
    Trainer for TopK SAE.

    Training recipe:
        - fp32 training (no mixed precision)
        - Adam optimizer (no weight decay needed for SAEs)
        - Cosine annealing with warmup LR schedule
        - Dead feature tracking via n_forward_passes_since_fired
        - Auxiliary TopK loss for dead-feature revival
        - Train step order: forward -> backward -> clip -> step -> schedule
    """

    def __init__(
        self,
        model: TopKSAE,
        lr: float = 1e-3,
        device: torch.device = None,
        compile_model: bool = False,
        max_grad_norm: float = 1.0,
        total_training_steps: int = 0,
        lr_warmup_steps: int = 500,
        lr_end_factor: float = 0.1,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)

        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile (default mode)...")
            compile_start = time.perf_counter()
            self.model = torch.compile(self.model)
            compile_time = time.perf_counter() - compile_start
            logger.info(f"Model compilation completed in {compile_time:.2f}s")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.total_training_steps = total_training_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_end_factor = lr_end_factor

        if total_training_steps > 0:
            self.scheduler = get_scheduler(
                "cosineannealingwarmup",
                optimizer=self.optimizer,
                warm_up_steps=lr_warmup_steps,
                training_steps=total_training_steps,
                lr_end=lr_end_factor,
            )
        else:
            self.scheduler = get_scheduler("constant", optimizer=self.optimizer)

        # Dead feature tracking
        self.n_forward_passes_since_fired = torch.zeros(
            model.d_sae, device=self.device, dtype=torch.long
        )

    def _get_dead_neuron_mask(self) -> torch.Tensor:
        """Return boolean mask of features that haven't fired in dead_feature_window steps."""
        unwrapped = self._unwrap_model()
        return (self.n_forward_passes_since_fired > unwrapped.config.dead_feature_window).bool()

    def _unwrap_model(self) -> TopKSAE:
        """Get underlying TopKSAE even if wrapped by torch.compile."""
        if hasattr(self.model, "_orig_mod"):
            return self.model._orig_mod
        return self.model

    def _update_dead_feature_tracking(self, sparse_acts: torch.Tensor):
        """Update per-feature firing tracker."""
        with torch.no_grad():
            did_fire = (sparse_acts > 0).float().sum(dim=0) > 0
            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire] = 0

    def initialize_b_dec(self, dataloader, num_tokens: int = 50000):
        """Initialize b_dec from training data before the training loop."""
        unwrapped = self._unwrap_model()
        if unwrapped.config.b_dec_init_method == "zeros":
            logger.info("b_dec init: zeros (no initialization needed)")
            return

        logger.info(f"Collecting up to {num_tokens:,} tokens for b_dec initialization...")
        all_acts = []
        collected = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device=self.device, dtype=torch.float32)
            all_acts.append(batch)
            collected += batch.shape[0]
            if collected >= num_tokens:
                break

        all_acts = torch.cat(all_acts, dim=0)[:num_tokens]
        logger.info(f"Collected {all_acts.shape[0]:,} tokens for b_dec initialization")
        unwrapped.initialize_b_dec(all_acts)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_metrics(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        sparse_acts: torch.Tensor,
        mse_loss: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute training metrics for SAE quality monitoring."""
        with torch.no_grad():
            unwrapped = self._unwrap_model()
            l0 = (sparse_acts > 0).float().sum(dim=-1).mean().item()
            cos_sim = F.cosine_similarity(reconstruction, x, dim=-1).mean().item()
            rel_error = (
                ((reconstruction - x).norm(dim=-1) / (x.norm(dim=-1) + EPSILON))
                .mean()
                .item()
            )

            residual = x - reconstruction
            residual_var = (residual**2).mean()
            x_centered = x - x.mean(dim=0, keepdim=True)
            total_var = (x_centered**2).mean()
            explained_variance = (1.0 - (residual_var / (total_var + EPSILON))).item()

            activation_density = (l0 / unwrapped.d_sae) * 100.0
            l1_norm = sparse_acts.abs().sum(dim=-1).mean().item()

            dead_mask = self._get_dead_neuron_mask()
            dead_pct = (dead_mask.sum().item() / unwrapped.d_sae) * 100.0

        return {
            "loss": (mse_loss + aux_loss).item(),
            "mse_loss": mse_loss.item(),
            "ghost_loss": aux_loss.item(),
            "l0": l0,
            "cos_sim": cos_sim,
            "rel_error": rel_error,
            "explained_variance": explained_variance,
            "activation_density": activation_density,
            "l1_norm": l1_norm,
            "dead_pct": dead_pct,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------
    # Train / eval steps
    # ------------------------------------------------------------------

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step.

        Step order (SAELens style, no per-step decoder norm management):
            1. Forward with dead_neuron_mask
            2. Update dead feature tracking
            3. loss.backward()
            4. Gradient clipping
            5. optimizer.step()
            6. scheduler.step()
        """
        self.model.train()
        batch = batch.to(device=self.device, dtype=torch.float32, non_blocking=True)

        # 1. Forward with dead neuron mask
        self.optimizer.zero_grad()
        dead_neuron_mask = self._get_dead_neuron_mask()
        reconstruction, sparse_acts, loss, mse_loss, aux_loss = self.model(
            batch, dead_neuron_mask
        )

        # 2. Update dead feature tracking
        self._update_dead_feature_tracking(sparse_acts)

        # Skip step if loss is non-finite
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss ({loss.item():.4f}), skipping step")
            self.optimizer.zero_grad()
            metrics = self.calculate_metrics(
                batch, reconstruction, sparse_acts, mse_loss, aux_loss
            )
            metrics["loss"] = float("nan")
            return metrics

        # 3. Backward
        loss.backward()

        # 4. Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )

        # 5. Optimizer step
        self.optimizer.step()

        # 6. Scheduler step
        self.scheduler.step()

        metrics = self.calculate_metrics(
            batch, reconstruction, sparse_acts, mse_loss, aux_loss
        )
        return metrics

    @torch.no_grad()
    def eval_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Evaluation step without gradient updates."""
        self.model.eval()
        batch = batch.to(device=self.device, dtype=torch.float32, non_blocking=True)

        reconstruction, sparse_acts, loss, mse_loss, aux_loss = self.model(batch)
        metrics = self.calculate_metrics(
            batch, reconstruction, sparse_acts, mse_loss, aux_loss
        )
        return metrics
