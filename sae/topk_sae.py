"""
Trainable Top-K Sparse Autoencoder for Orbis world model activations.

This implementation is adapted from SAEBench's TopKSAE but modified for training
with MSE loss only (no sparsity penalty) and decoder weight normalization.
"""

import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TopKSAEConfig:
    """Configuration for TopK SAE."""
    d_in: int  # Input dimension (hidden_size of ST-Transformer, e.g., 768)
    expansion_factor: int = 16  # SAE width = d_in * expansion_factor
    k: int = 64  # Number of active features (top-k)
    
    # Derived
    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion_factor


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder.
    
    Architecture:
        - Encoder: Linear(d_in -> d_sae) + ReLU + Top-K selection
        - Decoder: Linear(d_sae -> d_in) with unit-norm columns
    
    Training uses MSE reconstruction loss only (no L1 sparsity penalty).
    Decoder weights are normalized after each optimizer step.
    """
    
    def __init__(self, config: TopKSAEConfig):
        super().__init__()
        self.config = config
        self.d_in = config.d_in
        self.d_sae = config.d_sae
        self.k = config.k
        
        # Encoder weights and biases
        self.W_enc = nn.Parameter(torch.empty(self.d_in, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        
        # Decoder weights and bias
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_in))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming uniform for encoder, orthogonal for decoder."""
        # Encoder: Kaiming uniform
        nn.init.kaiming_uniform_(self.W_enc, a=0, mode='fan_in', nonlinearity='relu')
        
        # Decoder: Initialize with normalized random vectors
        nn.init.normal_(self.W_dec, mean=0, std=1.0 / self.d_in**0.5)
        self.normalize_decoder_weights()
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize decoder weight columns to unit norm."""
        norms = torch.norm(self.W_dec, dim=1, keepdim=True)
        self.W_dec.data = self.W_dec.data / (norms + 1e-8)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse feature activations.
        
        Args:
            x: Input tensor of shape (..., d_in)
            
        Returns:
            Sparse feature activations of shape (..., d_sae) with at most k non-zero entries
        """
        # Center input
        x_centered = x - self.b_dec
        
        # Linear projection + ReLU
        pre_acts = F.relu(x_centered @ self.W_enc + self.b_enc)
        
        # Top-K selection
        topk_values, topk_indices = pre_acts.topk(self.k, dim=-1, sorted=False)
        
        # Create sparse output
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(dim=-1, index=topk_indices, src=topk_values)
        
        return sparse_acts
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse feature activations back to input space.
        
        Args:
            feature_acts: Sparse activations of shape (..., d_sae)
            
        Returns:
            Reconstructed activations of shape (..., d_in)
        """
        return feature_acts @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (..., d_in)
            
        Returns:
            Tuple of (reconstruction, sparse_activations)
        """
        sparse_acts = self.encode(x)
        reconstruction = self.decode(sparse_acts)
        return reconstruction, sparse_acts
    
    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations without gradient tracking."""
        return self.encode(x)
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "TopKSAE":
        """Load model from checkpoint."""
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
            f"expansion={self.config.expansion_factor}x)"
        )


class TopKSAETrainer:
    """
    Trainer for TopK SAE with decoder weight normalization and metric tracking.
    
    Optimized for RTX 2080 (Turing) with:
        - HuggingFace Accelerate for fp16 mixed precision
        - Fused AdamW optimizer for reduced kernel launches
        - torch.compile (default mode) for optimized execution
        - non_blocking transfers for overlapped data movement
    """
    
    def __init__(
        self,
        model: TopKSAE,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device = None,  # Note: accelerator.device will be used
        dead_feature_window: int = 1000,  # Reset dead feature stats every N steps
        compile_model: bool = False,  # Use torch.compile (PyTorch 2.0+)
    ):
        # Initialize Accelerator with fp16 (NOT bf16 - Turing doesn't support it natively)
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.device = self.accelerator.device
        
        # Move model to device before compilation
        self.model = model.to(self.device)
        
        # Optionally compile model for faster execution (default mode for Turing stability)
        if compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile (default mode)...")
            compile_start = time.perf_counter()
            self.model = torch.compile(self.model)
            compile_time = time.perf_counter() - compile_start
            logger.info(f"Model compilation completed in {compile_time:.2f}s")
        
        # Fused AdamW for reduced kernel launch overhead (supported on Turing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        
        # Prepare model and optimizer with accelerate (handles gradient scaling internally)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        # Metric Tracking State
        self.dead_feature_window = dead_feature_window
        self.step_counter = 0
        self.feature_hits = torch.zeros(model.d_sae, device=self.device)
        self.current_dead_pct = 0.0
    
    def calculate_metrics(self, x: torch.Tensor, reconstruction: torch.Tensor, sparse_acts: torch.Tensor) -> dict:
        """
        Compute Phase 1 training metrics for SAE quality monitoring.
        
        Metrics computed:
            - loss (MSE): Mean squared error between input and reconstruction
            - rel_error: Relative reconstruction error ||x - x_hat|| / ||x||
            - explained_variance (R²): 1 - Var(residual) / Var(input)
            - cos_sim: Mean cosine similarity between input and reconstruction
            - l0: Average number of active features per sample (fixed at k for Top-K)
            - activation_density: Percentage of dictionary active per sample (l0/d_sae * 100)
            - l1_norm: Total magnitude of sparse activations (monitors energy drift)
            - dead_pct: Rolling window percentage of features with zero activations
        """
        with torch.no_grad():
            # MSE Loss
            mse = F.mse_loss(reconstruction, x).item()
            
            # L0: Average number of active features per sample
            l0 = (sparse_acts > 0).float().sum(dim=-1).mean().item()
            
            # Cosine Similarity
            cos_sim = F.cosine_similarity(reconstruction, x, dim=-1).mean().item()
            
            # Relative Reconstruction Error: ||x - x_hat|| / ||x||
            rel_error = ((reconstruction - x).norm(dim=-1) / (x.norm(dim=-1) + 1e-8)).mean().item()
            
            # Explained Variance (R²): 1 - Var(residual) / Var(input)
            residual = x - reconstruction
            residual_var = (residual ** 2).mean()
            # Compute variance as E[(x - E[x])^2] using batch mean
            x_centered = x - x.mean(dim=0, keepdim=True)
            total_var = (x_centered ** 2).mean()
            explained_variance = (1.0 - (residual_var / (total_var + 1e-8))).item()
            
            # Activation Density: k/d_sae as percentage
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            activation_density = (l0 / unwrapped_model.d_sae) * 100.0
            
            # L1 Norm: Total energy of sparse activations (monitors feature explosion)
            l1_norm = sparse_acts.abs().sum(dim=-1).mean().item()
            
            # Dead Feature Tracking (Only update statistics during training)
            if self.model.training:
                batch_hits = (sparse_acts > 0).any(dim=0).float()
                self.feature_hits += batch_hits
                self.step_counter += 1
                
                # Periodically calculate dead % and reset counter
                if self.step_counter % self.dead_feature_window == 0:
                    dead_count = (self.feature_hits == 0).sum().item()
                    self.current_dead_pct = (dead_count / unwrapped_model.d_sae) * 100.0
                    self.feature_hits.zero_()  # Reset for next window

        return {
            "loss": mse,
            "l0": l0,
            "cos_sim": cos_sim,
            "rel_error": rel_error,
            "explained_variance": explained_variance,
            "activation_density": activation_density,
            "l1_norm": l1_norm,
            "dead_pct": self.current_dead_pct,
        }

    def train_step(self, batch: torch.Tensor) -> dict:
        """
        Perform a single training step with fp16 mixed precision via Accelerate.
        """
        self.model.train()
        batch = batch.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        
        # Accelerate handles autocast and gradient scaling internally
        with self.accelerator.autocast():
            reconstruction, sparse_acts = self.model(batch)
            loss = F.mse_loss(reconstruction, batch)
        
        # Accelerate handles gradient scaling internally
        self.accelerator.backward(loss)
        self.optimizer.step()
        
        # Normalize decoder weights (full precision, outside autocast)
        # Use unwrap_model to access custom method through accelerate wrapper
        self.accelerator.unwrap_model(self.model).normalize_decoder_weights()
        
        # Compute Metrics (detached, no gradient tracking)
        metrics = self.calculate_metrics(batch, reconstruction.float(), sparse_acts.float())
        metrics["loss"] = loss.item()
        
        return metrics
    
    @torch.no_grad()
    def eval_step(self, batch: torch.Tensor) -> dict:
        """
        Perform evaluation step without gradient updates.
        """
        self.model.eval()
        batch = batch.to(self.device, non_blocking=True)
        
        reconstruction, sparse_acts = self.model(batch)
        
        # Calculate all metrics (without updating dead feature stats)
        metrics = self.calculate_metrics(batch, reconstruction, sparse_acts)
        
        return metrics
