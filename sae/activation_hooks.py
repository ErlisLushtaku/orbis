"""
Hook-based activation extraction from Orbis ST-Transformer layers.

This module provides utilities to extract intermediate activations from
the STDiT (Spatial-Temporal Diffusion Transformer) blocks using forward hooks.
"""

from typing import Callable, List, Optional, Tuple, Dict, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
from einops import rearrange


class ActivationExtractor:
    """
    Extract activations from specific layers of the Orbis ST-Transformer.
    
    Usage:
        extractor = ActivationExtractor(model, layer_idx=12)
        
        with extractor.capture():
            output = model(input)
        
        activations = extractor.get_activations()
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 12,
        flatten_spatial: bool = True,
    ):
        """
        Initialize the activation extractor.
        
        Args:
            model: The Orbis world model (with model.vit as the ST-Transformer)
            layer_idx: Which transformer block to extract from (0-indexed)
            flatten_spatial: If True, flatten (B, F, N, D) -> (B*F*N, D) for SAE
        """
        self.model = model
        self.layer_idx = layer_idx
        self.flatten_spatial = flatten_spatial
        
        self._activations: List[torch.Tensor] = []
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        
        # Get the target block
        self._target_block = self._get_target_block()
    
    def _get_target_block(self) -> nn.Module:
        """Get the transformer block at the specified layer index."""
        # Orbis model structure: model.vit.blocks[layer_idx]
        if hasattr(self.model, 'vit'):
            vit = self.model.vit
        else:
            vit = self.model
        
        if not hasattr(vit, 'blocks'):
            raise ValueError("Model does not have 'blocks' attribute. Expected STDiT architecture.")
        
        if self.layer_idx >= len(vit.blocks):
            raise ValueError(
                f"Layer index {self.layer_idx} out of range. "
                f"Model has {len(vit.blocks)} blocks (0-{len(vit.blocks)-1})."
            )
        
        return vit.blocks[self.layer_idx]
    
    def _hook_fn(self, module: nn.Module, input: Tuple, output: torch.Tensor):
        """Hook function to capture activations."""
        # STDiTBlock output shape: (B, F, N, D)
        # where B=batch, F=frames, N=spatial tokens, D=hidden_dim
        acts = output.detach()
        
        if self.flatten_spatial:
            # Flatten to (B*F*N, D) for SAE training
            acts = rearrange(acts, 'b f n d -> (b f n) d')
        
        self._activations.append(acts)
    
    @contextmanager
    def capture(self):
        """Context manager to capture activations during forward pass."""
        self._activations = []
        self._handle = self._target_block.register_forward_hook(self._hook_fn)
        try:
            yield self
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
    
    def get_activations(self) -> torch.Tensor:
        """
        Get captured activations.
        
        Returns:
            Concatenated activations from all forward passes during capture.
            Shape: (total_tokens, hidden_dim) if flatten_spatial=True
                   else list of (B, F, N, D) tensors
        """
        if not self._activations:
            raise RuntimeError("No activations captured. Use within capture() context.")
        
        if self.flatten_spatial:
            return torch.cat(self._activations, dim=0)
        else:
            return self._activations
    
    def clear(self):
        """Clear stored activations."""
        self._activations = []


class ActivationIntervenor:
    """
    Intervene on activations by replacing them with SAE reconstructions.
    
    Used for computing "Loss Recovered" metric - measuring how well SAE
    reconstructions preserve model behavior.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sae: nn.Module,
        layer_idx: int = 12,
    ):
        """
        Initialize the activation interventor.
        
        Args:
            model: The Orbis world model
            sae: Trained SAE model with encode/decode methods
            layer_idx: Which transformer block to intervene on
        """
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._target_block = self._get_target_block()
    
    def _get_target_block(self) -> nn.Module:
        """Get the transformer block at the specified layer index."""
        if hasattr(self.model, 'vit'):
            vit = self.model.vit
        else:
            vit = self.model
        
        return vit.blocks[self.layer_idx]
    
    def _intervention_hook(
        self, 
        module: nn.Module, 
        input: Tuple, 
        output: torch.Tensor
    ) -> torch.Tensor:
        """Hook function that replaces activations with SAE reconstructions."""
        # output shape: (B, F, N, D)
        original_shape = output.shape
        device = output.device
        dtype = output.dtype
        
        # Flatten for SAE
        flat_acts = rearrange(output, 'b f n d -> (b f n) d')
        
        # Pass through SAE (encode -> decode)
        self.sae.eval()
        with torch.no_grad():
            reconstructed, _, _, _, _ = self.sae(flat_acts.to(self.sae.W_enc.device))
            reconstructed = reconstructed.to(device=device, dtype=dtype)
        
        # Reshape back
        reconstructed = rearrange(
            reconstructed, 
            '(b f n) d -> b f n d',
            b=original_shape[0],
            f=original_shape[1],
            n=original_shape[2],
        )
        
        return reconstructed
    
    @contextmanager
    def intervene(self):
        """Context manager to apply SAE intervention during forward pass."""
        self._handle = self._target_block.register_forward_hook(self._intervention_hook)
        try:
            yield self
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None


class ZeroAblationIntervenor:
    """
    Intervene on activations by replacing them with zeros.
    
    Used for computing the zero-ablation baseline (L_zero) in the 
    Normalized Loss Recovered metric:
        NLR = (L_zero - L_sae) / (L_zero - L_base)
    
    This represents the worst-case scenario where all information from
    the target layer is removed, providing an upper bound on error.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 12,
    ):
        """
        Initialize the zero ablation interventor.
        
        Args:
            model: The Orbis world model
            layer_idx: Which transformer block to intervene on
        """
        self.model = model
        self.layer_idx = layer_idx
        
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._target_block = self._get_target_block()
    
    def _get_target_block(self) -> nn.Module:
        """Get the transformer block at the specified layer index."""
        if hasattr(self.model, 'vit'):
            vit = self.model.vit
        else:
            vit = self.model
        
        return vit.blocks[self.layer_idx]
    
    def _zero_hook(
        self, 
        module: nn.Module, 
        input: Tuple, 
        output: torch.Tensor
    ) -> torch.Tensor:
        """Hook function that replaces activations with zeros."""
        # Return zeros with same shape, dtype, and device as output
        return torch.zeros_like(output)
    
    @contextmanager
    def intervene(self):
        """Context manager to apply zero ablation during forward pass."""
        self._handle = self._target_block.register_forward_hook(self._zero_hook)
        try:
            yield self
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None


def extract_activations_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    device: torch.device,
    max_batches: Optional[int] = None,
    t_noise: float = 0.0,
    frame_rate: int = 5,
) -> torch.Tensor:
    """
    Extract activations from multiple batches.
    
    Args:
        model: Orbis world model
        dataloader: DataLoader yielding (images, labels) or similar
        layer_idx: Which transformer block to extract from
        device: Device to run inference on
        max_batches: Maximum number of batches to process (None = all)
        t_noise: Noise timestep for denoising (0.0 = clean)
        frame_rate: Frame rate conditioning value
        
    Returns:
        Concatenated activations tensor of shape (total_tokens, hidden_dim)
    """
    model.eval()
    extractor = ActivationExtractor(model, layer_idx=layer_idx, flatten_spatial=True)
    
    all_activations = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            elif isinstance(batch, dict):
                imgs = batch.get('images', batch.get('image'))
            else:
                imgs = batch
            
            imgs = imgs.to(device)
            
            # Encode frames through tokenizer
            x = model.encode_frames(imgs)
            
            # Prepare for denoising step
            b = x.shape[0]
            t = torch.full((b,), t_noise, device=device)
            
            # Add noise if t > 0
            if t_noise > 0:
                target_t, _ = model.add_noise(x, t)
            else:
                target_t = x
            
            # Ensure correct shape for vit input
            if target_t.dim() == 4:
                target_t = target_t.unsqueeze(1)
            
            # Run through transformer with activation capture
            fr = torch.full((b,), frame_rate, device=device)
            
            with extractor.capture():
                # Context is previous frames (if multi-frame)
                if target_t.shape[1] > 1:
                    context = target_t[:, :-1]
                    target = target_t[:, -1:]
                else:
                    context = None
                    target = target_t
                
                _ = model.vit(target, context, t, frame_rate=fr)
            
            activations = extractor.get_activations()
            all_activations.append(activations.cpu())
    
    return torch.cat(all_activations, dim=0)
