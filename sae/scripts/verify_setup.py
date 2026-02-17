#!/usr/bin/env python3
"""
Verification script to ensure SAE training pipeline is correctly set up.

Run this script to check:
1. All modules import correctly
2. SAE can be instantiated with correct dimensions
3. Forward/backward passes work
4. Hooks can be attached to a dummy model
5. Decoder weights are strictly normalized (SAEBench standard)

Usage:
    python orbis/sae/scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root and orbis root to path (scripts are in sae/scripts/, so go up 3 levels for project, 2 for orbis)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORBIS_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIS_ROOT))


def test_imports():
    """Test that all modules import correctly."""
    print("[test] Testing imports...")
    
    try:
        from sae import (
            TopKSAE,
            TopKSAEConfig,
            TopKSAETrainer,
            ActivationExtractor,
            ActivationIntervenor,
            prepare_activation_cache,
            load_activation_cache,
            create_activation_dataloader,
            compute_loss_recovered,
            compute_dead_features,
            compute_activation_density,
        )
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_sae_creation():
    """Test SAE instantiation."""
    print("[test] Testing SAE creation...")
    
    try:
        import torch
        from sae import TopKSAE, TopKSAEConfig
        
        # Orbis config: hidden_size=768
        config = TopKSAEConfig(
            d_in=768,
            expansion_factor=16,
            k=64,
        )
        
        sae = TopKSAE(config)
        
        print(f"  [OK] Created SAE: {sae}")
        print(f"       d_in={config.d_in}, d_sae={config.d_sae}, k={config.k}")
        
        # Check weight shapes
        assert sae.W_enc.shape == (768, 768 * 16), f"W_enc shape mismatch"
        assert sae.W_dec.shape == (768 * 16, 768), f"W_dec shape mismatch"
        print("  [OK] Weight shapes correct")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_forward_backward():
    """Test forward and backward passes."""
    print("[test] Testing forward/backward passes...")
    
    try:
        import torch
        import torch.nn.functional as F  # <--- Make sure F is imported
        from sae import TopKSAE, TopKSAEConfig
        
        config = TopKSAEConfig(d_in=768, expansion_factor=16, k=64)
        sae = TopKSAE(config)
        
        # Random input batch
        x = torch.randn(32, 768)  # (batch, d_in)
        
        # Forward pass
        reconstruction, sparse_acts, _, _, _ = sae(x)
        
        assert reconstruction.shape == x.shape, "Reconstruction shape mismatch"
        assert sparse_acts.shape == (32, 768 * 16), "Sparse acts shape mismatch"
        print("  [OK] Forward pass shapes correct")
        
        # Check sparsity
        active = (sparse_acts > 0).sum(dim=1).float().mean()
        print(f"       Average L0: {active.item():.1f} (expected ~64)")
        
        # Backward pass (Manually compute loss since sae.loss is gone)
        loss = F.mse_loss(reconstruction, x)  # <--- CHANGED
        loss.backward()
        
        assert sae.W_enc.grad is not None, "No gradient for W_enc"
        print("  [OK] Backward pass successful")
        print(f"       Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decoder_normalization():
    """Test decoder initialization and fold_W_dec_norm."""
    print("[test] Testing decoder initialization and fold_W_dec_norm...")
    
    try:
        import torch
        from sae import TopKSAE, TopKSAEConfig
        
        config = TopKSAEConfig(d_in=768, expansion_factor=16, k=64)
        sae = TopKSAE(config)
        
        # 1. Check init: decoder rows should have decoder_init_norm (0.1)
        norms = torch.norm(sae.W_dec, dim=1)
        target_norm = config.decoder_init_norm
        target = torch.full_like(norms, target_norm)
        max_diff = torch.max(torch.abs(norms - target)).item()
        
        print(f"       [Init] Max deviation from {target_norm} norm: {max_diff:.9f}")
        assert max_diff < 1e-5, f"Init norms wrong! Max diff: {max_diff}"
        print(f"  [OK] Initial decoder norms = {target_norm}")
        
        # 2. Check tied init: W_enc should be W_dec.T
        assert torch.allclose(sae.W_enc.data, sae.W_dec.data.T, atol=1e-6), \
            "W_enc should be W_dec.T after initialization"
        print("  [OK] W_enc = W_dec.T (tied initialization)")
        
        # 3. Test fold_W_dec_norm: forward pass should be unchanged
        x = torch.randn(8, 768)
        out_before, _, _, _, _ = sae(x)
        
        sae.fold_W_dec_norm()
        
        # After folding, decoder rows should be unit-norm
        folded_norms = torch.norm(sae.W_dec, dim=1)
        unit_target = torch.ones_like(folded_norms)
        max_diff_folded = torch.max(torch.abs(folded_norms - unit_target)).item()
        assert max_diff_folded < 1e-5, f"Folded norms not unit! Max diff: {max_diff_folded}"
        print(f"  [OK] fold_W_dec_norm produces unit-norm decoder rows")
        
        # Forward pass after folding should give same result
        # (need to disable rescaling since norms are now folded)
        sae.config.rescale_acts_by_decoder_norm = False
        out_after, _, _, _, _ = sae(x)
        assert torch.allclose(out_before, out_after, atol=1e-4), \
            f"Forward pass changed after fold_W_dec_norm! Max diff: {(out_before - out_after).abs().max().item()}"
        print("  [OK] Forward pass unchanged after fold_W_dec_norm")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test model save/load functionality."""
    print("[test] Testing save/load...")
    
    try:
        import torch
        import tempfile
        from sae import TopKSAE, TopKSAEConfig
        
        config = TopKSAEConfig(d_in=768, expansion_factor=16, k=64)
        sae = TopKSAE(config)
        
        # Random input for reference
        x = torch.randn(8, 768)
        out1, _, _, _, _ = sae(x)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        
        sae.save(save_path)
        print(f"  [OK] Saved to {save_path}")
        
        # Load
        sae2 = TopKSAE.load(save_path)
        out2, _ = sae2(x)
        
        assert torch.allclose(out1, out2), "Outputs differ after load"
        print("  [OK] Load successful, outputs match")
        
        # Cleanup
        import os
        os.unlink(save_path)
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_trainer():
    """Test training loop."""
    print("[test] Testing trainer...")
    
    try:
        import torch
        from sae import TopKSAE, TopKSAEConfig, TopKSAETrainer
        
        config = TopKSAEConfig(d_in=768, expansion_factor=16, k=64)
        sae = TopKSAE(config)
        
        trainer = TopKSAETrainer(
            model=sae,
            lr=1e-3,
            device=torch.device("cpu"),
        )
        
        # Dummy training step
        x = torch.randn(32, 768)
        metrics = trainer.train_step(x)
        
        print(f"  [OK] Training step successful")
        print(f"       Metrics: {metrics}")
        
        # Check decoder still normalized after step
        norms = torch.norm(sae.W_dec, dim=1)
        max_diff = torch.max(torch.abs(norms - torch.ones_like(norms))).item()
        
        assert max_diff < 1e-4, f"Decoder not normalized after training step. Max diff: {max_diff}"
        print(f"  [OK] Decoder normalized after training step (Max diff: {max_diff:.9f})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SAE Training Pipeline Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("SAE Creation", test_sae_creation),
        ("Forward/Backward", test_forward_backward),
        ("Decoder Normalization", test_decoder_normalization),
        ("Save/Load", test_save_load),
        ("Trainer", test_trainer),
    ]
    
    results = []
    for name, test_fn in tests:
        print()
        result = test_fn()
        results.append((name, result))
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print()
    print(f"Passed: {passed}/{len(results)}")
    
    if failed > 0:
        print(f"\n[WARNING] {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tests passed!")
        print("\nNext steps:")
        print("  1. Ensure Orbis experiment directory exists with checkpoint")
        print("  2. Ensure CoVLA/Cityscapes dataset is available")
        print("  3. Run training with:")
        print("     python orbis/sae/scripts/train_sae.py \\")
        print("         --exp_dir /path/to/orbis_experiment \\")
        print("         --data_path /path/to/data \\")
        print("         --k 64 --expansion_factor 16")
        sys.exit(0)


if __name__ == "__main__":
    main()