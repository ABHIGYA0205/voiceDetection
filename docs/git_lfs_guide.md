# Git LFS & Model Checkpoint Policy

## Overview
VoiceShield strictly forbids committing raw audio recordings, large evaluation datasets, or arbitrary large binary artifacts directly into standard Git tree objects. Model weights and checkpoints must be governed through documented releases and Git LFS.

## 1. Prerequisites
Ensure Git LFS is installed and initialized on your macOS machine:
```bash
# Check installation
git-lfs --version

# Initialize Git LFS hooks in the repository
git lfs install
```

## 2. Tracked Extensions
The repository `.gitattributes` file is configured to automatically route the following binary formats to Git LFS:
- `*.pt` (PyTorch model checkpoints)
- `*.pth` (Legacy PyTorch weights)
- `*.onnx` (Exported runtime graphs)
- `*.safetensors` (Safe HuggingFace tensor archives)
- `*.bin` (General model binary weights)

## 3. Workflow for Approved Checkpoints
Before committing any checkpoint:
1. **Model Card Required**: Every committed checkpoint must have a corresponding entry in `model_cards/` detailing architecture, training data hash, license, calibration temperature, and verification metrics.
2. **Verify LFS Staging**:
   ```bash
   git add models/checkpoint_name.pt
   git lfs status
   ```
   Confirm that the file is marked under `Git LFS objects to be committed`.
3. **Small Checkpoint Rule**: Checkpoints tracked in Git LFS must be production candidate weights (<100 MB). Intermediate training iteration dumps must be stored in external object storage (e.g., S3 or GCS), not Git LFS.
