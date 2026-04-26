# AGENTS.md

This file provides context and guidelines for working in this repository, particularly for AI agents or contributors making changes. The repository implements **Heterogeneous Test-Time Training (HTT)** for multi-modal person and vehicle Re-identification (ReID), as described in the AAAI 2024 paper. It supports multimodal inputs (RGB, IR, TI, and text captions), Vision Transformers (ViT-B/16 as student), frozen Nomic embeddings (vision/text v1.5), editing adapters, projection heads, and custom losses (CIM + 3M, MSE Distill, InfoNCE V-T, Triplet, ID CE with memory bank).

The codebase is structured to minimize changes while extending functionality—e.g., for the architecture diagram involving frozen encoders, modality-type embeddings, CLS concatenation, and multi-loss training. Always prioritize backward compatibility with existing configs (e.g., `vit_base_4M-RGB+IR+TI+All-Caption.yml`).

Key principles:
- **Modularity**: Changes should be config-driven (via `config/defaults.py` and YAML files in `configs/`).
- **Multimodality Focus**: Handle RGB/IR/TI as vision modalities (preprocessed for Nomic-vision), captions as text (via Nomic-text). Use frozen models from Hugging Face (`nomic-ai/nomic-embed-vision-v1.5`, `nomic-ai/nomic-embed-text-v1.5`).
- **ViT Integration**: Student ViT is in `model/backbones/vit_pytorch.py`; adapters and projections in `model/make_model.py`.
- **Losses**: Extend `loss/make_loss_multi.py` for new losses; ensure weighting via config (e.g., `LOSS.MSE_DISTILL_WEIGHT`).
- **Validation**: Always test with `train.py`/`test.py` and check mAP/CMC metrics.
- **Style**: Follow PEP8; use `black` for formatting. Document new features in README.md or inline comments.

Agents should explore context in:
- `README.md`: High-level overview, training commands.
- `fig/network.png`: Visual architecture (motivation for CIM, MTT).
- Nested AGENTS.md if in subdirs (none currently; use this root file).

For migrations: The codebase is evolving to full diagram support (Nomic, adapter, V-T alignment). Avoid breaking single-modal baselines.

## Contributor Guide

### Dev Environment Tips
- **Setup**: Use Python 3.8+ (tested up to 3.12). Create a virtual env: `python -m venv env; source env/bin/activate`.
- Install dependencies: `pip install -r requirements.txt`. Add `transformers` and `sentence-transformers` for Nomic/Hugging Face integration if not present.
- **No Internet in Tools**: Rely on pre-installed libs (e.g., torch, timm, yacs). Do not attempt `pip install` in code_execution tool.
- **Running Projects**: 
  - For a specific dataset/modality: `python train.py --config_file configs/RGBNT201/vit_base.yml` (single-modal) or `configs/RGBNT201/vit_base_4M-RGB+IR+TI+All-Caption.yml` (multimodal with captions).
  - Multimodal training: `python train_multi.py --config_file configs/Market1501/vit_base_4M_C.yml`.
  - Test-time training: `python test_time_train.py --config_file configs/RGBNT201/vit_base_ttt.yml`.
  - Inference: `python test.py --config_file <config> --TEST.WEIGHT <path_to_model.pth>`.
- **Adding Packages**: To add a new dataset or modality, update `datasets/make_dataloader.py` and create a YAML in `configs/<dataset>/`. For example, spin up Nomic integration by loading models in `model/make_model.py` with `from transformers import AutoModel`.
- **Navigation**: Use `ls configs/` to list configs; `grep -r "ViT" model/` for ViT-related code. Confirm package names in `requirements.txt`.
- **Debugging**: Set `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` for GPU; log with `utils/logger.py`. For multimodality, verify inputs in `datasets/preprocessing.py` (e.g., resize to 224x224 for Nomic-vision).

### Testing Instructions
- **CI Overview**: Check `.github/workflows/` for any automated pipelines (currently minimal; focus on local validation).
- **Run Tests**: 
  - Full suite for a config: `python test.py --config_file <config>`.
  - Filter by project: No turbo/pnpm; instead, run `python train.py --config_file configs/<dataset>/vit_base.yml --SOLVER.MAX_EPOCHS 1` for quick training smoke test.
  - From root: `python test.py` with default config.
- **Focused Testing**: For losses, add print statements in `loss/make_loss_multi.py` (e.g., log MSE Distill value). For ViT, check outputs in `model/backbones/vit_pytorch.py`.
- **Validation Steps**:
  - Fix errors: Run until mAP/CMC are reported in logs (e.g., >50% mAP for baselines).
  - After changes (e.g., adding adapter): Run `python train.py` with small epochs; verify no NaN losses.
  - Lint: Manually check with `black --check .` or `flake8`; add tests in `utils/metrics.py` for new eval funcs.
  - Multimodal Specific: Ensure Nomic frozen (`.requires_grad = False`); test V-T alignment with InfoNCE < 1.0.
  - Update/Add Tests: For new features (e.g., memory bank in losses), add unit tests in a new `tests/` dir if needed, or inline assertions.
- **Pre-Merge**: All logs should show green metrics; no crashes on CUDA.

### PR Instructions
- **Title Format**: `[<component>] <Descriptive Title>`, e.g., `[model] Implement frozen Nomic embeddings and editing adapter`, `[loss] Add MSE Distill and InfoNCE V-T losses`.
- **Description**: Include why (e.g., "Implements diagram architecture for multimodality"), what changed (files modified), how tested (configs run, metrics before/after), and any config updates.
- **Commits**: Atomic; e.g., one for model, one for loss. Use `git add <file>`; message like "Add mod-type embeds to ViT".
- **Reviews**: Reference issues/paper; ensure compatibility with baselines.
- **Branching**: From main; name like `feature/nomic-integration`.
- **Presentation**: If adding docs, update README.md with new commands/examples. For agents, note how changes align with diagram (e.g., "Extends forward pass for CLS concat").
