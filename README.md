# Solar Filament Segmentation

Kaggle research notebooks for the MAGFiLO 1.0 solar filament segmentation challenge: baseline training and inference, comparisons of segmentation backbones, and adaptation of Surya’s HelioSpectFormer to single-channel H-alpha imagery.

This repository contains work in progress exported from Kaggle. The notebooks retain their original code, metadata, and saved outputs; filenames have been simplified for navigation.

## Notebooks

| Notebook | Contents |
| --- | --- |
| [Vanilla training pipeline](notebooks/vanilla-training-pipeline.ipynb) | U-Net++ with an ImageNet-pretrained ResNet34 encoder, training and validation utilities, checkpointing, and training history. |
| [Backbone experiments](notebooks/backbone-experiments.ipynb) | U-Net++, DeepLabV3+, and SegFormer model definitions and experiment utilities. Some experiment calls are commented out. |
| [Baseline inference pipeline](notebooks/baseline-inference-pipeline.ipynb) | Loads the U-Net++ checkpoint, predicts probability masks, extracts instances, encodes masks as COCO RLE, and writes a submission CSV. |
| [Surya training pipeline](notebooks/surya-training-pipeline.ipynb) | Adapts HelioSpectFormer for H-alpha input, resizes spectral filters, replaces the decoder, and configures two-GPU model parallelism. |

## Environment and inputs

These notebooks are configured for Kaggle, with competition data mounted at:

```text
/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
```

Access the competition data through Kaggle under its applicable terms. Data, model weights, and generated checkpoints are not included in this repository.

The notebooks use Python, PyTorch, NumPy, pandas, Pillow, SciPy, tqdm, and `segmentation-models-pytorch`. Additional notebook-specific imports include torchvision, matplotlib, PyYAML, einops, and pycocotools. This is not a pinned or independently reproduced environment.

- **Training:** attach the competition dataset and review the notebook’s paths and GPU settings before running.
- **Baseline inference:** also attach a training output containing `unetplusplus_resnet34_focal_dice.pt`; the notebook searches Kaggle inputs for that file.
- **Surya:** separately provide the Surya implementation at `/kaggle/working/surya_minimal` and `config.yaml` plus `surya.366m.v1.pt` under `/kaggle/working/surya_weights`. Its current model-parallel configuration requires at least two CUDA GPUs. Those external files are not bundled here.

## Current snapshot notes

These exports have not been rerun end to end outside Kaggle. Saved outputs reflect earlier interactive sessions and may not match a fresh sequential execution.

- The vanilla training export includes a saved `NameError` for `run_experiment`, even though that function appears in an earlier cell. A clean sequential run is needed to verify the current notebook.
- The backbone experiments export includes a saved CUDA out-of-memory error during the SegFormer experiment.
- The Surya notebook depends on the separately prepared code and weights described above.

The initial upload preserves these research snapshots rather than silently changing the experiments or claiming they are fully reproducible.
