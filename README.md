# Diabetic Retinopathy Stage Detection

A CNN with transfer learning that classifies retinal fundus images into five diabetic retinopathy stages:

| Label | Stage |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Project structure

```
configs/           Experiment configuration (hyperparameters, paths)
data/
  raw/             Original Kaggle dataset (not committed)
  processed/       Preprocessed images (not committed)
  splits/          Train / validation / test split CSVs
notebooks/         Exploratory data analysis and visualisation
src/
  data/            Dataset loading, preprocessing, augmentation
  models/          CNN architectures (transfer learning backbones)
  training/        Training loop, callbacks, schedulers
  evaluation/      Metrics, confusion matrix, curves, error analysis
  utils/           Shared helpers (seeding, logging, plotting)
api/               Inference API (FastAPI), deployed to Hugging Face Spaces
web/               Web page for uploading images and viewing results, deployed to Vercel
models/            Saved model checkpoints (not committed)
outputs/
  figures/         Plots used in the report
  logs/            Training logs
  metrics/         Evaluation results
tests/             Unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

[Diabetic Retinopathy 2015 Data Colored Resized](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized)
(EyePACS origin, 35,126 images at 224×224, CC0 licence).

**Option A: Kaggle Notebook (recommended, no download).** Attach the dataset to the notebook with
*Add Input*. The notebooks detect it under `/kaggle/input` automatically.

**Option B: local download** (about 2 GB, needs a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d sovitrath/diabetic-retinopathy-2015-data-colored-resized -p data/raw --unzip
```

If the data is somewhere else, point to it with `export DR_DATA_ROOT=/path/to/dataset`.

## Preprocessing

`src/data/preprocessing.py` applies, in order: black border crop, resize to 224×224,
3×3 median denoising, CLAHE on the LAB lightness channel, Ben Graham blur subtraction
(σ = 10) and a retina mask. ImageNet normalisation is applied when converting to a tensor.
Each step can be switched on or off under `preprocessing:` in `configs/config.yaml`.

## Usage

All settings (paths, image size, preprocessing, hyperparameters) live in `configs/config.yaml`.

```bash
# Patient-level stratified 70/15/15 split -> data/splits/{train,val,test}.csv
python -m src.data.split

# Exploratory data analysis (figures saved to outputs/figures/)
jupyter notebook notebooks/01_eda.ipynb

# Preprocessing demonstration and quality measures
jupyter notebook notebooks/02_preprocessing.ipynb

# Augmentation and class balancing figures
jupyter notebook notebooks/03_augmentation_balancing.ipynb

# Train (two-stage transfer learning, see src/training/train.py for all options)
python -m src.training.train --run-name effb0_cw
python -m src.training.train --run-name effb0_sampler --balancing sampler
python -m src.training.train --run-name resnet50_cw --backbone resnet50
python -m src.training.train --run-name effb0_raw --no-enhancement   # preprocessing ablation

# Unit tests
pytest
```

Each training run writes `models/<run>/best.pt`, `outputs/logs/<run>/history.csv`,
`outputs/figures/<run>_curves.png` and a summary row in `outputs/metrics/experiments.csv`.

## Model and training

- Backbones (timm, ImageNet-pretrained): EfficientNet-B0 (main), ResNet50, MobileNetV3-Large
- Two-stage transfer learning: frozen backbone (head only, lr 1e-3, 3 epochs), then full
  fine-tuning (lr 1e-4, cosine schedule, early stopping on validation QWK, patience 5)
- AdamW, weight decay 1e-4, dropout 0.3, label smoothing 0.05, mixed precision on GPU
- Class imbalance: class-weighted loss or weighted sampler (square-root inverse frequency)
- Augmentation (training only): rotation, flips, zoom/shift, brightness/contrast, small hue shift

### Running on Kaggle

In a new Kaggle Notebook (GPU on, dataset attached, internet on):

```python
!git clone https://github.com/RoshanChampika1/dr-stage-detection.git
%cd dr-stage-detection
!pip install -q timm albumentations
!python -m src.training.train --run-name effb0_cw
```

Turn the GPU on (*Settings → Accelerator → GPU T4 x2*) and internet on (needed to download
the pretrained weights). On Kaggle, preprocessed images are cached in `/kaggle/temp/processed`.
