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

# Evaluate on the held-out test split (metrics, confusion matrix, ROC, errors, Grad-CAM)
python -m src.evaluation.evaluate --runs effb0_cw resnet50_cw

# Rebuild the results tables from all per-run result files
python -m src.evaluation.summarize

# Unit tests
pytest
```

Each training run writes `models/<run>/best.pt`, `outputs/logs/<run>/history.csv`,
`outputs/figures/<run>_curves.png` and `outputs/metrics/<run>_summary.json`. Evaluation writes
`<run>_test.json`, per-class metrics, predictions and figures. `experiments.csv` (validation) and
`test_results.csv` (test) are rebuilt from these per-run files, so runs from different sessions merge.

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

## Web demo
The demo runs the trained model **in the browser** with ONNX Runtime Web: the photograph is
analysed on the user's device and never uploaded. The page is static, so it is hosted for free
on Vercel (or any static host).

- `web/`: the page (`index.html`, `style.css`, `app.js`) and `preprocess.js`, a JavaScript port
  of the training preprocessing (adaptive crop, OpenCV-identical resize, ImageNet normalisation,
  image-quality checks). Tested against the Python code: within ±1 grey level per pixel and the
  same predictions.
- `web/model/`: `dr_model.onnx` (network, 16 MB) and `model_meta.json` (class names,
  preprocessing, screening threshold, classifier weights for the heatmap), created by:

```bash
python -m src.deployment.export_onnx --checkpoint models/effb0_raw_lr3e-4/best.pt
```

  The export checks that ONNX and PyTorch agree and that the heatmap computed from the feature
  maps and classifier weights equals Grad-CAM (exact for global-average-pool + linear heads).

Run locally: `python -m http.server 8000 --directory web`, then open http://localhost:8000.

Deploy on Vercel: import the GitHub repository, set **Root Directory** to `web`, framework
preset **Other**, no build command.

`api/` contains the same inference as a **FastAPI server** (for a clinic server or a paid
container host): `POST /predict`, `GET /health`, the page at `/app/`, and `api/space/Dockerfile`
for Docker. Run it with `uvicorn api.main:app --port 8000` after copying the checkpoint to
`api/model/best.pt`.

**Research prototype only. Not a medical device and not for diagnosis.**

