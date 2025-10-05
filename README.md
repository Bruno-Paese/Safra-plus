## Safra+ — Bloom Detection with Satellite Data and Deep Learning

Safra+ is a proof of concept that integrates Earth observation data with machine learning to detect and monitor plant blooming events at scale. The project combines visual evidence from satellite imagery with vegetation indices (e.g., NDVI) and a compact neural network to classify bloom vs. non-bloom scenarios, supporting environmental, agricultural, and ecological decision-making.

### Why blooming from space?
- **Spectral signatures**: Blooming events produce distinct spectral patterns observable from space, especially in visible and NIR channels.
- **NDVI support**: NDVI correlates strongly with vegetation vigor and crop phenology; it helps contextualize blooms within crop growth cycles.

### Data sources and context
- **SENTINEL (Copernicus/ESA)**: Primary imagery source; visible and Near-Infrared (NIR) channels enable both visual detection and vegetation index computation.
- **NDVI literature**: Prior work validates NDVI time-series for crop phenology monitoring, including UFSM’s study in irrigated corn areas.
- **Case study — Corn (maize)**: Chosen for its global relevance and well-documented NDVI curve. FAO reported ~201.66 million hectares of corn in 2024.

---

## Project structure

```
Safra-plus/
  MLPTest.py
  dataset/
    train/
      florado/
      não florado/
    test/
      florado/
      não florado/
```

- The dataset is organized for Keras `image_dataset_from_directory`, with subfolders per class.
- Class names are in Portuguese: `florado` (blooming) and `não florado` (non-blooming). Non‑ASCII folder names are supported on modern filesystems.

---

## Model overview

This repository implements a compact Multi-Layer Perceptron (MLP) in TensorFlow/Keras for binary classification of 100×100 RGB images (flattened to 30,000 features):

- **Input**: 100×100×3 RGB, normalized to [0, 1], then flattened
- **Dense(1024, ReLU)** → **Dropout(0.3)**
- **Dense(256, ReLU)** → **Dropout(0.3)**
- **Dense(64, ReLU)** → **Dropout(0.2)**
- **Dense(1, Sigmoid)** (binary output)
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Binary cross-entropy
- **Metric**: Accuracy

Training and evaluation artifacts include confusion matrix, precision/recall/F1, and a heatmap visualization.

Notes:
- This simple baseline demonstrates that compact models can learn bloom-related patterns even from RGB.
- Future versions can extend inputs with NIR or NDVI-derived channels, or adopt CNN/ViT architectures.

---

## Setup

### Requirements
- Python 3.10+
- Recommended: virtual environment
- GPU (optional) with compatible TensorFlow build

### Install

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

If you have a CUDA-capable GPU, install a GPU-enabled TensorFlow build per the official TensorFlow install guide.

### Dataset
Place your images under `dataset/train` and `dataset/test`, each with two subfolders:

```text
dataset/
  train/
    florado/
    não florado/
  test/
    florado/
    não florado/
```

Images will be resized to 100×100 during loading. RGB PNGs are expected.

---

## How to run

```bash
python MLPTest.py
```

What it does:
- Builds the MLP classifier and prints its summary
- Loads `dataset/train` and `dataset/test` via `image_dataset_from_directory`
- Normalizes images to [0, 1] and flattens to 30,000-d vectors
- Trains for 10 epochs with validation on the test split
- Prints test accuracy, confusion matrix, precision/recall/F1, and shows a heatmap

Outputs:
- Console metrics
- A confusion matrix heatmap window (matplotlib)

Tip: If running headless, consider using a non-interactive backend or saving figures to disk.

---

## Roadmap: from prototype to platform

Safra+ envisions a multi-modal, cloud-native platform with two main microservice blocks:

- **Data ingestion**: Stream satellite observations (SENTINEL), compute/ingest NDVI, and store in a sharded NoSQL database (e.g., MongoDB) optimized for write-heavy workloads.
- **Inference & alerting**: Execute trained models at scale, generate alerts, and persist results in an application database driving a web UI for global access.

Enhancements under consideration:
- Integrate pre-calculated NDVI from Copernicus datasets; generate NDVI on-the-fly from raw bands
- Add agricultural records (e.g., USDA CroplandCROS) for crop-type validation
- Expand training with manually labeled historical data; refine via user feedback loops
- Extend model inputs with NIR/NDVI channels; explore CNNs and transformers for spatial context

---

## Business model

- **Free tier**: Regional alerts derived from preprocessed satellite data and low‑resolution models.
- **Premium tier**: Farm‑level, high‑resolution analyses and real‑time, crop‑specific alerts with full‑resolution imagery.

Example: A commercial orange producer can time pollinator deployment (e.g., bee hives) using bloom alerts for improved yields.

Competitive edge: Near real-time satellite streaming and high-availability infrastructure for rapid, reliable decision support.

---

## References

- Bortolotto, R. P., Fontana, D. C., & Kuplich, T. M. (2022). Using NDVI time-series profiles for monitoring corn plant phenology of irrigated areas in southern Brazil. Agrociencia Uruguay, 26(1), 291. https://doi.org/10.31285/AGRO.26.291
- Wang, J., Chen, C., Wang, J., et al. (2025). NDVI Estimation Throughout the Whole Growth Period of Multi-Crops Using RGB Images and Deep Learning. Agronomy, 15(1), 63. https://doi.org/10.3390/agronomy15010063
- Raun, W. R., Solie, J. B., Martin, K. L., et al. (2005). Growth stage, development, and spatial variability in corn evaluated using optical sensor readings. Journal of Plant Nutrition, 28(2), 177.

---

## License

Add your preferred license (e.g., MIT, Apache-2.0) here.


