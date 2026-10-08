# Histopathology Cancer Detection with ResNet-18 & Grad-CAM

[![Tests](https://github.com/ardaatikk/histopathology-cancer-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/ardaatikk/histopathology-cancer-detection/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep learning pipeline for detecting **breast cancer metastasis in lymph node histopathology images** using **transfer learning with an ImageNet-pretrained ResNet-18**, with quantitative evaluation and Grad-CAM explainability.

The project covers the complete workflow from image preprocessing and model training to validation, test inference, quantitative evaluation, and visual interpretation of model predictions.

> **Note:** This project was developed for research and educational purposes and is not intended for clinical diagnosis or medical decision-making.

---

## Overview

Cancer diagnosis from histopathology images is a challenging computer vision problem where subtle tissue patterns may distinguish cancerous from non-cancerous samples.

This project implements an end-to-end deep learning workflow for binary classification of histopathology images.

The pipeline includes:

- ResNet-18 based image classification
- Dataset preprocessing and normalization
- Training and validation pipelines
- Test-set probability prediction
- Accuracy, precision, recall, F1 and ROC-AUC evaluation
- Confusion matrix and ROC curve generation
- Grad-CAM based model explainability
- Grad-CAM analysis for arbitrary input images
- Reproducible utility scripts for dataset preparation and visualization

---

## Model Performance

The trained model was evaluated on a held-out validation set containing **350 images**.

| Metric | Score |
|---|---:|
| Accuracy | **88.29%** |
| Precision | **85.37%** |
| Recall / Sensitivity | **97.67%** |
| F1 Score | **91.11%** |
| ROC-AUC | **96.63%** |

The high recall indicates that the model identified most positive samples in this validation split. However, these results should be interpreted in the context of the dataset and should not be treated as evidence of clinical performance.

### Slide-Independent 5-Fold Cross-Validation

The ResNet-18 classifier was evaluated using **5-fold slide-independent cross-validation** on 1,399 labeled histopathology patches originating from 233 slides.

All patches from the same slide were assigned to the same fold to prevent slide-level data leakage between training and validation.

Each fold was evaluated using its corresponding trained checkpoint.

| Fold | Samples | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 281 | 94.66% | 96.93% | 94.05% | 95.47% | 0.9799 |
| 2 | 278 | 94.24% | 93.10% | 97.59% | 95.29% | 0.9762 |
| 3 | 281 | 94.31% | 98.11% | 92.31% | 95.12% | 0.9823 |
| 4 | 279 | 97.85% | 96.53% | 100.00% | 98.24% | 0.9984 |
| 5 | 280 | 91.79% | 96.18% | 89.88% | 92.92% | 0.9725 |
| **Mean ± SD** | **1,399 total** | **94.57 ± 2.16%** | **96.17 ± 1.86%** | **94.77 ± 4.06%** | **95.41 ± 1.89%** | **0.9819 ± 0.0100** |

Additional cross-validation metrics:

| Metric | Mean ± SD |
|---|---:|
| Specificity | 94.29% ± 3.00% |
| Balanced Accuracy | 94.53% ± 1.89% |
| Average Precision (AP) | 0.9818 ± 0.0127 |

### Aggregated Confusion Matrix

The aggregated validation predictions across all five folds produced:

| | Predicted Non-cancer | Predicted Cancer |
|---|---:|---:|
| **Actual Non-cancer** | 529 | 32 |
| **Actual Cancer** | 44 | 794 |

Across the five validation folds, the model correctly classified 1,323 of 1,399 images.

Performance varied across folds. Fold 4 achieved 100% sensitivity, whereas Fold 5 achieved 89.88% sensitivity, indicating variability across different slide groups.

### Initial Baseline Evaluation

An earlier experiment using a single train/validation split achieved:

| Metric | Score |
|---|---:|
| Accuracy | 88.29% |
| Precision | 85.37% |
| Recall | 97.67% |
| F1 Score | 91.11% |
| ROC-AUC | 0.9663 |

These baseline results were obtained under a different evaluation protocol and are provided for historical context. They should not be interpreted as a directly comparable performance improvement.

> **Evaluation limitation:** Slide-independent cross-validation reduces slide-level leakage but does not guarantee patient-level independence. Furthermore, checkpoints were selected using validation performance within their respective folds. The reported cross-validation metrics may therefore be optimistic and do not represent independent external test performance.

---

## Evaluation Results

### Confusion Matrix

<p align="center">
  <img src="assets/confusion_matrix.png" width="600" alt="Confusion Matrix">
</p>

### ROC Curve

<p align="center">
  <img src="assets/roc_curve.png" width="600" alt="ROC Curve">
</p>

The ROC-AUC of **0.9663** indicates strong class-separation performance on the held-out validation data.

---

## Explainable AI (XAI)

To improve the interpretability of model predictions, this project implements multiple **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualization methods.

These methods help investigate which spatial regions influence the model's predictions.

### Standard Grad-CAM

Standard Grad-CAM highlights image regions that positively contribute to a selected class score.

The visualization is generated using gradients from the final convolutional block of ResNet-18.

### Class-Specific Grad-CAM

Class-specific Grad-CAM generates separate activation maps for both classification outputs:

- **Class 0:** Non-cancer
- **Class 1:** Cancer

This allows the spatial evidence associated with each class to be inspected independently.

Because the maps are normalized separately, their color intensities should not be interpreted as directly comparable measures of class confidence.

### Signed Grad-CAM

Signed Grad-CAM extends the standard visualization by retaining both positive and negative activation contributions before applying ReLU.

It distinguishes between:

- **Positive contributions:** Regions that increase the selected class score under the Grad-CAM approximation.
- **Negative contributions:** Regions that decrease the selected class score under the Grad-CAM approximation.

Signed Grad-CAM is generated separately for both classes.

These visualizations can help identify regions associated with supporting or opposing model evidence.

### Prediction Category Analysis

The project also supports Grad-CAM visualization for four prediction categories:

| True Positive | True Negative |
|---|---|
| ![True Positive](assets/gradcam/true_positive.png) | ![True Negative](assets/gradcam/true_negative.png) |

| False Positive | False Negative |
|---|---|
| ![False Positive](assets/gradcam/false_positive.png) | ![False Negative](assets/gradcam/false_negative.png) |

Examining both correct and incorrect predictions provides additional insight into potential model failure patterns.

### Interpretation Limitations

Grad-CAM visualizations represent approximate spatial attribution derived from model activations and gradients.

They do not:

- Provide pixel-level tumor segmentation.
- Establish causal explanations of model predictions.
- Confirm that highlighted regions contain malignant tissue.
- Replace expert pathological assessment.

> These visualizations are intended for model interpretability research, not clinical decision-making.

---

## Single-Image Grad-CAM Analysis

The repository supports explainability analysis for individual histopathology images using a trained ResNet-18 checkpoint.

### Generate Visualizations

```bash
python src/gradcam.py \
  --model_file checkpoints/cv_v1_normalized_fold_1/best_checkpoint.pt \
  --image path/to/image.jpeg
```

The script generates three visualization files:

| Output | Description |
|---|---|
| `gradcam_<image_name>.png` | Standard Grad-CAM visualization |
| `gradcam_classes_<image_name>.png` | Class-specific Grad-CAM maps |
| `gradcam_signed_<image_name>.png` | Signed Grad-CAM maps |

Generated visualizations are saved under:

```text
outputs/gradcam/
```

### Available Explainability Methods

| Method | Implementation | Status |
|---|---|---|
| Standard Grad-CAM | `src/gradcam.py` | Implemented |
| Class-Specific Grad-CAM | `src/gradcam.py` | Implemented |
| Signed Grad-CAM | `src/gradcam.py` | Implemented |
| Integrated Gradients | `src/integrated_gradients.py` | Under development |

Integrated Gradients is being explored as an additional attribution method and is not yet part of the released workflow.

### Important Considerations

Grad-CAM visualizations depend on the trained checkpoint and the selected target class.

Different checkpoints may produce different attribution maps for the same image.

The resulting heatmaps should be interpreted as model-specific explanations rather than ground-truth tumor localization.

---

## Dataset

This project uses histopathology image patches derived from the **CAMELYON17 dataset**, which focuses on detecting breast cancer metastases in lymph node tissue.

The project dataset contains **2,337 histopathology image patches**.

| Subset | Images | Purpose |
|---|---:|---|
| Labeled dataset | 1,399 | Training and cross-validation |
| Unlabeled test dataset | 938 | Inference |
| **Total** | **2,337** | |

### Labeled Dataset

The labeled dataset contains 1,399 images originating from **233 slides**.

The classification task uses two classes:

- **Class 0:** Non-cancer
- **Class 1:** Cancer

The labeled images are organized through `data/labeled.csv`.

### Slide-Independent Cross-Validation

To reduce data leakage, the project uses **5-fold slide-independent cross-validation**.

All image patches originating from the same slide are assigned to the same fold.

This ensures that images from a given slide cannot simultaneously appear in the training and validation subsets of the same fold.

| Fold | Validation Images |
|---|---:|
| 1 | 281 |
| 2 | 278 |
| 3 | 281 |
| 4 | 279 |
| 5 | 280 |
| **Total** | **1,399** |

Cross-validation split metadata is stored under:

`data/folds/`

The split generation process is implemented in:

`scripts/create_cv_folds.py`

> Slide-independent splitting prevents overlap at the slide level but does not necessarily guarantee patient-independent evaluation.

### Unlabeled Test Dataset

An additional 938 images are reserved for inference.

Because ground-truth labels are not available for this subset, classification metrics cannot be calculated for these images.

The corresponding metadata is stored in `data/test.csv`.

### Dataset Availability

Dataset metadata and cross-validation split definitions are included in the repository.

The histopathology image files themselves are not distributed as part of the current repository contents.

Users must obtain the required image data separately and arrange it according to the paths referenced in the dataset metadata.

> The 2,337 images used in this project represent a prepared subset of CAMELYON17 and should not be interpreted as the complete CAMELYON17 dataset.

---

## Image Preprocessing

Histopathology images undergo preprocessing before being passed to the ResNet-18 classifier.

### Image Resizing

Input images are resized to:

```text
224 × 224 pixels
```

The original image patches have a resolution of 512 × 512 pixels.

### RGB Normalization

Images are normalized using dataset-derived RGB mean and standard deviation values.

For cross-validation, **normalization statistics are calculated independently for each fold using only its training images**.

This prevents information from the validation subset from influencing the preprocessing statistics.

The resulting statistics are stored in:

```text
data/folds/fold_1_rgb_stats.json
data/folds/fold_2_rgb_stats.json
data/folds/fold_3_rgb_stats.json
data/folds/fold_4_rgb_stats.json
data/folds/fold_5_rgb_stats.json
```

The statistics can be calculated using:

```bash
python scripts/calculate_rgb_stats.py
```

### Preprocessing Pipeline

The preprocessing pipeline consists of:

1. Loading the histopathology image.
2. Converting the image to RGB.
3. Resizing it to 224 × 224 pixels.
4. Converting it to a PyTorch tensor.
5. Applying RGB normalization using the corresponding training-fold statistics.

The preprocessing implementation is located in:

`src/preprocessing.py`

### Data Leakage Prevention

The cross-validation workflow includes safeguards designed to reduce data leakage:

- Slide-independent training and validation splits.
- Fold-specific RGB normalization calculated from training images.
- Dataset integrity checks for duplicate images and split consistency.
- Automated tests verifying slide separation between training and validation.

These measures improve experimental reliability but do not replace external validation.

---

## Model Architecture

The classifier is based on **ResNet-18** and uses a transfer learning approach.

The network is initialized with **ImageNet-pretrained weights**, the original ImageNet classification layer is removed, and a custom binary classification head is added for breast cancer metastasis detection.

During training, the ResNet-18 backbone is **fine-tuned end-to-end** together with the custom classification head rather than being kept frozen.

Conceptually:

```text
Histopathology Image
        │
        ▼
 Resize + Normalize
        │
        ▼
ImageNet-pretrained ResNet-18
        │
        ▼
 End-to-End Fine-Tuning
        │
        ▼
 512-d Feature Vector
        │
        ▼
 Linear (512 → 128) + ReLU
        │
        ▼
   Linear (128 → 2)
        │
        ▼
Cancer / Non-cancer
```

The model produces two logits corresponding to the binary classes.

For inference, softmax probabilities are calculated and the probability associated with the cancer class is exported as `cancer_score`.

---

## Project Structure

```text
histopathology-cancer-detection/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── assets/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam/
│
├── checkpoints/
│   ├── cv_v1_normalized_fold_1/
│   ├── cv_v1_normalized_fold_2/
│   ├── cv_v1_normalized_fold_3/
│   ├── cv_v1_normalized_fold_4/
│   └── cv_v1_normalized_fold_5/
│
├── data/
│   ├── labeled.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── folds/
│       ├── fold_1_train.csv
│       ├── fold_1_validation.csv
│       ├── fold_1_rgb_stats.json
│       └── ... (corresponding files for folds 2–5)
│
├── images/                  # Dataset images (not tracked)
│
├── scripts/
│   ├── audit_dataset.py
│   ├── calculate_rgb_stats.py
│   ├── create_cv_folds.py
│   ├── evaluate_cross_validation.py
│   ├── plot_training.py
│   ├── run_cross_validation.py
│   └── split_dataset.py
│
├── src/
│   ├── dataset.py
│   ├── dataset_inference.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   ├── test_cross_validation.py
│   ├── test_dataset.py
│   ├── test_gradcam.py
│   ├── test_inference.py
│   └── test_model.py
│
├── outputs/                 # Generated results (not tracked)
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

### Directory Overview

| Directory | Purpose |
|---|---|
| `src/` | Model architecture, preprocessing, training, inference, evaluation and explainability |
| `scripts/` | Dataset auditing, cross-validation orchestration and utility scripts |
| `tests/` | Automated tests for model behavior, data integrity and cross-validation |
| `data/` | Dataset metadata, annotations, fold definitions and RGB statistics |
| `checkpoints/` | Locally generated model checkpoints |
| `assets/` | Selected evaluation figures and Grad-CAM examples |
| `outputs/` | Generated training logs, predictions and evaluation results |

Model checkpoints and dataset images must be provided or generated locally where they are not distributed with the repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ardaatikk/histopathology-cancer-detection.git
cd histopathology-cancer-detection
```

Creating a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

The project supports both conventional model training and slide-independent cross-validation.

### Standard Training

A single training run can be started with:

```bash
python src/train.py
```

Training parameters can be configured through command-line arguments:

```bash
python src/train.py \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_epochs 20 \
  --momentum 0.9
```

This workflow is useful for individual experiments and baseline comparisons.

### Slide-Independent Cross-Validation

The main evaluation protocol uses **5-fold slide-independent cross-validation**.

Cross-validation folds are generated using `StratifiedGroupKFold`, with slide identifiers serving as grouping variables.

This ensures that patches from the same slide are not divided between training and validation within a fold.

#### Step 1: Generate Cross-Validation Folds

```bash
python scripts/create_cv_folds.py \
  --input_file data/labeled.csv \
  --output_dir data/folds \
  --n_splits 5 \
  --seed 42
```

This generates training and validation CSV files for each fold.

#### Step 2: Run Cross-Validation Training

```bash
python scripts/run_cross_validation.py \
  --experiment_prefix cv_v1_normalized \
  --num_epochs 15 \
  --n_splits 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --momentum 0.9 \
  --seed 42
```

The cross-validation workflow trains a separate model for each fold.

Each model uses its corresponding training-fold RGB normalization statistics.

The training configuration supports additional options, including weight decay and learning-rate scheduling.

### Model Checkpoints

Trained model checkpoints are stored under:

```text
checkpoints/
```

For the cross-validation experiment, checkpoints follow the naming convention:

```text
checkpoints/cv_v1_normalized_fold_1/
checkpoints/cv_v1_normalized_fold_2/
checkpoints/cv_v1_normalized_fold_3/
checkpoints/cv_v1_normalized_fold_4/
checkpoints/cv_v1_normalized_fold_5/
```

Each fold is evaluated using its corresponding trained checkpoint.

---

## Evaluation

The repository supports individual checkpoint evaluation and aggregated cross-validation reporting.

### Evaluate a Single Checkpoint

To evaluate a trained model on a validation dataset:

```bash
python src/evaluate.py \
  --model_file checkpoints/cv_v1_normalized_fold_1/best_checkpoint.pt \
  --validation_file data/folds/fold_1_validation.csv \
  --output_dir outputs/evaluation/cv_v1_normalized_fold_1
```

The evaluation script calculates:

- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1 Score
- Balanced Accuracy
- ROC-AUC
- Average Precision (AP)
- Confusion matrix

The script also generates evaluation visualizations.

### Evaluate All Cross-Validation Folds

To evaluate the five trained fold checkpoints and aggregate their results:

```bash
python scripts/evaluate_cross_validation.py \
  --experiment_prefix cv_v1_normalized \
  --n_splits 5
```

The aggregated evaluation results are stored under:

```text
outputs/evaluation/cv_v1_normalized/
```

The cross-validation summary is available at:

```text
outputs/evaluation/cv_v1_normalized/cv_summary.txt
```

### Evaluation Protocol

Each fold is evaluated using validation images originating from slides excluded from that fold's training set.

The final reported metrics represent the mean and standard deviation across five validation folds.

The aggregated confusion matrix combines validation predictions from all folds.

> Checkpoint selection was performed using validation performance. Therefore, the reported results should be interpreted as internal cross-validation performance rather than unbiased independent test performance.

---

## Test Inference

The repository supports probability-based inference on unlabeled histopathology images.

The test dataset contains **938 images** without ground-truth labels.

### Run Inference

Run inference using a trained model checkpoint:

```bash
python src/inference.py \
  --model_file checkpoints/weights_epoch_100.pt \
  --test_file data/test.csv \
  --img_dir images \
  --output_dir outputs
```

The script generates a CSV file containing the image identifier and predicted cancer probability.

For the checkpoint shown above, the output is saved as:

```text
outputs/predictions__weights_epoch_100.csv
```

### Prediction Format

```csv
img_id,cancer_score
slide_...,0.4092
slide_...,0.9955
```

The `cancer_score` represents the softmax probability assigned to the cancer class (Class 1).

### Cross-Validation Checkpoint Compatibility

The current inference implementation uses the default preprocessing configuration.

Cross-validation checkpoints were trained using fold-specific RGB normalization statistics.

Therefore, inference with these checkpoints requires ensuring that the preprocessing configuration matches the corresponding training fold.

Updating the inference pipeline to automatically restore normalization statistics from the checkpoint is a planned improvement.

> Because the test dataset does not contain ground-truth labels, classification metrics cannot be calculated for these predictions.

---

## Grad-CAM Validation Examples

The repository supports Grad-CAM analysis of both individual images and validation predictions.

### Analyze Validation Predictions

Generate Grad-CAM examples for True Positive, True Negative, False Positive and False Negative predictions:

```bash
python src/gradcam.py \
  --model_file checkpoints/cv_v1_normalized_fold_1/best_checkpoint.pt \
  --validation_file data/folds/fold_1_validation.csv \
  --output_dir outputs/gradcam/fold_1
```

The script selects examples from the specified validation set and generates visualizations for the available prediction categories.

Some categories may be absent from a particular fold. For example, Fold 4 produced no false negatives.

### Analyze an Individual Image

```bash
python src/gradcam.py \
  --model_file checkpoints/cv_v1_normalized_fold_1/best_checkpoint.pt \
  --image path/to/image.jpeg
```

The generated visualizations are saved under:

```text
outputs/gradcam/
```

These visualizations are intended for model interpretation and do not constitute tumor segmentation or clinical evidence.

---

## Utility Scripts

The repository provides utility scripts for dataset preparation, validation, cross-validation and experiment analysis.

### Dataset Auditing

Inspect dataset integrity and identify potential data quality issues:

```bash
python scripts/audit_dataset.py
```

### Generate Cross-Validation Folds

Create reproducible slide-independent cross-validation splits:

```bash
python scripts/create_cv_folds.py \
  --input_file data/labeled.csv \
  --output_dir data/folds \
  --n_splits 5 \
  --seed 42
```

The script uses `StratifiedGroupKFold` to preserve class distributions while keeping images from the same slide within the same fold.

### Calculate RGB Statistics

Calculate normalization statistics for image preprocessing:

```bash
python scripts/calculate_rgb_stats.py \
  --image_folder images \
  --train_csv data/folds/fold_1_train.csv \
  --output_json data/folds/fold_1_rgb_stats.json
```

For cross-validation experiments, RGB statistics must be calculated using training images only.

### Run Cross-Validation

Train the models for all five folds:

```bash
python scripts/run_cross_validation.py \
  --experiment_prefix cv_v1_normalized \
  --num_epochs 15 \
  --n_splits 5
```

### Evaluate Cross-Validation

Evaluate trained checkpoints and aggregate performance metrics:

```bash
python scripts/evaluate_cross_validation.py \
  --experiment_prefix cv_v1_normalized \
  --n_splits 5
```

### Plot Training History

Visualize model training history:

```bash
python scripts/plot_training.py \
  --log_file outputs/training/training_TIMESTAMP.csv
```

### Create a Train/Validation Split

The repository also supports conventional stratified splitting for baseline experiments:

```bash
python scripts/split_dataset.py \
  --data_file path/to/labeled_dataset.csv
```

This workflow is separate from the slide-independent cross-validation protocol.

---

## Limitations

Although the model demonstrates promising experimental performance, several limitations should be considered.

### Dataset Size and Generalization

The labeled dataset contains 1,399 histopathology image patches.

This is relatively small for medical image classification, and the model may not generalize reliably to substantially different datasets.

Performance may vary across medical centers, scanners, staining protocols and tissue preparation procedures.

### Slide-Level Independence

The cross-validation procedure ensures that images from the same slide do not appear in both training and validation within a fold.

However, slide-independent splitting does not necessarily guarantee patient-independent evaluation.

### Validation-Based Model Selection

The reported cross-validation results were obtained using checkpoints selected according to validation performance.

Because model selection and performance reporting involve the same validation folds, the results may contain optimistic bias.

An independent test dataset with ground-truth annotations would provide stronger evidence of generalization.

### Explainability Limitations

Grad-CAM and its variants provide approximate visual explanations of model behavior.

The generated attribution maps do not establish causal relationships or provide clinically validated tumor localization.

They should not be interpreted as segmentation masks.

### Image Resolution

The original histopathology patches are 512 × 512 pixels but are resized to 224 × 224 pixels for classification.

This resizing may remove or reduce fine-grained tissue information relevant to prediction.

A systematic comparison of input resolutions has not yet been completed.

### Clinical Validation

The model has not undergone prospective clinical evaluation or independent external validation.

The reported results represent experimental machine-learning performance and should not be interpreted as evidence of clinical diagnostic reliability.

---

## Testing

The repository includes automated tests covering model functionality, data processing, inference, explainability and cross-validation integrity.

The latest locally verified test run completed with **37 passing tests**.

The test suite covers:

- Dataset loading and preprocessing.
- Model construction and forward passes.
- Checkpoint loading.
- Inference output generation.
- Grad-CAM visualization and normalization.
- Class-specific and signed Grad-CAM behavior.
- Duplicate image detection.
- Slide-independent train/validation separation.
- Cross-validation fold coverage.
- Validation coverage across all folds.
- Fold-specific RGB statistics consistency.

### Run Tests

```bash
python -m pytest -v
```

### Continuous Integration

Automated tests are executed through GitHub Actions.

The workflow is defined in:

```text
.github/workflows/tests.yml
```

The CI pipeline helps detect regressions and validate the integrity of the repository.

---

## Reproducibility

The repository is structured to support reproducible machine-learning experiments.

### Included Resources

The repository provides:

- Model architecture and preprocessing implementations.
- Training and evaluation pipelines.
- Dataset annotation metadata.
- Slide-independent cross-validation split definitions.
- Fold-specific RGB normalization statistics.
- Cross-validation orchestration scripts.
- Explainability implementations.
- Automated tests and GitHub Actions CI.

### Experiment Configuration

The reported cross-validation experiment uses:

| Parameter | Value |
|---|---|
| Architecture | ResNet-18 |
| Initialization | ImageNet-pretrained |
| Task | Binary classification |
| Input resolution | 224 × 224 |
| Cross-validation | 5 folds |
| Split strategy | StratifiedGroupKFold |
| Grouping variable | Slide ID |
| Random seed | 42 |
| Maximum epochs | 15 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Normalization | Training-fold RGB statistics |

### Reproduction Requirements

The original histopathology image files and trained model checkpoints may not be included in the repository.

To reproduce the experiments, users must obtain the required image data, install the project dependencies and execute the training and evaluation pipelines.

Reproducing the exact reported results may also depend on library versions, hardware and other sources of computational nondeterminism.

### Experimental Scope

The reported results are based on internal slide-independent cross-validation.

They should not be interpreted as independently validated clinical performance.

---

## Dataset Source

This project uses image samples derived from the **CAMELYON17 (CAMelyon17) dataset**, a public histopathology dataset developed for the automated detection and classification of breast cancer metastases in hematoxylin and eosin (H&E) stained lymph node sections.

The original CAMELYON17 dataset contains whole-slide images collected from five medical centers in the Netherlands. The 2,337 images used in this repository represent the subset prepared for this project and should not be interpreted as the complete CAMELYON17 dataset.

### References

- [CAMELYON17 Grand Challenge](https://camelyon17.grand-challenge.org/) — official dataset and challenge documentation
- Bandi, P. et al. *From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: The CAMELYON17 Challenge.* IEEE Transactions on Medical Imaging. DOI: 10.1109/TMI.2018.2867350
- Litjens, G. et al. *1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset.* GigaScience. DOI: 10.1093/gigascience/giy065

---

## Technologies

- Python
- PyTorch
- Torchvision
- ResNet-18
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- Grad-CAM

---

## Disclaimer

This repository is a **research and educational machine-learning project**.

It is **not a medical device**, has not been clinically validated, and must not be used to diagnose cancer or make medical decisions.

---

## Author

**Arda Atik**

Artificial Intelligence Engineer

- GitHub: [@ardaatikk](https://github.com/ardaatikk)