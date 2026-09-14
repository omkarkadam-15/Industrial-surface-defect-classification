# Industrial Surface Defect Classification — Transfer Learning

An end-to-end deep learning computer vision project for **automated steel surface defect classification** using **transfer learning with MobileNetV2**.

> **Model:** MobileNetV2 + Custom Classification Head  
> **Validation Accuracy:** 98.89%  
> **Peak Validation Accuracy:** 99.17%  
> **Validation Loss:** 0.0336  
> **Trainable Parameters:** 7,686  
> **Defect Classes:** 6

## Project Overview

This project addresses automated inspection of steel manufacturing surfaces, where defects such as cracks, scratches, inclusions, and pits can affect product quality. The model classifies grayscale industrial images into six defect categories.

The project uses **MobileNetV2 pretrained on ImageNet** as a frozen feature extractor and trains only a lightweight classification head. This transfer-learning strategy reuses pretrained visual representations while substantially reducing the number of parameters that need to be trained.

### Defect Categories

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Business Value

Automated surface inspection can help manufacturing organisations:

- Reduce dependence on manual visual inspection
- Improve consistency of defect detection
- Reduce inspection and rework costs
- Scale quality-control processes
- Support lightweight or edge-based inspection applications
- Adapt pretrained computer-vision models to industrial datasets with limited training resources

## Dataset

The project uses the **NEU Surface Defect Dataset**, containing grayscale steel surface images from an industrial manufacturing context.

| Split | Images | Images / Class | Classes |
|---|---:|---:|---:|
| Training | 1,440 | 240 | 6 |
| Validation | 360 | 60 | 6 |
| **Total** | **1,800** | — | **6** |

The dataset is balanced across the six classes.

## Computer Vision Preprocessing

The original images are grayscale, while ImageNet-pretrained MobileNetV2 expects three-channel RGB input.

The preprocessing pipeline:

1. Converts grayscale images to RGB when required.
2. Resizes images to **224 × 224** pixels.
3. Scales pixel values to `[0, 1]`.
4. Applies ImageNet mean and standard-deviation normalization.
5. Uses batch size **16**.
6. Uses caching and prefetching to improve the input pipeline.

## Transfer Learning Approach

### Why MobileNetV2?

MobileNetV2 was selected as a lightweight CNN pretrained on ImageNet. It provides a balance between visual feature extraction capability and computational efficiency, making it suitable for texture-based industrial defect classification.

### Architecture

```text
Input Image
224 × 224 × 3
       │
       ▼
Pretrained MobileNetV2
(ImageNet weights)
       │
       │ Frozen
       ▼
Feature Maps
       │
       ▼
Global Average Pooling
       │
       ▼
Dropout (20%)
       │
       ▼
Dense Layer
6 Outputs
       │
       ▼
Softmax
       │
       ▼
Defect Class
```

The MobileNetV2 convolutional backbone was frozen and only the custom classifier head was trained.

## Model Architecture

The custom classification head consists of:

- **GlobalAveragePooling2D:** Converts MobileNetV2 feature maps into a compact feature vector.
- **Dropout (20%):** Helps improve generalization.
- **Dense + Softmax:** Produces probabilities for the six defect classes.

### Parameter Count

| Parameter Type | Count |
|---|---:|
| Total parameters | 2,265,670 |
| Trainable parameters | **7,686** |
| Non-trainable parameters | 2,257,984 |

Only **7,686 parameters** were trained.

## Training Configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Loss: Categorical Crossentropy
Batch Size: 16
Epochs: 20
Input Size: 224 × 224 × 3
Random Seed: 42
```

## Results

### Overall Performance

| Metric | Result |
|---|---:|
| Training Accuracy | 99.93% |
| Validation Accuracy | **98.89%** |
| Peak Validation Accuracy | **99.17%** |
| Validation Loss | **0.0336** |
| Macro F1-score | **0.99** |
| Weighted F1-score | **0.99** |

### Class-wise Performance

| Defect Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Crazing | 1.00 | 1.00 | 1.00 |
| Inclusion | 1.00 | 0.93 | 0.97 |
| Patches | 1.00 | 1.00 | 1.00 |
| Pitted Surface | 1.00 | 1.00 | 1.00 |
| Rolled-in Scale | 1.00 | 1.00 | 1.00 |
| Scratches | 0.94 | 1.00 | 0.97 |

The main errors occurred between visually similar **Inclusion** and **Scratches** patterns.

## Training Behaviour

The model converged rapidly during the 20 training epochs.

At the end of training:

- Training accuracy: **100%**
- Validation accuracy: approximately **99%**
- Training loss: **0.0032**
- Validation loss: **0.0336**

The project diagnostics reported a small training-validation accuracy gap and no evidence of significant overfitting.

## Prediction Analysis

Validation predictions were visualized by comparing the true label with the predicted defect class. Representative images were displayed for all six categories to qualitatively inspect model behaviour.

## Key Technical Insights

### 1. Transfer learning reduced training complexity

Freezing MobileNetV2 allowed the project to reuse pretrained visual features instead of training a deep CNN from scratch.

### 2. Very few parameters required training

Only **7,686 parameters** were trainable compared with more than **2.26 million total parameters**.

### 3. MobileNetV2 transferred effectively to industrial textures

The pretrained feature extractor achieved strong validation performance on the steel surface defect classification task.

### 4. Balanced dataset

Each class contained 240 training images and 60 validation images, reducing concerns about class imbalance.

### 5. Visually similar defects were the main source of errors

Inclusion had a recall of 0.93, while Scratches had a precision of 0.94, indicating limited confusion between these visually similar patterns.

## Project Workflow

```text
Industrial Problem Definition
          │
          ▼
NEU Surface Defect Dataset
          │
          ▼
Dataset Understanding
          │
          ▼
Class Distribution Analysis
          │
          ▼
Image Preprocessing
  ├── Grayscale → RGB
  ├── Resize to 224×224
  └── ImageNet Normalization
          │
          ▼
MobileNetV2 Selection
          │
          ▼
Freeze Pretrained Backbone
          │
          ▼
Custom Classification Head
  ├── Global Average Pooling
  ├── Dropout
  └── Dense + Softmax
          │
          ▼
Train Classifier Head
          │
          ▼
Validation Evaluation
          │
          ▼
Prediction Analysis
          │
          ▼
Industrial Defect Classification
```

## Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MobileNetV2
- Transfer Learning
- Computer Vision
- Image Classification
- Google Colab / GPU

## Repository Structure

```text
industrial-surface-defect-classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Transfer_Learning_Surface_Defect.ipynb
│
├── data/
│   └── README.md
│
└── results/
    └── README.md
```

> Update filenames and folders to match the actual files committed to the repository.

## Dataset structure

```text
NEU-DET/
├── train/
│   └── images/
│       ├── crazing/
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
│
└── validation/
    └── images/
        ├── crazing/
        ├── inclusion/
        ├── patches/
        ├── pitted_surface/
        ├── rolled-in_scale/
        └── scratches/
```

## Future Improvements

Potential extensions include:

- Fine-tuning selected upper MobileNetV2 layers
- Data augmentation
- Testing on additional external industrial data
- Confusion-matrix analysis
- Grad-CAM or other visual explainability methods
- Model quantization for edge deployment
- Benchmarking against other lightweight CNN architectures

> These are proposed extensions and were not part of the current implementation.

## Project Highlights

```text
Dataset           : NEU Surface Defect Dataset
Training Images   : 1,440
Validation Images : 360
Classes           : 6
Input Size        : 224 × 224 × 3

Backbone          : MobileNetV2
Pretrained On     : ImageNet
Transfer Method   : Frozen feature extractor
Trainable Params  : 7,686

Validation Acc.   : 98.89%
Peak Val. Acc.    : 99.17%
Macro F1           : 0.99
Weighted F1        : 0.99
```

## Disclaimer

This project is an academic/portfolio demonstration of industrial computer vision and automated surface-defect classification. It should be validated on production-specific data and operating conditions before being used in real manufacturing quality-control systems.

## Author

**Omkar Kadam**

Machine Learning / AI
