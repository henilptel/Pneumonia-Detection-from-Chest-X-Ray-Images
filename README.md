# Pneumonia Detection from Chest X-Ray Images

## Project Overview
This project implements an ensemble-based deep learning approach for automatically detecting pneumonia from chest X-ray images. The system combines multiple convolutional neural network architectures to achieve superior diagnostic performance compared to individual models.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)

## Project Structure
```
├── chest_xray\                    
│   └── chest_xray\
│       ├── train\                 # Training dataset
│       │   ├── NORMAL\            # Normal X-ray images
│       │   └── PNEUMONIA\         # Pneumonia X-ray images
│       ├── val\                   # Validation dataset
│       │   ├── NORMAL\
│       │   └── PNEUMONIA\
│       └── test\                  # Test dataset
│           ├── NORMAL\
│           └── PNEUMONIA\
│
├── model_weights\                # Saved model weights
│   ├── resnet_model_ft.h5        
│   ├── densenet_model_ft.h5      
│   ├── vgg_model_ft.h5           
│   └── ensemble_model.h5         
│
├── Project.ipynb                 
├── ChestDataset.zip              
├── README.md                     
└── .gitignore                    
```

## Dataset
The project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle containing:
- Normal chest X-rays
- Pneumonia chest X-rays (including bacterial and viral pneumonia)

The dataset is organized into train, validation, and test sets. Images are preprocessed and augmented to improve model generalization.

## Installation

### Prerequisites
- Python 3.6+
- TensorFlow 2.x

### Setup
1. Clone the repository
2. Install required packages:
   ```
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
   ```
3. Extract dataset from ChestDataset.zip (will be done automatically in notebook)

## Methodology

### Data Preparation
- Image resizing to 224x224 pixels
- Data augmentation techniques including rotation, zooming, and brightness adjustment
- Class weighting to handle data imbalance

### Model Architecture
The project implements an ensemble of three CNN architectures:
1. **ResNet50** - Deep residual network with skip connections
2. **DenseNet121** - Dense convolutional network with dense blocks
3. **VGG16** - Classic deep CNN architecture

### Training Approach
- Two-phase training strategy:
  - Phase 1: Training with frozen pre-trained weights
  - Phase 2: Fine-tuning with selective layer unfreezing
- Learning rate scheduling and early stopping
- Weighted ensemble combining individual model predictions

## Results
The ensemble model achieves superior performance compared to individual models:

| Model | Accuracy |
|-------|----------|
| ResNet50 | ~90% |
| DenseNet121 | ~92% |
| VGG16 | ~89% |
| **Weighted Ensemble** | **~94%** |

The model demonstrates high precision and recall, making it suitable for clinical decision support.

## Usage
1. Open `Project.ipynb` in Jupyter Notebook or Google Colab
2. Run cells sequentially to:
   - Extract and prepare data
   - Train individual models
   - Create and evaluate ensemble model
3. Pre-trained models can be loaded from the `model_weights/` directory

