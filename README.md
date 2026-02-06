# Brain Tumor Segmentation using Deep Learning

Data is downloaded from: <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/47436290-13a0-4b30-8fbc-cdd7698f9c89" />


## Project Overview

This project explores **automatic brain tumor segmentation** from MRI scans using deep learning. Accurate segmentation of brain tumors, including Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET), is crucial for treatment planning, disease monitoring, and quantitative analysis in neuro-oncology.  

We investigate three architectures:
- **MultiResUNet** – robust multi-scale feature representation.  
- **SegResNet** – fast Dice convergence and strong performance.  
- **Swin-UNet (custom variant)** – transformer-based model for global context learning.  

Our experiments focus primarily on **MultiResUNet**, showing stable training, high Dice scores, and accurate delineation of tumor subregions on the BraTS 2020 dataset.

---

## Dataset

- **Brain Tumor Segmentation (BraTS) 2020 Challenge**  
- **Modalities used:** T1, T1Gd, T2, T2-FLAIR (we use first three modalities for 2D segmentation)  
- **Total MRI scans:** 369  
- **Slices:** 57,195 (32,773 without tumors, 24,422 with tumors)  
- **Split:** 70% training, 20% validation, 10% testing  

> Dataset includes expert-annotated masks for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

---

## Preprocessing

- Extract **central axial slices** from 3D MRI volumes  
- Resize slices to **224×224 pixels**  
- **Min-max normalization** for intensity standardization  
- Convert labels to **three-channel one-hot masks** for WT, TC, and ET  
- Data augmentation for training: **random horizontal flips**  

---

## Model Architectures

### 1. MultiResUNet
- Encoder-decoder with **MultiRes blocks** for multi-scale feature learning  
- **Residual connections** reduce semantic gaps between encoder and decoder  
- Output: 3-channel probability map for WT, TC, ET  
- Loss: **Dice loss**  
- Optimizer: **Adam**  
- Early stopping and **ReduceLROnPlateau** scheduler  

### 2. SegResNet
- Residual network with skip connections  
- Rapid convergence on Dice score  

### 3. Swin-UNet (custom)
- Transformer-based architecture with **shifted window attention**  
- Captures long-range dependencies  

> Only MultiResUNet results were fully validated in this project. SegResNet and Swin-UNet are included for future work.

---

## Training Details

- **Epochs:** 50  
- **Batch size:** 16  
- **Metrics:** Dice coefficient per tumor region (WT, TC, ET)  
- Stable training and validation curves observed, with minimal overfitting  

---

## Results

### Quantitative
- Smooth decrease in training and validation loss  
- Dice coefficient steadily improves over epochs  
- MultiResUNet achieves high overlap with ground-truth masks  

### Qualitative
- Tumor boundaries accurately delineated  
- WT, TC, ET correctly identified, including small Enhancing Tumor regions  
- Predicted masks maintain anatomical consistency and avoid false positives  

---

## Usage

1. Clone this repository:  
```bash
git clone https://github.com/<your-username>/brain-tumor-segmentation.git
cd brain-tumor-segmentation
