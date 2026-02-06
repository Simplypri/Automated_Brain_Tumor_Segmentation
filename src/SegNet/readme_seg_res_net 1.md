# SegResNet‑2D Brain MRI Tumor Segmentation (MONAI)

## Overview
This repository contains a **2D medical image segmentation pipeline** implemented using **MONAI** and **PyTorch**. The code trains a **SegResNet** model for multi‑class brain MRI tumor segmentation using paired image–mask datasets. Model performance is evaluated using **Dice Loss**, with training and validation curves plotted for analysis.

This implementation is suitable for academic coursework, research prototyping, and benchmarking segmentation models on medical imaging datasets.

---

## Features
- 2D **SegResNet** architecture (MONAI)
- Dice Loss optimization
- Data augmentation (flip, rotation)
- Training / validation split
- Model checkpoint saving
- Loss score visualization

---

## Dataset Structure
Ensure your dataset follows the structure below:

```
Data/
├── image/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
├── mask/
│   ├── mask_001.png
│   ├── mask_002.png
│   └── ...
```

> Images and masks must be **aligned and sorted identically**.

---

## Model Architecture
- **Network**: SegResNet (2D)
- **Input channels**: 4
- **Output channels**: 3 (multi‑class segmentation)
- **Initial filters**: 32
- **Dropout probability**: 0.1

---

## Transforms Used
### Training
- Load image and label
- Ensure channel‑first format
- Intensity scaling (0–1)
- Random horizontal flip
- Random 90‑degree rotation

### Validation
- Load image and label
- Ensure channel‑first format
- Intensity scaling

---

## Training Details
- **Loss Function**: DiceLoss (sigmoid)
- **Optimizer**: Novograd
- **Learning Rate**: 1e‑3
- **Batch Size**: 1
- **Epochs**: 50
- **Validation Split**: 20%

Model checkpoints are saved after every epoch.

---

## Installation
Create a virtual environment and install dependencies:

```bash
pip install torch monai matplotlib
```

(Optional but recommended)
```bash
pip install numpy scipy
```

---

## Running the Code
Execute the training script:

```bash
python monai_7.py
```

Checkpoints will be saved to:
```
checkpoints_1/
```

---

## Output
- Training & validation loss curves
- Saved model weights per epoch

---

## Results
The model demonstrates stable convergence using Dice loss, with improved Dice scores over epochs, indicating effective segmentation learning. Data augmentation improves generalization performance on the validation set.



---

## References
1. MONAI Consortium. *MONAI: An open‑source framework for deep learning in healthcare.*
2. Isensee et al., *nnU‑Net: a self‑configuring method for deep learning‑based biomedical image segmentation*

---

## Author
**Varalakshmi Perumal**  
PhD in Intelligent Systems Engineering (Bioengineering)
Luddy School of Informatics, Computing, and Engineering
Indiana University Bloomington

---

## License
This project is intended for **academic use only**.

