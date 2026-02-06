# -*- coding: utf-8 -*-
"""
BraTS_MultiResNet2.py
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

# Paths and Hyper parameters

data_dir = "/N/u/prpremk/BigRed200/brats_h5/data"
output_dir = "/N/u/prpremk/BigRed200/brats_results"
os.makedirs(output_dir, exist_ok=True)

batch_size = 8
img_size = 224
epochs = 50
alpha = 1.67

# Dataset Loader

class BraTSSliceDataset2D(Sequence):
    """
This code defines a custom Keras Sequence-based data loader designed to efficiently generate 2D slices from
BraTS .h5 MRI volumes for training segmentation models. For each batch, the loader reads HDF5 files from disk, extracts
the central axial slice from the multimodal MRI volume, and retains only the first three channels (e.g., FLAIR, T1,
T1ce). It performs per-channel min–max normalization, converts the BraTS mask into three one-hot channels representing
Whole Tumor, Tumor Core, and Enhancing Tumor, and resizes both images and masks to a uniform input resolution of 224×224.
Optional data augmentation is applied if provided. Finally, it returns batches of normalized slices and corresponding
one-hot encoded masks, enabling deterministic, memory-efficient training for 2D segmentation models.
"""
    def __init__(self, data_dir, file_indices=None, batch_size=8, transform=None, include_mask=True, img_size=224):
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
        if file_indices is not None:
            self.files = [self.files[i] for i in file_indices]
        self.batch_size = batch_size
        self.transform = transform
        self.include_mask = include_mask
        self.data_dir = data_dir
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def _normalize_channels(self, image):
        image = image.astype(np.float32)
        norm_img = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            mn, mx = image[:, :, c].min(), image[:, :, c].max()
            if mx - mn > 0:
                norm_img[:, :, c] = (image[:, :, c] - mn) / (mx - mn)
        return norm_img

    def _mask_to_channels(self, mask):
        wt = (mask > 0).astype(np.float32)
        tc = np.logical_or(mask == 1, mask == 4).astype(np.float32)
        et = (mask == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=-1)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs, batch_masks = [], []

        for f in batch_files:
            path = os.path.join(self.data_dir, f)
            with h5py.File(path, "r") as hf:
                img_data = hf["image"][:, :, :,:3] if hf["image"].ndim == 4 else hf["image"][:, :, :3]
                mask_data = hf["mask"][:] if ("mask" in hf.keys() and self.include_mask) else None

            # Middle slice
            if img_data.ndim == 4:
                slice_idx = img_data.shape[2] // 2
                image = img_data[:, :, slice_idx, :3]
            else:
                image = img_data

            if mask_data is not None:
                if mask_data.ndim == 3:
                    slice_idx = mask_data.shape[2] // 2
                    mask = mask_data[:, :, slice_idx]
                else:
                    mask = mask_data
            else:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

            image = self._normalize_channels(image)
            mask = self._mask_to_channels(mask)

            # Resize
            image = tf.image.resize(image, [self.img_size, self.img_size]).numpy()
            mask = tf.image.resize(mask, [self.img_size, self.img_size], method='nearest').numpy()

            if self.transform:
                augmented = self.transform(image, mask)
                image, mask = augmented

            batch_imgs.append(image)
            batch_masks.append(mask)

        return np.array(batch_imgs, dtype=np.float32), np.array(batch_masks, dtype=np.float32)

# MultiResUNet2D Model
"""
This code defines the MultiResUNet2D architecture, an enhanced version of U-Net designed for medical image segmentation.
It includes a custom MultiResBlock, which combines multiple convolutional paths with different receptive fields and a 
residual shortcut to capture rich multi-scale features. The network builds an encoder using stacked MultiResBlocks and 
max-pooling for downsampling, a bottleneck layer for deep feature extraction, and a decoder that upsamples the feature 
maps using transposed convolutions while concatenating corresponding encoder features (skip connections). The final 
output layer generates a multi-channel segmentation map using a sigmoid activation. Overall, the code constructs a full 
2D MultiResUNet model for tasks such as tumor segmentation from MRI slices.
"""
def ConvBNReLU(x, filters, kernel_size=3, padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def MultiResBlock(x, out_filters, alpha=1.67):
    W = int(alpha * out_filters)
    w1 = W // 6
    w2 = W // 3
    w3 = W - w1 - w2
    shortcut = layers.Conv2D(W, 1, padding='same')(x)
    x1 = ConvBNReLU(x, w1)
    x2 = ConvBNReLU(x1, w2)
    x3 = ConvBNReLU(x2, w3)
    out = layers.Concatenate()([x1, x2, x3])
    out = layers.BatchNormalization()(out)
    out = layers.Add()([out, shortcut])
    out = layers.ReLU()(out)
    return out

def MultiResUNet2D(input_shape=(224,224,3), out_channels=3, filters=[32,64,128,256], alpha=1.67):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    x1 = MultiResBlock(inputs, filters[0], alpha); p1 = layers.MaxPooling2D(2)(x1)
    x2 = MultiResBlock(p1, filters[1], alpha); p2 = layers.MaxPooling2D(2)(x2)
    x3 = MultiResBlock(p2, filters[2], alpha); p3 = layers.MaxPooling2D(2)(x3)
    x4 = MultiResBlock(p3, filters[3], alpha)  # bottleneck

    # Decoder
    u3 = layers.Conv2DTranspose(int(alpha*filters[2]), 2, strides=2, padding='same')(x4)
    c3 = layers.Concatenate()([u3, x3])
    u3 = MultiResBlock(c3, filters[2], alpha)

    u2 = layers.Conv2DTranspose(int(alpha*filters[1]), 2, strides=2, padding='same')(u3)
    c2 = layers.Concatenate()([u2, x2])
    u2 = MultiResBlock(c2, filters[1], alpha)

    u1 = layers.Conv2DTranspose(int(alpha*filters[0]), 2, strides=2, padding='same')(u2)
    c1 = layers.Concatenate()([u1, x1])
    u1 = MultiResBlock(c1, filters[0], alpha)

    outputs = layers.Conv2D(out_channels, 1, activation='sigmoid')(u1)
    model = models.Model(inputs, outputs)
    return model

# Dice and BCE Loss
"""
This block defines the Dice coefficient and related loss functions used for training a segmentation model. The dice_coeff
function computes the Dice similarity between predicted and ground-truth masks by flattening them, calculating overlap, 
and normalizing by the total foreground pixels in both masks. dice_loss converts this score into a loss value by 
subtracting it from 1, making higher overlap correspond to lower loss. The bce_dice_loss function combines Binary 
Cross-Entropy with Dice loss, balancing pixel-wise accuracy and region-overlap quality, which is a common and effective 
strategy in medical image segmentation.
"""
def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)
#
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# Preprocessing / Augmentation
"""
This code defines two preprocessing utility functions for preparing training images and masks. The preprocess function 
resizes both the MRI slice and its corresponding segmentation mask to a fixed spatial resolution, normalizes the image 
intensities to the range [0, 1], and preserves label integrity by using nearest-neighbor interpolation for masks. 
The augment function first applies these preprocessing steps and then performs a random horizontal flip, introducing 
controlled variability to improve model generalization while maintaining anatomical consistency.
"""
def preprocess(img, mask):
    img = tf.image.resize(img, (img_size, img_size))
    mask = tf.image.resize(mask, (img_size, img_size), method='nearest')
    img = img / 255.0
    return img, mask

def augment(img, mask):
    img, mask = preprocess(img, mask)
    img = tf.image.random_flip_left_right(img)
    return img, mask

# Dataset Splits
"""
This block partitions the available BraTS .h5 volumes into training, validation, and test subsets, then instantiates 
three data loader objects accordingly. It first lists and sorts all dataset files, computes a 70/20/10 split, and 
generates index ranges for each subset. Using these index lists, it constructs three BraTSSliceDataset2D generators: 
the training generator applies on-the-fly augmentation, while the validation and test generators apply only deterministic
 preprocessing. This setup ensures consistent batching, controlled data augmentation, and a reproducible separation of 
 data for model training and evaluation.
"""
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
total_size = len(all_files)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)

train_idx = list(range(0, train_size))
val_idx = list(range(train_size, train_size + val_size))
test_idx = list(range(train_size + val_size, total_size))

train_dataset = BraTSSliceDataset2D(data_dir, file_indices=train_idx, batch_size=batch_size, transform=augment)
val_dataset = BraTSSliceDataset2D(data_dir, file_indices=val_idx, batch_size=batch_size, transform=preprocess)
test_dataset = BraTSSliceDataset2D(data_dir, file_indices=test_idx, batch_size=batch_size, transform=preprocess)

# Instantiate and Compile Model
"""
This block sets up and trains the MultiResUNet2D model for brain tumor segmentation. It first instantiates the model 
with a specified input shape and output channels and compiles it using the combined Binary Cross-Entropy + Dice loss 
and the Dice coefficient as a metric.
"""
model = MultiResUNet2D(input_shape=(img_size,img_size,3), out_channels=3)
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coeff])

# Callbacks
"""
Three callbacks are defined:
1. ModelCheckpoint – saves the best model based on validation loss.
2. EarlyStopping – stops training if validation loss does not improve for 10 epochs and restores the best weights.
3. ReduceLROnPlateau – reduces the learning rate by half if validation loss stagnates for 5 epochs.
"""
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "multiresunet_brats.h5", save_best_only=True, monitor="val_loss"
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True ##5
)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
)

# Train Model
"""
the model is trained using the training dataset, validated on the validation dataset, and run for a specified 
number of epochs, with callbacks managing optimization and model saving automatically.
"""
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# Utility: Save Non-Empty Slice
"""
This code defines a utility function to visualize and save predictions for non-empty slices from a dataset. For each 
batch, it identifies the first slice where the specified mask channel (WT, TC, or ET) is non-empty. It then extracts 
the input image, ground-truth mask, and model-predicted mask for that slice, converts the prediction to a binary mask 
using a threshold, and displays them side by side using Matplotlib. Finally, the figure is saved to the output directory.
"""
def save_nonempty_slice(model, dataset, channel=0, threshold=0.5, prefix="val"):
    for batch_img, batch_mask in dataset:
        nonempty_indices = [i for i in range(len(batch_mask)) if batch_mask[i][:,:,channel].sum() > 1e-5]
        if len(nonempty_indices) == 0:
            continue
        idx = nonempty_indices[0]
        img = batch_img[idx]
        true_mask = batch_mask[idx][:,:,channel]
        pred_mask = model.predict(np.expand_dims(img, axis=0))[0][:,:,channel]
        pred_binary = (pred_mask > threshold).astype(np.uint8)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Input Image"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(true_mask, cmap="Reds", alpha=0.6); plt.title("GT Mask"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(pred_binary, cmap="Blues", alpha=0.6); plt.title("Pred Mask"); plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{prefix}_channel{channel}.png"))
        plt.close()
        break

# Save non-empty slices
save_nonempty_slice(model, val_dataset, channel=0, prefix="WT")
save_nonempty_slice(model, val_dataset, channel=1, prefix="TC")
save_nonempty_slice(model, val_dataset, channel=2, prefix="ET")

# Test Batch Visualization
"""
This block generates visual comparisons of predicted and ground-truth masks for the first test slice. It first retrieves
the first batch of images and masks from the test dataset, then obtains model predictions and converts them to binary 
masks using a threshold of 0.2. For each tumor subregion (Whole Tumor, Tumor Core, Enhancing Tumor), it plots the 
ground-truth mask in red and the predicted mask in blue side by side, adds titles, and saves the figures to the output 
directory. This provides a clear qualitative evaluation of the model’s segmentation performance on test data.
"""
test_imgs, test_masks = test_dataset[0]
pred_masks = model.predict(test_imgs)
pred_binary = (pred_masks > 0.2).astype("float32")
labels = ["WT", "TC", "ET"]

for i in range(3):
    plt.figure(figsize=(10,4))
    plt.suptitle(f"{labels[i]} segmentation")
    plt.subplot(1,2,1); plt.imshow(test_masks[0][:,:,i], cmap='Reds'); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(pred_binary[0][:,:,i], cmap='Blues'); plt.title("Prediction"); plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"test_slice0_{labels[i]}.png"))
    plt.close()