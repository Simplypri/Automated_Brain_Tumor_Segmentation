import os
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, EnsureTyped
)
from monai.data import Dataset
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.optimizers import Novograd


def main():

	images_dir = "/N/slate/vaperu/DLS_Project/Jobsubmission/Data/image"
	labels_dir = "/N/slate/vaperu/DLS_Project/Jobsubmission/Data/mask"
	images = sorted(glob.glob(os.path.join(images_dir, "*.png")))  # or *.jpg, *.tif
	labels = sorted(glob.glob(os.path.join(labels_dir, "*.png")))
	data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]


	val_frac = 0.2
	val_size = int(len(data_dicts) * val_frac)
	train_files = data_dicts[:-val_size]
	val_files = data_dicts[-val_size:]


	train_transforms = Compose([
				LoadImaged(keys=["image", "label"]),
				EnsureChannelFirstd(keys=["image", "label"]),  # Add channel dim
				ScaleIntensityd(keys=["image"]),               # Normalize to 0-1
				RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # horizontal flip
				RandRotate90d(keys=["image", "label"], prob=0.5),
				EnsureTyped(keys=["image", "label"]),
			])

	val_transforms = Compose([
				LoadImaged(keys=["image", "label"]),
				EnsureChannelFirstd(keys=["image", "label"]),
				ScaleIntensityd(keys=["image"]),
				EnsureTyped(keys=["image", "label"]),
			])

			# ------------------------------
			# Datasets & Loaders
			# ------------------------------
	train_ds = Dataset(train_files, train_transforms)
	val_ds   = Dataset(val_files, val_transforms)

	train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
	val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

			# ------------------------------
			# Model, Loss, Optimizer
			# ------------------------------

	model = SegResNet(
				spatial_dims=2,
				in_channels=4,
				out_channels=3,
				init_filters=32,
				blocks_down=[1,2,2,4],
				blocks_up=[1,1,1],
				dropout_prob=0.1
			)

	loss_function = DiceLoss(sigmoid=True)
	optimizer = Novograd(model.parameters(), lr=1e-3)
	dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

			# ------------------------------
			# Training Loop + Track metrics
			# ------------------------------
	max_epochs = 50
	train_losses, val_losses = [], []
	train_dices, val_dices = [], []

	for epoch in range(max_epochs):
		print(f"Epoch {epoch+1}/{max_epochs}")
		dice_metric.reset()
		# TRAIN
		model.train()
		train_loss, train_dice = 0, 0
		for batch in train_loader:
			inputs = batch["image"]
			labels = batch["label"]
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_function(outputs, labels)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			with torch.no_grad():
				pred = torch.sigmoid(outputs)
				train_dice += dice_metric(pred, labels).mean().item()

		train_loss /= len(train_loader)
		train_dice /= len(train_loader)
		train_losses.append(train_loss)
		train_dices.append(train_dice)


		model.eval()
		val_loss, val_dice = 0, 0
		with torch.no_grad():
			for batch in val_loader:
				inputs = batch["image"]
				labels = batch["label"]

				outputs = model(inputs)
				val_loss += loss_function(outputs, labels).mean().item()
				pred = torch.sigmoid(outputs)
				val_dice += dice_metric(pred, labels).mean().item()

		val_loss /= len(val_loader)
		val_dice /= len(val_loader)
		val_losses.append(val_loss)
		val_dices.append(val_dice)

		print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

		
		os.makedirs("checkpoints", exist_ok=True)
		torch.save(model.state_dict(), f"checkpoints_1/segresnet2d_epoch_{epoch+1}.pth")

	plt.figure(figsize=(10,5))
	plt.plot(train_losses, label="Train Loss")
	plt.plot(val_losses, label="Val Loss")
	plt.title("Loss Curves")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid()
	plt.show()


pass


if __name__ == "__main__":
	main()


