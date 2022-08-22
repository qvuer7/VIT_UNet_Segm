import torch

from NN.Transformer.Model.factory import load_model
from NN.Dataset.Dataset import md
from torch.utils.data import DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from NN.UNet.Model.UNet import UNet
import NN.utils.torch_conf as ptu
from NN.UNet.train.train_fncs import train_fn, val_fn
from torch.utils.tensorboard import SummaryWriter
import numpy as np


LEARNING_RATE = 0.01
BATCH_SIZE = 4
EPOCHS = 10
USE_GPU = False
ptu.set_gpu_mode(USE_GPU)
model_path = '/NN/Transformer/checkpoints/checkpoint.pth'  #Yout transformer moadel path
images_dir = '/data/VikramDataset/images/training'  #Yout images path
masks_dir = '/data/VikramDataset/annotations/training'  #Your annotations path

images_ld = sorted(os.listdir(images_dir))


if '.DS_Store' in images_ld:
    images_ld.remove('.DS_Store')

masks_ld = sorted(os.listdir(masks_dir))
if '.DS_Store' in masks_ld:
    masks_ld.remove('.DS_Store')
images = list(map(lambda x: os.path.join(images_dir, x), images_ld))
masks = list(map(lambda x: os.path.join(masks_dir, x), masks_ld))
dataset_df = pd.DataFrame({'image': images, 'mask': masks})

train_df, val_df = train_test_split(dataset_df, test_size=0.15)

train_ds = md(train_df)
val_ds = md(val_df)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

transformer, variant = load_model(model_path)
unet = UNet(n_channels=4, n_classes=1)

transformer.to(ptu.device)
unet.to(ptu.device)

optimizer = torch.optim.Adam(unet.parameters(), LEARNING_RATE)
u_t_loss, t_t_loss = train_fn(transformer, variant, unet, optimizer, train_dl)

best_val = np.Inf
writer = SummaryWriter()

for i in range(EPOCHS):

    unet_train_loss, transformer_train_loss = train_fn(transformer, variant, unet, optimizer, train_dl, )
    unet_val_loss, transformer_val_loss = val_fn(transformer, variant, unet, val_dl)
    writer.add_scalar("UNet_Loss/train", unet_train_loss, i)
    writer.add_scalar("UNet_Loss/val", unet_val_loss, i)
    writer.add_scalar("Transformer_Loss/train", transformer_train_loss, i)
    writer.add_scalar("Transformer_Loss/val", transformer_val_loss, i)
    if unet_val_loss < best_val:
        torch.save(unet.state_dict(), 'best-model.pt')
        best_val = unet_val_loss
        print('Model Saved')
    print(f' epoch number {i}. UNET Train Loss: {unet_train_loss} Validation Loss : {unet_val_loss}')
    print(
        f' epoch number {i}.  TRANSFORMER Train Loss: {transformer_train_loss} Validation Loss : {transformer_val_loss}')
