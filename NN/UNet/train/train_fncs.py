from tqdm import tqdm
from NN.Transformer.Model.utils import transformer_inference
import torch
from torch import nn
import NN.utils.torch_conf as ptu
#model.to(device)
#

def train_fn(transformer, variant,model, optimizer, data_loader):
    total_loss_u = 0
    total_loss_t = 0
    transformer.eval()
    model.train()
    i = 0
    for image, mask in tqdm(data_loader):
        print(i)
        i+=1
        image = image.to(ptu.device)
        mask = mask.to(ptu.device)
        seg_map, logits_t = transformer_inference(transformer, variant, image)
        transformer_loss = nn.BCEWithLogitsLoss()(logits_t, mask)
        seg_map = seg_map.to(ptu.device)
        data_to_UNet = torch.cat((image, seg_map), dim=1).to(ptu.device)

        optimizer.zero_grad()
        logits_u, unet_loss = model.forward(data_to_UNet, mask=mask)
        unet_loss.backward()
        optimizer.step()
        total_loss_u += unet_loss.item()
        total_loss_t += transformer_loss.item()

    return total_loss_u/len(data_loader), total_loss_t/len(data_loader)


def val_fn(transformer, variant,model, data_loader):
    total_loss_u = 0
    total_loss_t = 0
    transformer.eval()
    model.eval()
    for image, mask in data_loader:
        image = image.to(ptu.device)
        mask = mask.to(ptu.device)
        seg_map, logits_t = transformer_inference(transformer, variant, image)
        transformer_loss = nn.BCEWithLogitsLoss()(logits_t, mask)
        seg_map = seg_map.to(ptu.device)
        data_to_UNet = torch.cat((image, seg_map), dim=1).to(ptu.device)


        logits_u, unet_loss = model.forward(data_to_UNet, mask=mask)

        total_loss_u += unet_loss.item()
        total_loss_t += transformer_loss.item()

    return total_loss_u/len(data_loader), total_loss_t/len(data_loader)


