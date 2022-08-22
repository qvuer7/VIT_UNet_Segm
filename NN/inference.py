from NN.Transformer.Model.factory import load_model
from NN.Transformer.Model.utils import transformer_inference
import NN.utils.torch_conf  as ptu
import torch
from PIL import Image
from NN.utils.visualization import normalize, highlight
import torchvision
import torchvision.transforms.functional as F
import numpy as np

def inference(transformer, variant, unet, image):
    transformer.eval()
    unet.eval()
    image = image.to(ptu.device)
    seg_map, logits = transformer_inference(transformer, variant, image)
    seg_map = seg_map.to(ptu.device)
    data_to_UNet = torch.cat((image, seg_map), dim=1).to(ptu.device)
    data_to_UNet = data_to_UNet.to(ptu.device)
    logits = unet.forward(data_to_UNet)
    logits = torch.sigmoid(logits)
    logits = (logits > 0.5) * 1.0
    return logits

def inference_one_image(transformer, variant, unet, image_path):

    image_pil = Image.open(image_path)
    image = normalize(image_pil)
    image = image.unsqueeze(0)
    image = image.to(ptu.device)
    seg_map, logits = transformer_inference(transformer, variant, image)

    data_to_UNet = torch.cat((image, seg_map), dim=1).to(ptu.device)
    data_to_UNet = data_to_UNet.to(ptu.device)
    logits = unet.forward(data_to_UNet)
    logits = torch.sigmoid(logits)
    seg_map = (logits > 0.5) * 1.0

    image_ori, image_h = highlight(image, seg_map)
    image = torch.Tensor(image_h[0])
    image = image.permute(2, 0, 1)
    image = F.resize(image, [image_pil.size[1], image_pil.size[0]],
                     interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image
