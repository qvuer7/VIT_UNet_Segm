from NN.Dataset.utils import STATS, IMAGE_SIZE
import numpy as np
import torchvision.transforms.functional as F
import torch



def normalize(image, type = 'vit', keep_dims = False):

    image_for_transformer = F.pil_to_tensor(image).float() / 255
    if keep_dims:

        t_w = round((IMAGE_SIZE[0] * image_for_transformer.shape[2]) / (image_for_transformer.shape[1]))
        image_for_transformer = F.resize(image_for_transformer, [IMAGE_SIZE[0], t_w])
    else:
        image_for_transformer = F.resize(image_for_transformer, IMAGE_SIZE)
    image_for_transformer = F.normalize(image_for_transformer, STATS[type]['mean'], STATS[type]['std'])

    return image_for_transformer

def mask_aug(mask, keep_dims = False):

    mask = F.pil_to_tensor(mask)
    if keep_dims:
        t_w = round((IMAGE_SIZE[0] * mask.shape[2]) / (mask.shape[1]))
        mask = F.resize(mask, [IMAGE_SIZE[0], t_w])
    else:
        mask = F.resize(mask, IMAGE_SIZE)

    mask = torch.round(mask.float() / 255)

    return mask


def renormalize(image, type = 'vit'):
    images = []
    for i in range(image.shape[0]):
        im = image[i, :, :, :].numpy()
        im = im.transpose((1, 2, 0))
        mean = np.array(STATS[type]['mean'])
        std = np.array(STATS[type]['std'])
        im = im * std + mean
        im = im * 255.0
        im = np.round(im.clip(0, 255)).astype(np.uint8)
        images.append(im)
    return images

def highlight(image, mask):
    images = []
    images_h = []
    for i in range(image.shape[0]):
        im = image[i, :, :, :].numpy()
        im = im.transpose((1, 2, 0))
        mean = np.array(STATS['vit']['mean'])
        std = np.array(STATS['vit']['std'])
        im = im * std + mean
        im = im * 255.0
        im = np.round(im.clip(0, 255)).astype(np.uint8)
        images.append(im)

        im_l = im.copy()
        m = mask[i, :, :, :].numpy()
        m = m.squeeze()
        human = np.where(m == 1)
        im_l[human[0], human[1], :] = [255, 0, 0]

        images_h.append(im_l)
    return images, images_h

def main():
    pass



if __name__ == '__main__':
    main()