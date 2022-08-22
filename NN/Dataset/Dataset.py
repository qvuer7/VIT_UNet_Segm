from torch.utils.data import Dataset
import pandas as pd
from NN.utils.visualization import normalize, mask_aug, renormalize, highlight
import os
from PIL import Image
import matplotlib.pyplot as plt


class md(Dataset):
    def __init__(self, df, type='vit'):
        self.df = df
        self.type = type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_pil = Image.open(self.df.iloc[idx]['image']).convert('RGB')
        mask_pil = Image.open(self.df.iloc[idx]['mask'])


        image = normalize(image_pil, self.type)
        mask = mask_aug(mask_pil)

        return image, mask


def main():
    pass


if __name__ == '__main__':
    main()
