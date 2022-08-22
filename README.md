# Trasnformer NN ued for segmentation with additional UNet in the end of Transformer for human segmentation.
## Code provided in this directory able to perform training of UNet based on VIT(Vision Transformer) output as well as further inference of whole NN structure.
##### Content:
1. NN 
  - Transformer - Contains all files related to transformer model(structure, utils(loading, inferencing etc.))
  - UNet - Contains all files related to the UNet model(structure, utils)
  - Dataset
  - Utils
2. data - directory for your data used for UNet trianing possible to use any (e.g [This small dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset))

## Own configuration
To train your own UNet model based on VIT please configure /NN/train.py file for image, annotation and transformer model paths. 
Trasnformer model used in this work found in [here](https://github.com/rstrudel/segmenter) . Transformer models checkpoints and variants could be found [here](https://github.com/rstrudel/segmenter)


## Results
![Losses](https://github.com/qvuer7/VIT_Unet_Segm/blob/6a1796fd56adbb86568c5f693d6427056707969f/Screenshot%202022-08-17%20at%2020.16.05.png)
![Vizualisation](https://github.com/qvuer7/VIT_Unet_Segm/blob/6a1796fd56adbb86568c5f693d6427056707969f/Screenshot%202022-08-18%20at%2017.09.53.png)

