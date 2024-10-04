# STRD-Net

Pytorch codes for [STRD-Net: A dual-encoder semantic segmentation network for urban green space extraction](https://ieeexplore.ieee.org/document/10671599)

## Environment

Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

You may need to do the following.

> conda create -n xxx python=3.8 -y
>
> conda activate xxx
>
> pip install -r requirements.txt

###requirements.txt

> torchvision==0.14.0
>
> torch==1.13.0
>
> timm==0.9.2
>
> numpy==1.24.2
>
> tqdm==4.65.0
>
> tensorboard==2.13.0
>
> sklearn



## Reference

-[ST-UNet](https://github.com/XinnHe/ST-UNet)

-[SwinTransformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)

-[SoftPool](https://github.com/alexandrosstergiou/SoftPool)

-[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)

## Citations

>@ARTICLE{10671599,
>author={Yu, Mouzhe and He, Liheng and Shen, Zhehui and Lv, Meng},
>journal={IEEE Transactions on Geoscience and Remote Sensing},
>title={STRD-Net: A Dual-Encoder Semantic Segmentation Network for Urban Green Space Extraction},
>year={2024},
>volume={62},}
