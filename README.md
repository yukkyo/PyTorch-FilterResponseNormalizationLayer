## PyTorch-Filter Response Normalization Layer(FRN)

PyTorch implementation of Filter Response Normalization Layer(FRN)

[\[1911\.09737\] Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)

## 0. How to apply FRN to your model

### 0.1 How to convert your `BatchNorm2d + ReLU` to `FRN + TLU`

`bnrelu_to_frn()` is function for converting "BatchNorm2d + ReLU(or LeakyReLU)" to "FRN + TLU".  
So the ReLUs which previous layer is not BatchNorm2d, is not converted. Similarly for BatchNorm2d.

```python3
from frn import bnrelu_to_frn

# Classification example
import torchvision.models as models
model_cls = models.resnet18(pretrained=False)
model_cls = bnrelu_to_frn(model_cls)

# Segmentation example
import segmentation_models_pytorch as smp
model_seg = smp.Unet('resnet18', classes=3, activation='softmax')
model_seg = bnrelu_to_frn(model_seg)
```

## 1. Experiment(Classification)

We use [Best Artworks of All Time \| Kaggle](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) dataset.
This dataset contains 49 artists and their pictures.  
In this experiment, we classify artist by picture.


### 1.0 Assumed libraries

- torch==1.3.1
- torchvision==0.4.2
- cnn-finetune==0.6.0
- catalyst==19.11.6
- albumentations==0.4.3

### 1.1 Get dataset

If you can use kaggle API command, you can download easily

```bash
$ cd input
$ kaggle datasets download -d ikarus777/best-artworks-of-all-time
$ unzip best-artworks-of-all-time.zip -d artworks
```

Or download directly from [Best Artworks of All Time \| Kaggle](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)


I assume the following directory structure.

```text
input
├── artworks
│   ├── artists.csv
│   ├── images
│   │   └── images
│   │       ├── Alfred_Sisley
│   │       │   ├── Alfred_Sisley_1.jpg
│   │       │   ├── Alfred_Sisley_10.jpg
│   │       │   ├── ...
```

### 1.2 Train(and Valid)

#### Compare pre-train or not

FRN with FP16 is not worked now...

```bash
$ python train_cls.py --use-pretrain --model se_resnext50_32x4d --fp16
$ python train_cls.py --use-pretrain --model se_resnext50_32x4d --frn
```

### 1.3 Results

Coming soon...
