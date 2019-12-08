## PyTorch-Filter Response Normalization Layer(FRN)

PyTorch implementation of Filter Response Normalization Layer(FRN)

[\[1911\.09737\] Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)

## 0. How to apply FRN to your model

Add your own FRN and TLU.
For now, it is difficult to change easily with functions.
(Because many models use the same ReLU in various places.)


## 1. Experiment(Classification)

We use [Best Artworks of All Time \| Kaggle](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) dataset.
This dataset contains 49 artists and their pictures.  
In this experiment, we classify artist by picture.


### 1.0 Assumed libraries

- torch==1.3.1
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
$ python train_cls.py --fp16
$ python train_cls.py --frn --fp16
```

### 1.3 Results

Coming soon...
