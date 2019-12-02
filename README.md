
## PyTorch implementation Filter Response Normalization Layer(FRN)

[\[1911\.09737\] Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)


## How to convert your model

`bnrelu_to_frn()` is function for converting "BatchNorm2d + ReLU(or LeakyReLU)" to "FRN + TLU".  
So the ReLUs which previous layer is not BatchNorm2d, is not converted.  
Similarly for BatchNorm2d.


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

## Experiment

I want to do a simple experiment on Classification and Segmentation.  
Coming soon...
