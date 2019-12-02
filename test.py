import torch
import torchvision.models as models
import segmentation_models_pytorch as smp
from torchsummary import summary

from frn import bnrelu_to_frn


def test_clsmodel():
    model = models.resnet18(pretrained=False)

    # Before model
    summary(model, (3, 128, 128), batch_size=2)

    # After model
    model = bnrelu_to_frn(model)
    summary(model, (3, 128, 128), batch_size=2)


def test_segmodel():
    model = smp.Unet('resnet18', classes=3, activation='softmax')

    print('*'*30)
    print(f'before_model: \n{model}')
    print('*'*30)

    x = torch.rand(2, 3, 128, 128)
    output = model(x)
    print(f'input.size(): {x.size()}')
    print(f'output.size(): {output.size()}')

    # Convert model
    model = bnrelu_to_frn(model)
    print('*'*30)
    print(f'after_model: \n{model}')
    print('*'*30)
    output = model(x)
    print(f'input.size(): {x.size()}')
    print(f'output.size(): {output.size()}')


def test(cls=True):
    if cls:
        test_clsmodel()
    else:
        test_segmodel()


if __name__ == '__main__':
    test(cls=False)
