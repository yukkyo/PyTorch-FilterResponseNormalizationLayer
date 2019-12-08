import torch
from torchsummary import summary

from senet import se_resnext50_32x4d
from senet_frn import se_resnext50_32x4d_frn


def test_clsmodel():
    # Base Model
    model = se_resnext50_32x4d(pretrained=None)
    model.last_linear = torch.nn.Linear(512 * 16, 2)
    summary(model, (3, 256, 256), batch_size=2)

    # Use FRN Model
    model = se_resnext50_32x4d_frn(pretrained=None)
    model.last_linear = torch.nn.Linear(512 * 16, 2)
    summary(model, (3, 256, 256), batch_size=2)


if __name__ == '__main__':
    test_clsmodel()
