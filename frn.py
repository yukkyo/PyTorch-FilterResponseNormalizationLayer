import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import ReLU, LeakyReLU


class TLU(nn.Module):
    def __init__(self, inplace=True):
        """
        max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau
        """
        super(TLU, self).__init__()
        self.inplace = inplace
        self.tau = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def forward(self, x):
        return F.relu(x - self.tau, inplace=self.inplace) + self.tau

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class FRN(nn.Module):
    def __init__(self, num_features, init_eps=1e-6):
        """
        weight = gamma, bias = beta

        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.

        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = init_eps

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
            torch.tensor(init_eps), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = (x ** 2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * (nu2 + self.eps.abs())**(-0.5)

        # Scale and Bias
        x = self.weight * x + self.bias
        return x


def bnrelu_to_frn(module):
    """
    Convert 'BatchNorm2d + ReLU' to 'FRN + TLU'
    """
    mod = module
    before_name = None
    before_child = None
    is_before_bn = False

    for name, child in module.named_children():
        if is_before_bn and isinstance(child, (ReLU, LeakyReLU)):
            # Convert BN to FRN
            if isinstance(before_child, BatchNorm2d):
                mod.add_module(
                    before_name, FRN(num_features=before_child.num_features))
            else:
                raise NotImplementedError()

            # Convert ReLU to TLU
            mod.add_module(name, TLU(inplace=child.inplace))
        else:
            mod.add_module(name, bnrelu_to_frn(child))

        before_name = name
        before_child = child
        is_before_bn = isinstance(child, BatchNorm2d)
    return mod
