import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

# Block used to construct the network
class ConvNextBlock(nn.Module):

    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size = 7, expand_ratio = 4, stride = 1, drop_path=0.,layer_scale_init_value=1e-6):
        super().__init__()
        padding = padding = (kernel_size - 1) // 2 
        self.dwconv = nn.Conv2d(dim, dim, kernel_size = kernel_size, stride = 1,  padding=padding, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
  
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class ResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio, stride = 1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(dim, dim , kernel_size, stride = stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride = stride , padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # print("ResNet forward : ")
        # print (f"{identity.shape = }")
        # print (f"{out.shape = }")
        
        out += identity

        out = self.relu(out)
        
        return out


class InvBottleNeckBlock(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio = 4, stride = 1):
        super(InvBottleNeckBlock, self).__init__()
        
        hidden_dim = round(dim * expand_ratio)
        
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride = 1, padding=(kernel_size - 1) // 2, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # print("InvertedBottleNeck forward : ")
        # print (f"{identity.shape = }")
        # print (f"{out.shape = }")

        out += identity
        out = self.relu(out)
        
        return out


def conv3x3(ch_in, ch_out, stride):
  return (
      nn.Sequential(
          nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
          nn.BatchNorm2d(ch_out),
          nn.ReLU6(inplace=True)
      )
  )


# Convolution 1x1
def conv1x1(ch_in, ch_out):
  return (
      nn.Sequential(
          nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
          nn.BatchNorm2d(ch_out),
          nn.ReLU6(inplace=True)
      )
  )