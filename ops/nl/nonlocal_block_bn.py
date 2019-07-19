import torch
import torch.nn.functional as F
from libs import InPlaceABN, InPlaceABNSync
from torch import nn
from torch.nn import init
import math

class _NonLocalNd_bn(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(_NonLocalNd_bn, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        self.bn_query=bn_nd(planes,requires_grad=False)
        self.bn_key=bn_nd(planes,requires_grad=False)
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        #init.constant_(self.norm.weight, 0)
        #init.constant_(self.norm.bias, 0)
        #self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.bn_query(self.conv_query(x))
        # [N, C', T, H', W']
        key = self.bn_key(self.conv_key(input_x))
        value = self.conv_value(input_x)

        # [N, C', T x H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', T x H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map/self.scale
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out = out.transpose(1, 2)
        # [N, C', T,  H, W]
        out = out.view(out.size(0), out.size(1), *x.size()[2:])
        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out = self.conv_out(out)
        # if self.norm is not None:
        #     out = self.norm(out)
        out = self.gamma * out

        out = residual + out
        return out


class NonLocal2d_bn(_NonLocalNd_bn):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False):
        super(NonLocal2d, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample, use_gn=use_gn, lr_mult=lr_mult, use_out=use_out)


class NonLocal3d_bn(_NonLocalNd_bn):

    def __init__(self, inplanes, planes, downsample, use_gn, lr_mult, use_out):
        super(NonLocal3d, self).__init__(dim=3, inplanes=inplanes, planes=planes, downsample=downsample, use_gn=use_gn, lr_mult=lr_mult, use_out=use_out)
