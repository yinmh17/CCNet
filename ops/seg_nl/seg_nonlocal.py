import torch
import torch.nn.functional as F
from libs import InPlaceABN, InPlaceABNSync
from torch import nn
from torch.nn import init
import math

class SegNonLocal2d(nn.Module):

    def __init__(self, inplanes, planes, downsample=False,  lr_mult=None, use_out=False, k_seg=True, q_seg=True):
        conv_nd = nn.Conv2d
        if downsample:
            max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            max_pool = None
        bn_nd = nn.BatchNorm2d

        super(SegNonLocal2d, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
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
        self.k_seg = k_seg
        self.q_seg = q_seg

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

    def forward(self, x, x_dsn):
        # [N, C, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
            k_dsn = self.downsample(x_dsn)
        else:
            input_x = x
            k_dsn = x_dsn
            
        # [N, 19, H' x W']
        k_dsn = k_dsn.detach().view(k_dsn.size(0), 19, k_dsn.size(2)*k_dsn.size(3))
        # [N, 19, H x W]
        q_dsn = x_dsn.detach().view(x.size(0), 19, x.size(2)*x.size(3))
        # [N, C', H, W]
        query = self.conv_query(x)
        # [N, C', H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)
        
        if self.k_seg:
            # [N, C', 19]
            key = torch.bmm(key, k_dsn.transpose(1,2))
            value = torch.bmm(value, k_dsn.transpose(1,2))
        if self.q_seg:
            # [N, C', 19]
            query = torch.bmm(query, q_dsn.transpose(1,2))
            
        
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map/self.scale
        sim_map = self.softmax(sim_map)

        # [N, H x W, C']
        out = torch.bmm(sim_map, value.transpose(1, 2))
        if self.q_seg:
            out = torch.bmm(q_dsn.transpose(1,2), out)
        # [N, C', H x W]
        out = out.transpose(1, 2)
        # [N, C', H, W]
        out = out.view(out.size(0), out.size(1), *x.size()[2:])
        # [N, C, H, W]
        if self.conv_out is not None:
            out = self.conv_out(out)
        # if self.norm is not None:
        #     out = self.norm(out)
        out = self.gamma * out

        out = residual + out
        return out
