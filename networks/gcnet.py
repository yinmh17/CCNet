#import encoding.nn as nn
#import encoding.functions as F
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os

from libs import InPlaceABN, InPlaceABNSync
from ops import NonLocal2d, NonLocal2d_bn, NonLocal2dCos, ContextBlock, MultiheadBlock, MultiheadSpatialBlock, MultiRelationBlock
from ops import MultiheadRelationBlock, GloreUnit, ProjMultiheadBlock, ProjSpatialBlock, MaskNonLocal2d, NonLocal2dGc


BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, 
                 fist_dilation=1, multi_grid=1, with_att=False, att=None, att_pos=None, att_loc=None, rank=None):
        super(Bottleneck, self).__init__()
        self.rank=rank
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.expansion=4
        
        self.with_att=with_att
        self.att=att
        self.att_pos=att_pos
        self.att_loc=att_loc
        
        if self.att_pos == 'after_3x3':
            att_inplanes = planes
        else:
            att_inplanes = planes * self.expansion
            
        if self.with_att:
            if self.att == 'ct':
                self.context_block = ContextBlock(att_inplanes, ratio=1./4, one_fc=True)
            elif self.att == 'nl':
                self.context_block = NonLocal2d(att_inplanes, att_channels // 2)
            elif self.att == 'multi_gc':
                self.context_block =  MultiheadBlock(att_inplanes, ratio=1./4, one_fc=True, 
                                                     head_num=8, pre_group=1, post_group=8)
            elif self.att == 'glore' and self.rank in self.att_loc:
                self.context_block = GloreUnit(att_inplanes, att_inplanes//4)
            else:
                self.context_block=None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.with_att :
            if self.att != 'glore':
                if self.att_pos == 'after_3x3':
                    out = self.context_block(out)
            elif self.att_pos == 'after_3x3' and self.rank in self.att_loc:
                out = self.context_block(out)
                

        out = self.conv3(out)
        out = self.bn3(out)
        if self.with_att :
            if self.att != 'glore':
                if self.att_pos == 'after_1x1':
                    out = self.context_block(out)
            elif self.att_pos == 'after_1x1' and self.rank in self.att_loc:
                out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)
        if self.with_att :
            if self.att != 'glore':
                if self.att_pos == 'after_add':
                    out = self.context_block(out)
            elif self.att_pos == 'after_add' and self.rank in self.att_loc:
                out = self.context_block(out)

        return out

class GCBModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, type='nl_bn'):
        super(GCBModule, self).__init__()
        assert type in ['baseline','gcb', 'nl', 'nl_bn', 'nl_gc', 'nl_cos', 'mask_nl', 'multi', 'multi_spatial', 'multi_relation', 'multihead_relation', 'glore', 'proj_multi', 'proj_spatial']
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        if type == 'gcb':
            self.ctb = ContextBlock(inter_channels, ratio=1./4, one_fc=True)
        elif type == 'nl':
            self.ctb = NonLocal2d(inter_channels, inter_channels // 2)
        elif type == 'nl_bn':
            self.ctb = NonLocal2d_bn(inter_channels, inter_channels // 2, downsample=False, whiten_type=['channel'])
        elif type == 'multi':
            self.ctb = MultiheadBlock(inter_channels, ratio=1./4, one_fc=True, 
                                      head_num=8, pre_group=1, post_group=8)
        elif type == 'multi_relation':
            self.ctb = MultiRelationBlock(inter_channels, ratio=1.0/4, one_fc=True, mask_num=8, relation_num=8,
                                          pre_group=1, post_group=1, geo_feature_dim=64, key_feature_dim=64)
        elif type == 'multihead_relation':
            self.ctb = MultiheadRelationBlock(inter_channels, ratio=1.0/4, one_fc=True, head_num=8,
                                          pre_group=1, post_group=1, geo_feature_dim=64, key_feature_dim=64)
        elif type == 'multi_spatial':
            self.ctb = MultiheadSpatialBlock(inter_channels, ratio=1./4, head_num=8)
        elif type == 'glore':
            self.ctb = GloreUnit(inter_channels, inter_channels//4, interact = 'graph')
        elif type == 'proj_multi':
            self.ctb = ProjMultiheadBlock(inter_channels, ratio=1./4, one_fc=True, mask_num=1, pre_group=1, post_group=1)
        elif type == 'proj_spatial':
            self.ctb = ProjSpatialBlock(inter_channels, ratio=1./4, one_fc=True, mask_num=8, pre_group=1, post_group=1, share_proj=True)
        elif type == 'baseline':
            self.ctb = None
        elif type == 'mask_nl':
            self.ctb = MaskNonLocal2d(inter_channels, inter_channels // 2, mask_type = 'softmax', use_key_mask=False, use_query_mask=True, mask_pos='before')
        elif type == 'nl_cos':
            self.ctb = NonLocal2dCos(inter_channels, inter_channels // 2)
        elif type == 'nl_gc':
            self.ctb = NonLocal2dGc(inter_channels, inter_channels // 2, downsample=True)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        if self.ctb is not None:
            for i in range(recurrence):
                output = self.ctb(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, with_att, att, att_stage, att_pos, att_location):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0], with_att=with_att*att_stage[0], att=att, att_pos=att_pos, att_loc=att_location[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, with_att=with_att*att_stage[1], att=att, att_pos=att_pos, att_loc=att_location[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, with_att=with_att*att_stage[2], att=att, att_pos=att_pos, att_loc=att_location[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1), with_att=with_att*att_stage[3], att=att, att_pos=att_pos, att_loc=att_location[3])
        #self.layer5 = PSPModule(2048, 512)
        self.head = GCBModule(2048, 512, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, with_att=False, att=None, att_pos=None, att_loc=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, 
                            multi_grid=generate_multi_grid(0, multi_grid), with_att=with_att, att=att, att_pos=att_pos, att_loc=att_loc, rank=0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, 
                                multi_grid=generate_multi_grid(i, multi_grid), with_att=with_att, att=att, att_pos=att_pos, att_loc=att_loc, rank=i))

        return nn.Sequential(*layers)

    def forward(self, x, recurrence=1):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x, recurrence)
        return [x, x_dsn]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, with_att=False, att='glore', att_stage=[False, True, True, False], att_pos='after_add', att_location=[[],[0,2],[5,11,17],[]])
    return model
