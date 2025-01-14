import torch
import torch.nn.functional as F
from libs import InPlaceABN, InPlaceABNSync
from torch import nn
from torch.nn import init
import math

class MaskNonLocal2d(nn.Module):

    def __init__(self, inplanes, planes, downsample=False, use_gn=False, 
                 lr_mult=None, use_out=True, out_bn=True, mask_type='common', use_key_mask=True, 
                 use_query_mask=False, mask_pos='after', whiten_type=[None], temperature=None, use_softmax=True):

        assert mask_type in ['softmax', 'sigmoid', 'common']
        assert mask_pos in ['before', 'after']
        conv_nd = nn.Conv2d
        if downsample:
            max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            max_pool = None
        bn_nd = nn.BatchNorm2d


        super(MaskNonLocal2d, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_query_mask==True:
            self.conv_query_mask=conv_nd(inplanes, 1, kernel_size=1)
        if use_key_mask==True:
            self.conv_key_mask=conv_nd(inplanes, 1, kernel_size=1)
        if mask_type=='sigmoid':
            self.sigmoid_key=nn.Sigmoid()
            self.sigmoid_query=nn.Sigmoid()
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)
        self.use_key_mask=use_key_mask
        self.use_query_mask=use_query_mask
        self.mask_type=mask_type
        self.mask_pos=mask_pos
        self.temperature=temperature
        self.whiten_type=whiten_type
        self.use_softmax=use_softmax

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
        # [N, C, H, W]
        residual = x
        # [N, C, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x
            
        batch, channel, _, _ = input_x.size()

        # [N, C', H, W]
        query = self.conv_query(x)
        # [N, C', H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)
        
        if self.use_key_mask:
        # [N, 1, H', W'] --> [N, 1, H'x W']
            key_mask=self.conv_key_mask(input_x).view(batch, -1).unsqueeze(1)
        if self.use_query_mask:
        # [N, 1, H, W] --> [N, H x W, 1]
            query_mask=self.conv_query_mask(x).view(batch,-1).unsqueeze(-1)
        
        
        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)
        
        if 'channel' in self.whiten_type :
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        
        if self.mask_pos == 'before':
            if self.use_key_mask:
                key += key_mask
            if self.use_query_mask:
                query += query_mask.permute(0,2,1)
            # [N, H x W, H' x W']
            sim_map = torch.bmm(query.transpose(1, 2), key)
            sim_map = sim_map/self.scale
            if self.temperature:
                sim_map = sim_map/self.temperature
            if self.use_softmax:
                sim_map = self.softmax(sim_map)
        
        if self.mask_pos =='after':
            sim_map = torch.bmm(query.transpose(1, 2), key)
            sim_map = sim_map/self.scale
            if self.temperature:
                sim_map = sim_map/self.temperature
            if self.mask_type=='softmax':
                if self.use_key_mask:
                    sim_map += key_mask
                if self.use_query_mask:
                    sim_map += query_mask
                if self.use_softmax:
                    sim_map = self.softmax(sim_map)

            elif self.mask_type == 'sigmoid':
                if self.use_softmax:
                    sim_map = self.softmax(sim_map)
                if self.use_key_mask:
                    key_mask=self.sigmoid_key(key_mask)
                    sim_map=sim_map*key_mask
                if self.use_query_mask:
                    query_mask=self.sigmoid_query(query_mask)
                    sim_map=sim_map*query_mask
                    
            elif self.mask_type == 'common':
                if self.use_key_mask:
                    sim_map += key_mask
                if self.use_query_mask:
                    sim_map += query_mask
                if self.use_softmax:
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
        if self.out_bn:
            out = self.out_bn(out)
        out = self.gamma * out

        out = residual + out
        return out
