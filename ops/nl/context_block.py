import torch
import torch.nn.functional as F
from torch import nn
import math
from mmcv.cnn import constant_init, kaiming_init
from functools import partial


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


def all_zero_init(m):
    if isinstance(m, nn.Sequential):
        for sub_m in m.modules():
            if isinstance(sub_m, nn.Conv2d):
                constant_init(sub_m, val=0)
                m.inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


def all_kaiming_init(m, mode):
    if isinstance(m, nn.Sequential):
        for sub_m in m.modules():
            if isinstance(sub_m, nn.Conv2d):
                kaiming_init(sub_m, mode=mode)
                m.inited = True
    else:
        kaiming_init(m, mode=mode)
        m.inited = True


def get_norm_op(norm, inplanes):
    if norm == 'affine':
        out_op = nn.BatchNorm2d(num_features=inplanes, momentum=0.)
    elif norm == 'ln':
        out_op = nn.LayerNorm([inplanes, 1, 1])
    else:
        out_op = None
    return out_op


init_methods = {
    'last_zero': last_zero_init,
    'all_zero': all_zero_init,
    'fan_out': partial(all_kaiming_init, mode='fan_out')
}


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, groups, num_head, pool, fusions, norm, style, lr_mult, drop_out, init_method, out_op):
        print('inplanes: {} planes: {} groups: {} num_head: {} pool: {} fusion: {} norm: {} style: {} '
              'lr_mult: {} drop_out: {} init_method: {} out_op: {}'.format(
            inplanes, planes, groups, num_head, pool, fusions, norm, style, lr_mult, drop_out, init_method, out_op
        ))
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att', 'multi_att']
        assert all([f in ['channel_add', 'channel_mul', 'spatial_add', 'spatial_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        assert len(fusions) > 1 or fusions[0] != 'spatial_add', 'spatial_add only is not supported'
        assert norm in [None, 'bn', 'gn', 'ln', 'bln', 'sbln', 'relu']
        assert style in ['senet', 'saliency']
        assert init_method in ['last_zero', 'all_zero', 'fan_out']
        assert out_op in [None, 'affine', 'ln', 'bln', 'sbln']
        self.inplanes = inplanes
        self.planes = planes
        self.groups = groups
        self.num_head = num_head
        self.pool = pool
        self.fusions = fusions
        self.norm = norm
        self.style = style
        self.init_method = init_method
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, num_head, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = self.build_bottle_neck()
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = self.build_bottle_neck()
        else:
            self.channel_mul_conv = None
        if 'spatial_add' in fusions:
            self.spatial_add_conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        else:
            self.spatial_add_conv = None
        if 'spatial_mul' in fusions:
            self.spatial_mul_conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        else:
            self.spatial_mul_conv = None
        if drop_out is not None:
            self.drop_out = nn.Dropout2d(drop_out)
        else:
            self.drop_out = None
        self.out_op = get_norm_op(norm=out_op, inplanes=inplanes)
        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        init_function = init_methods[self.init_method]
        if self.channel_add_conv is not None:
            init_function(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            init_function(self.channel_mul_conv)
        if self.spatial_add_conv is not None:
            init_function(self.spatial_add_conv)
        if self.spatial_mul_conv is not None:
            init_function(self.spatial_mul_conv)

        if self.out_op is not None:
            constant_init(self.out_op, 0)

    def build_bottle_neck(self):
        if self.planes is not None:
            if self.norm is not None:
                if self.norm == 'relu':
                    bottle_neck = nn.Sequential(
                        nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
                    )
                else:
                    bottle_neck = nn.Sequential(
                        nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                        get_norm_op(self.norm, self.planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
                    )
            else:
                bottle_neck = nn.Sequential(
                    nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
                )
        else:
            bottle_neck = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, groups=self.groups)
        return bottle_neck

    def spatial_pool(self, x, input_x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, nHead, C//nHead, H * W]
            input_x = input_x.view(batch, self.num_head, channel//self.num_head, height * width)
            # [N, nHead, H, W]
            context_mask = self.conv_mask(x)
            # [N, nHead, H * W]
            context_mask = context_mask.view(batch, self.num_head, height * width)
            # [N, nHead, H * W]
            context_mask = self.softmax(context_mask)
            # [N, nHead, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, nHead, C//nHead, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        elif self.pool == 'multi_att':
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.view(batch, 1, channel, height * width)
            # [N, nHead, H, W]
            context_mask = self.conv_mask(x)
            # [N, nHead, H * W]
            context_mask = context_mask.view(batch, self.num_head, height * width)
            # [N, nHead, H * W]
            context_mask = self.softmax(context_mask)
            # [N, nHead, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, nHead, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1]
            context = context.sum(dim=1)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(input_x)

        return context

    def forward(self, x):
        batch, channel, height, width = x.size()

        if self.style == 'senet':
            # [N, C, H, W]
            pool_space = x

            # [N, C, 1, 1]
            context = self.spatial_pool(x, pool_space)

            if self.channel_mul_conv is not None:
                # [N, C, 1, 1]
                channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
                out = x * channel_mul_term
            else:
                out = x
            if self.channel_add_conv is not None:
                # [N, C, 1, 1]
                channel_add_term = self.channel_add_conv(context)
                if self.out_op is not None:
                    channel_add_term = self.out_op(channel_add_term)
                if self.drop_out is not None:
                    channel_add_term = self.drop_out(channel_add_term)
                out = out + channel_add_term
        else:
            # saliency style
            if self.channel_mul_conv is not None:
                mul_pool_space = self.channel_mul_conv(x)
                mul_context = self.spatial_pool(x, mul_pool_space)
                # [N, C, 1, 1]
                channel_mul_term = torch.sigmoid(mul_context)
                out = x * channel_mul_term
            else:
                out = x
            if self.channel_add_conv is not None:
                # [N, C, 1, 1]
                add_pool_space = self.channel_add_conv(x)
                add_context = self.spatial_pool(x, add_pool_space)
                channel_add_term = add_context
                if self.out_op is not None:
                    channel_add_term = self.out_op(channel_add_term)
                if self.drop_out is not None:
                    channel_add_term = self.drop_out(channel_add_term)
                out = out + channel_add_term

        if self.spatial_add_conv is not None:
            out = out + self.spatial_add_conv(x)

        if self.spatial_mul_conv is not None:
            out = out * torch.sigmoid(self.spatial_mul_conv(x))

        return out
