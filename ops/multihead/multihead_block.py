import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class MultiheadBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 one_fc=False,
                 head_num=8,
                 pre_group=1,
                 post_group=8,
                 norm='ln'):
        super(MultiheadBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert norm in [None, 'bn', 'ln']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.head_num = head_num
        self.pre_group = pre_group
        self.post_group = post_group
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, self.head_num, kernel_size=1, groups=self.pre_group)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            if one_fc:
                if norm == 'ln':
                    self.channel_add_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.head_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),
                        nn.LayerNorm([self.inplanes, 1, 1]))
                else:
                    self.channel_add_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.head_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),)
            else:
                self.channel_add_conv = nn.Sequential(
                    nn.Conv2d(int(self.inplanes*self.head_num/self.pre_group), self.planes, kernel_size=1, groups=self.post_group),
                    nn.LayerNorm([self.planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

                    
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            if one_fc:
                if norm == 'ln':
                    self.channel_mul_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.head_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),
                        nn.LayerNorm([self.inplanes, 1, 1]))
                else:
                    self.channel_mul_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.head_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),)
            else:
                self.channel_mul_conv = nn.Sequential(
                    nn.Conv2d(int(self.inplanes*self.head_num/self.pre_group), self.planes, kernel_size=1, groups=self.post_group),
                    nn.LayerNorm([self.planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        # M: head_num
        # preG: pre_group (preG == M or preG ==1)
        # postG: post_group
        if self.pooling_type == 'att':
            #[B, preG, C/preG, H, W]
            input_x=x.view(batch, self.pre_group, int(channel/self.pre_group), height, width)
            #[B, preG, C/preG, HW]
            input_x=input_x.view(batch, self.pre_group, int(channel/self.pre_group), height*width)

            #[B, M, H, W]
            context_mask=self.conv_mask(x)
            #[B, M, HW]
            context_mask=context_mask.view(batch, self.head_num, height*width)
            #[B, M, HW]
            context_mask = self.softmax(context_mask)
            #[B, M, HW, 1]
            context_mask=context_mask.unsqueeze(-1)


            #[B, M, C/preG, 1]
            context=torch.matmul(input_x, context_mask)
            #[B, C*M/preG, 1, 1]
            context=context.view(batch, int(channel*self.head_num/self.pre_group), 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
