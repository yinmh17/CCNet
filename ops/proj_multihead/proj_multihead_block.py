import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ProjMultiheadBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 one_fc=False,
                 mask_num=8,
                 pre_group=1,
                 post_group=1,
                 norm='ln',
                 height=97,
                 width=97):
        super(ProjMultiheadBlock, self).__init__()
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
        self.mask_num = mask_num
        self.pre_group = pre_group
        self.post_group = post_group
        self.height = height
        self.width = height
        if pooling_type == 'att':
            self.conv_mask = nn.Conv1d(self.height*self.width, self.mask_num, kernel_size=1, groups=self.pre_group)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            if one_fc:
                if norm == 'ln':
                    self.channel_add_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.mask_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),
                        nn.LayerNorm([self.inplanes, 1, 1]))
                else:
                    self.channel_add_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.mask_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),)
            else:
                self.channel_add_conv = nn.Sequential(
                    nn.Conv2d(int(self.inplanes*self.mask_num/self.pre_group), self.planes, kernel_size=1, groups=self.post_group),
                    nn.LayerNorm([self.planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

                    
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            if one_fc:
                if norm == 'ln':
                    self.channel_mul_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.mask_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),
                        nn.LayerNorm([self.inplanes, 1, 1]))
                else:
                    self.channel_mul_conv=nn.Sequential(
                        nn.Conv2d(int(self.inplanes*self.mask_num/self.pre_group), self.inplanes, kernel_size=1, groups=self.post_group),)
            else:
                self.channel_mul_conv = nn.Sequential(
                    nn.Conv2d(int(self.inplanes*self.mask_num/self.pre_group), self.planes, kernel_size=1, groups=self.post_group),
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
            #[B, HW, C]
            input_x=x.view(batch,channel,height*width).permute(0,2,1)
            #[B, 1, M, C]
            x_proj_vec=self.conv_mask(input_x).unsqueeze(1)

            #[B, HW, C, 1]
            input_x=input_x.permute(0,2,1).view(batch, height*width, channel, 1)
            #[B, HW, M]
            context_mask=torch.matmul(x_proj_vec, input_x).view(batch, height*width, self.mask_num)
            #[B, M, HW, 1]
            context_mask=context_mask.permute(0,2,1).unsqueeze(-1)
            context_mask=self.softmax(context_mask)

            #[B, preG, C/preG, HW]
            input_x=input_x.view(batch, self.pre_group, int(channel/self.pre_group), height*width)
            #[B, M, C/preG, 1]
            context=torch.matmul(input_x, context_mask)
            #[B, C*M/preG, 1, 1]
            context=context.view(batch, int(channel*self.mask_num/self.pre_group), 1, 1)
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
