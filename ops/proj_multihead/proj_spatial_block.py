import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ProjSpatialBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 one_fc=False,
                 mask_num=8,
                 proj_num=128,
                 pre_group=1,
                 post_group=1,
                 norm='ln',
                 share_proj=False,
                 softmax_loc=[1,2],
                 scale=16
                 ):
        super(ProjSpatialBlock, self).__init__()
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
        self.proj_num = proj_num
        self.pre_group = pre_group
        self.post_group = post_group
        self.share_proj = share_proj
        self.softmax_loc = softmax_loc
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, self.mask_num, kernel_size=1, groups=self.pre_group)
            self.conv_mask_proj = nn.Conv2d(inplanes, self.proj_num, kernel_size=1, groups=self.pre_group)
            if self.share_proj == False:
                self.conv_value_proj = nn.Conv2d(inplanes, self.proj_num, kernel_size=1, groups=self.pre_group)
            if 1 in self.softmax_loc:
                self.softmax1 = nn.Softmax(dim=2)
            if 2 in self.softmax_loc:
                self.softmax2 = nn.Softmax(dim=2)
            if 3 in self.softmax_loc:
                self.softmax3 = nn.Softmax(dim=2)
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
            kaiming_init(self.conv_mask_proj, mode='fan_in')
            if self.share_proj == False:
                kaiming_init(self.conv_value_proj, mode='fan_in')
                self.conv_value_proj.inited = True
            self.conv_mask.inited = True
            self.conv_mask_proj.inited = True
            

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        # M: head_num
        # P: proj_num
        # preG: pre_group (preG == M or preG ==1)
        # postG: post_group
        if self.pooling_type == 'att':
            #[B, P, HW]
            x_mask_proj=self.conv_mask_proj(x).view(batch, self.proj_num, height*width)
            #[B, P, HW]
            if self.share_proj:
                x_value_proj=x_mask_proj
            else:
                x_value_proj=self.conv_value_proj(x).view(batch, self.proj_num, height*width)
            
            #[B, M, HW]
            context_mask=self.conv_mask(x).view(batch, self.mask_num, height*width)
            
            if 1 in self.softmax_loc:
                #[B, M, HW]
                context_mask=self.softmax1(context_mask)
                
            #[B, M, HW] X [B, HW, P] --> [B, M, P]
            context_mask=torch.matmul(context_mask, x_mask_proj.permute(0,2,1))
            if 1 not in self.softmax_loc:
                #[B, M, P]
                context_mask=context_mask*(1. / height/width)
            
            if 2 in self.softmax_loc:
                #[B, M, P]
                context_mask=self.softmax2(context_mask)
            
            #[B, M, HW]
            context=x.view(batch, channel, height*width)
            #[B, C, HW] X [B, HW, P] --> [B, C, P]
            context=torch.matmul(context, x_value_proj.permute(0,2,1))
            
            #[B, M, P, 1]
            context_mask=context_mask.view(batch, self.mask_num, self.proj_num, 1)
            #[B, preG, C/preG, P]
            context=context.view(batch, self.pre_group, channel//self.pre_group, self.proj_num)
            
            #[B, preG, C/preG, P] X [B, M, P, 1] --> [B, M, C/preG, 1]
            context=torch.matmul(context, context_mask)
            #[B, M*C/preG, 1, 1]
            context=context.view(batch, self.mask_num*channel//self.pre_group, 1, 1)
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
