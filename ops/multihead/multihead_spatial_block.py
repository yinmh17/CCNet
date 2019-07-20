import torch
import torch.nn.functional as F
from libs import InPlaceABN, InPlaceABNSync
from torch import nn
from torch.nn import init
import math
from mmcv.cnn import constant_init, kaiming_init

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)
        
class MultiheadSpatialBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 fusion_types=('channel_add', ),
                 head_num=4,
                 one_fc=False, Height=96, Width=96):
        super(MultiheadSpatialBlock, self).__init__()
        #assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        #self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.head_num = head_num
        
        self.conv_mask = nn.Conv2d(Height*Width, self.head_num, kernel_size=1, groups=self.head_num)
        self.softmax = nn.Softmax(dim=2)
        
        if one_fc:
            self.input_conv=nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),)
        else:
            self.input_conv=nn.Sequential(
                    nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                    nn.BatchNorm2d(self.planes),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.input_conv)

    def mask_pool(self, x):
        batch, channel, height, width = x.size()
        input_x=self.input_conv(x)
        #[B, HW, C, 1]
        input_x=input_x.permute(0,2,3,1).view(batch,height*width,channel,1)
 
        #[B, M2, C, 1]
        context_mask=self.conv_mask(input_x)
        #[B, M2, C, 1]
        context_mask=self.softmax(context_mask)

        #[B, M2, HW/M2, C]
        input_x=input_x.view(batch,self.head_num,int(height*width/self.head_num),channel)


        #[B, M2, HW/M2, 1]
        context=torch.matmul(input_x, context_mask)
        #[B, 1, H, M]
        context=context.view(batch,1,height*width,1).view(batch,1,height,width)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.mask_pool(x)

        out = x
        if 'channel_mul' in self.fusion_types:
            # [B, 1, H, W]
            channel_mul_term = torch.sigmoid(context)
            out = out * channel_mul_term
        if 'channel_add' in self.fusion_types:
            # [B, 1, H, W]
            channel_add_term = context
            out = out + channel_add_term

        return out
