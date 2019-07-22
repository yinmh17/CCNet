    
import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn
import numpy as np

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

class MultiheadRelationBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 one_fc=False,
                 head_num=8,
                 pre_group=1,
                 post_group=1,
                 norm='ln',
                 geo_feature_dim=64,
                 key_feature_dim=64):
        super(MultiheadRelationBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert norm in [None, 'bn', 'ln']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        
        #-------conv feature-----------------------------------------
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        
        #-------relation feature--------------------------------------
        self.head_num = head_num
        self.pre_group = pre_group
        self.post_group = post_group
        
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        #--------------------------------------------------------------
        
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, self.head_num, kernel_size=1, groups=self.pre_group)
            self.softmax = nn.Softmax(dim=2)
            self.LinearG = nn.Linear(geo_feature_dim, 1, bias=True)
            self.LinearK = nn.Linear(int(inplanes/pre_group), key_feature_dim, bias=True)
            self.LinearQ = nn.Linear(int(inplanes/pre_group), key_feature_dim, bias=True)
            self.LinearV = nn.Linear(int(inplanes/pre_group), key_feature_dim, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.conv_inplanes = self.head_num*self.dim_k
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_inplanes = inplanes
        if 'channel_add' in fusion_types:
            if one_fc:
                if norm == 'ln':
                    self.channel_add_conv=nn.Sequential(
                        nn.Conv2d(self.conv_inplanes, self.inplanes, kernel_size=1, groups=self.post_group),
                        nn.LayerNorm([self.inplanes, 1, 1]))
                else:
                    self.channel_add_conv=nn.Sequential(
                        nn.Conv2d(self.conv_inplanes, self.inplanes, kernel_size=1, groups=self.post_group),)
            else:
                self.channel_add_conv = nn.Sequential(
                    nn.Conv2d(self.conv_inplanes, self.planes, kernel_size=1, groups=self.post_group),
                    nn.LayerNorm([self.planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

                    
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            if one_fc:
                if norm == 'ln':
                    self.channel_mul_conv=nn.Sequential(
                        nn.Conv2d(self.conv_inplanes, self.inplanes, kernel_size=1, groups=self.post_group),
                        nn.LayerNorm([self.inplanes, 1, 1]))
                else:
                    self.channel_mul_conv=nn.Sequential(
                        nn.Conv2d(self.conv_inplanes, self.inplanes, kernel_size=1, groups=self.post_group),)
            else:
                self.channel_mul_conv = nn.Sequential(
                    nn.Conv2d(self.conv_inplanes, self.planes, kernel_size=1, groups=self.post_group),
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
    
    def Position_Embedding(self, context_mask, dim_g, wave_len=1000):
        batch, head_num, height, width = context_mask.size()
        #context_mask=context_mask.view(batch,head_num,height,width)
        meshH=torch.tensor(np.arange(height)).expand(height,width).unsqueeze(2)
        meshW=torch.tensor(np.arange(width)).expand(height,width).permute(1,0).unsqueeze(2)
        
        #[H, W, 2]
        mesh=torch.cat((meshH,meshW),2)
        mesh=mesh.view(1,1,height,width,2).float().cuda()
        context_mask=context_mask.view(batch,head_num,height,width,1)
        
        #[B, M, H, W, 2] --> [B, M, 2]
        mean_pos = (context_mask*mesh).sum(3).sum(2)
        #[B, M]
        cx=mean_pos[:,:,0]/height
        cy=mean_pos[:,:,1]/width
        
        #[B, M, M, 1]
        delta_x1=torch.clamp(cx.unsqueeze(2)-cx.unsqueeze(1), min=0).unsqueeze(-1)
        delta_x2=torch.clamp(cx.unsqueeze(1)-cx.unsqueeze(2), min=0).unsqueeze(-1)
        delta_y1=torch.clamp(cy.unsqueeze(2)-cy.unsqueeze(1), min=0).unsqueeze(-1)
        delta_y2=torch.clamp(cy.unsqueeze(1)-cy.unsqueeze(2), min=0).unsqueeze(-1)
        
        #[B, M, M, 4, 1]
        position_mat=torch.cat([delta_x1,delta_x2,delta_y1,delta_y2],dim=3)
        position_mat=position_mat.unsqueeze(-1)
        position_mat = 100. * position_mat

        feat_range = torch.arange(dim_g / 8)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))
        #[1, 1, 1, 1, dim_g/8]
        dim_mat = dim_mat.view(1, 1, 1, 1, -1).cuda()
        
        #[B, M, M, 4, dim_g/8]
        mul_mat = position_mat * dim_mat
        #[B, M, M, dim_g/2]
        mul_mat = mul_mat.view(batch,head_num,head_num,-1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        #[B, M, M, dim_g]
        embedding = torch.cat((sin_mat, cos_mat), -1)
        return embedding

    def relation_unit(self, x):
        batch, channel, height, width = x.size()
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
            #[B, M, C/preG]
            context=context.squeeze()
            #[B, M, H, W]
            context_mask = context_mask.view(batch, self.head_num, height, width)
            #[B, M, M, dim_g]
            position_embedding = self.Position_Embedding(context_mask, dim_g=self.dim_g)
            #[B, M*M, dim_g]
            position_embedding = position_embedding.view(batch, self.head_num*self.head_num, self.dim_g)
            #[B, M*M, 1]
            w_g = self.relu(self.LinearG(position_embedding))
            #[B, M, M]
            w_g = w_g.view(batch, self.head_num, self.head_num)

            #[B, M, dim_k]
            w_k = self.LinearK(context)
            #[B, M, 1, dim_k]
            w_k = w_k.view(batch, self.head_num, 1, self.dim_k)

            #[B, M, dim_k]
            w_q = self.LinearQ(context)
            #[B, 1, M, dim_k]
            w_q = w_q.view(batch, 1, self.head_num, self.dim_k)

            #[B, M, M]
            scaled_dot = torch.sum((w_k*w_q), -1)
            scaled_dot = scaled_dot / np.sqrt(self.dim_k)

            #[B, M, M]
            w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + scaled_dot
            w_mn = torch.nn.Softmax(dim=1)(w_mn)

            #[B, M, dim_k]
            w_v = self.LinearV(context)

            #[B, M, M, 1]
            w_mn = w_mn.view(batch, self.head_num, self.head_num, 1)
            #[B, M, 1, dim_k]
            w_v = w_v.view(batch, self.head_num, 1, self.dim_k)

            output = w_mn*w_v
            
            #[B, M*dim_k, 1, 1]
            output = torch.sum(output,1).view(batch, self.head_num*self.dim_k, 1, 1)
            
        else:
            # [N, C, 1, 1]
            output = self.avg_pool(x)

        return output

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.relation_unit(x)

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
