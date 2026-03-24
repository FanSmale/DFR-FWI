# -*- coding: utf-8 -*-
"""
@Time : 2026/1/12 17:02

@Author : Zeng Zifei
"""

from innovation.Biformer import Biformer
from innovation.DP import *
from thop import profile

class Stem(nn.Module):
    def __init__(self, in_fea):
        super(Stem,self).__init__()
        self.conv1 = nn.Conv2d(in_fea, 16, kernel_size=(4, 3), stride=(4, 1), padding=(0, 1))
        self.norm1 = nn.BatchNorm2d(16)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 3), stride=(4, 1), padding=(0, 1))
        self.norm2 = nn.BatchNorm2d(32)
        self.stem= nn.Sequential(
            nn.ReflectionPad2d((0, 0, 60, 60)),
            self.conv1,
            self.norm1
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.act1(x)
        x = self.norm2(self.conv2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock2(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_is_bn = True, para_active_func=nn.LeakyReLU(0.2, inplace=True)):
        super(ConvBlock2,self).__init__()
        if para_is_bn:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       para_active_func)
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       para_active_func)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       para_active_func)
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       para_active_func)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]

        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(ChannelSpatialAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * torch.sigmoid(x_channel_att)
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


class AttentionSkipConnection(nn.Module):
    def __init__(self, enc_ch, dec_ch):
        super().__init__()
        total_ch = enc_ch + dec_ch
        self.attention = ChannelSpatialAttention(in_channels=total_ch, out_channels=total_ch)
        self.reduce = nn.Conv2d(total_ch, dec_ch, kernel_size=1)

    def forward(self, enc_feat, dec_feat):
        enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[2:], mode='nearest')

        fused = torch.cat([enc_feat, dec_feat], dim=1)
        fused = self.attention(fused)
        fused = self.reduce(fused)
        return fused

class AttentionSkipConnection_interpolate(nn.Module):
    def __init__(self, enc_ch, dec_ch, output_lim = None):
        super().__init__()
        total_ch = enc_ch + dec_ch
        self.output_lim = output_lim

        self.attention = ChannelSpatialAttention(in_channels=total_ch, out_channels=total_ch)
        self.reduce = nn.Conv2d(total_ch, dec_ch, kernel_size=1)

    def forward(self, enc_feat, dec_feat):
        if enc_feat.shape[2:] != dec_feat.shape[2:]:
            dec_feat = F.interpolate(dec_feat,size=self.output_lim, mode='bilinear', align_corners=False)
        fused = torch.cat([enc_feat, dec_feat], dim=1)
        fused = self.attention(fused)
        fused = self.reduce(fused)
        return fused

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv = True, activ_fuc=nn.ReLU(inplace=True)):
        super(unetUp, self).__init__()
        self.output_lim = output_lim
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       activ_fuc)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class DP_AttentionSkip_Biformer23(nn.Module):
    def __init__(self, dims=[5, 32, 64, 128, 256, 512]):
        super(DP_AttentionSkip_Biformer23, self).__init__()

        # ( , 32, 70, 70)
        self.stem = Stem(dims[0])

        self.DP1 = DoublePath_Large(dims[1], dims[2], dims[2])
        self.maxpool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DP2 = DoublePath_Deform(dims[2], dims[3], dims[3])
        self.maxpool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DP3 = DoublePath_Deform(dims[3], dims[4], dims[4])
        self.maxpool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.convblock5 = ConvBlock2(dims[4], dims[5])

        self.skip1 = AttentionSkipConnection_interpolate(enc_ch=dims[4], dec_ch=dims[4])
        self.skip2 = AttentionSkipConnection_interpolate(enc_ch=dims[3], dec_ch=dims[3], output_lim = [35,35])
        self.skip3 = AttentionSkipConnection_interpolate(enc_ch=dims[2], dec_ch=dims[2])

        self.up1 = DeconvBlock(dims[5], dims[4], kernel_size=4, stride=2, padding=1)
        self.deconv1 = ConvBlock(dims[4], dims[4])
        self.up2 = DeconvBlock(dims[4], dims[3], kernel_size=4, stride=2, padding=1)
        self.Biformer2 = Biformer(
            dim=dims[3],
            n_win=5,
            topk=4,
            side_dwconv=3,
            mlp_dwconv=True,
            auto_pad=False
        )
        self.up3 = DeconvBlock(dims[3], dims[2], kernel_size=4, stride=2, padding=1)
        self.Biformer3 = Biformer(
            dim=dims[2],
            n_win=7,
            topk=4,
            side_dwconv=3,
            mlp_dwconv=True,
            auto_pad=False
        )


        self.deconv4 = ConvBlock2(dims[2], dims[1])
        self.deconv5 = ConvBlock_Tanh(dims[1], 1)

    def forward(self, x):
        x1 = self.stem(x)  # (None, 32, 70, 70)
        x2 = self.DP1(x1)  # {Tensor:(None,64,70,70)}
        x3 = self.maxpool1(x2)  # {Tensor:(None,64,35,35)}
        x4 = self.DP2(x3)  # {Tensor:(None,128,35,35)}
        x5 = self.maxpool2(x4)  # {Tensor:(None,128,18,18)}
        x6 = self.DP3(x5)  # {Tensor:(None,256,18,18)}
        x7 = self.maxpool3(x6)  # {Tensor:(None,256,9,9)}

        x_bottle = self.convblock5(x7)  # (None, 512, 9, 9)

        x = self.up1(x_bottle)  # (None, 256, 18, 18)
        x = self.skip1(x6, x)
        x = self.deconv1(x)  # (None, 256, 18, 18)

        x = self.up2(x)  # (None, 128, 36, 36)
        x = self.skip2(x4, x)  # (None, 128, 35, 35)
        x = self.Biformer2(x)  # (None, 128, 35, 35)

        x = self.up3(x)  # (None, 64, 70, 70)
        x = self.skip3(x2, x)
        x = self.Biformer3(x)

        x = self.deconv4(x)  # (None, 32, 70, 70)
        x = self.deconv5(x)  # (None, 1, 70, 70)
        return x


if __name__=='__main__':
    model = DP_AttentionSkip_Biformer23()
    input = torch.zeros((1, 5, 1000, 70))
    out = model(input)
    print(out.shape)
