import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchsummary import summary


__all__ = ['ERes2Net', 'ERes2NetV2']

class TemporalStatsPool(nn.Module):
    """TSTP
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self):
        super(TemporalStatsPool, self).__init__()

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)

        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

class TemporalAveragePooling(nn.Module):
    def __init__(self):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = x.mean(dim=-1)
        # To be compatable with 2D input
        x = x.flatten(start_dim=1)
        return x


class TemporalStatisticsPooling(nn.Module):
    def __init__(self):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, dim=2)
        var = torch.var(x, dim=2)
        x = torch.cat((mean, var), dim=1)
        return x


class SelfAttentivePooling(nn.Module):
    """SAP"""

    def __init__(self, in_dim, bottleneck_dim=128):
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # attention dim = 128
        super(SelfAttentivePooling, self).__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        return mean


class AttentiveStatsPool(nn.Module):
    """ASP"""

    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)

        return xo


class BasicBlockERes2Net(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2Net(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 mul_channel=1,
                 expansion=2,
                 base_width=32,
                 scale=2,
                 embd_dim=192,
                 pooling_type='TSTP',
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = m_channels
        self.expansion = expansion
        self.feat_dim = input_size
        self.embd_dim = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1,
                                       base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2,
                                       base_width=base_width, scale=scale)

        # Downsampling module for each layer
        self.layer1_downsample = nn.Conv2d(m_channels * 2 * mul_channel, m_channels * 4 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4 * mul_channel, m_channels * 8 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8 * mul_channel, m_channels * 16 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.fuse_mode12 = AFF(channels=m_channels * 4 * mul_channel)
        self.fuse_mode123 = AFF(channels=m_channels * 8 * mul_channel)
        self.fuse_mode1234 = AFF(channels=m_channels * 16 * mul_channel)

        self.n_stats = 1 if pooling_type == 'TAP' else 2
        if pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
        elif pooling_type == "TSTP":
            self.pooling = TemporalStatsPool()
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()
        # 分类层
        self.fc = nn.Linear(embd_dim, num_class)

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.expansion, self.in_planes, planes, stride, base_width, scale))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        #x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pooling(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)

            out = self.fc(embed_b)
        else:
            out = self.fc(embed_a)
        out = nn.functional.softmax(out)
        return out


class BasicBlockERes2NetV2(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=26, scale=2):
        super(BasicBlockERes2NetV2, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2NetV2_AFF(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=26, scale=2):
        super(BasicBlockERes2NetV2_AFF, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width, r=4))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2NetV2(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 block=BasicBlockERes2NetV2,
                 block_fuse=BasicBlockERes2NetV2_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 expansion=2,
                 base_width=26,
                 scale=2,
                 embd_dim=192,
                 pooling_type='TSTP',
                 two_emb_layer=False):
        super(ERes2NetV2, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = m_channels
        self.expansion = expansion
        self.embd_dim = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1,
                                       base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2,
                                       base_width=base_width, scale=scale)

        # Downsampling module
        self.layer3_ds = nn.Conv2d(m_channels * 8, m_channels * 16, kernel_size=3, padding=1, stride=2, bias=False)

        # Bottom-up fusion module
        self.fuse34 = AFF(channels=m_channels * 16, r=4)

        self.n_stats = 1 if pooling_type == 'TAP' else 2
        if pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
        elif pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(in_dim=self.stats_dim * self.expansion)
        elif pooling_type == "TSTP":
            self.pooling = TemporalStatsPool()
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()
        # 分类层
        self.fc = nn.Linear(embd_dim, num_class)

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.expansion, self.in_planes, planes, stride, base_width, scale))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        # x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        fuse_out34 = self.fuse34(out4, out3_ds)
        stats = self.pooling(fuse_out34)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            out = self.fc(embed_b)
        else:
            out = self.fc(embed_a)
        out = nn.functional.softmax(out)
        return out


if __name__ == "__main__":
    x = torch.randn([8,1,64,100])
    model = ERes2Net(num_class=10,input_size=64)
    x = model(x)
    print(x.shape)
    print(summary(model,(1,64,100),device="cpu"))

    flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
    # 输出 FLOPs 和参数数量，增加描述文字
    print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")


    model = ERes2NetV2(num_class=10,input_size=64)
    x = torch.randn([8, 1, 64, 100])
    x = model(x)
    print(x.shape)
    print(summary(model,(1,64,100),device="cpu"))

    flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
    # 输出 FLOPs 和参数数量，增加描述文字
    print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")