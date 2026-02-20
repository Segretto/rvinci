import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_norm(norm, ch, groups_gn=8):
    return nn.GroupNorm(min(groups_gn, ch), ch) if norm == 'gn' else nn.BatchNorm2d(ch)

class DSConv(nn.Module):
    """Depthwise 3x3 -> Norm -> GELU -> Pointwise 1x1 -> Norm -> GELU."""
    def __init__(self, in_ch, out_ch, norm='gn', groups_gn=8):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = make_norm(norm, in_ch, groups_gn)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = make_norm(norm, out_ch, groups_gn)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class DSDown2(nn.Module):
    """Depthwise stride-2 downsample + pointwise (cheap and crisp)."""
    def __init__(self, ch, norm='gn', groups_gn=8):
        super().__init__()
        self.dw  = nn.Conv2d(ch, ch, 3, stride=2, padding=1, groups=ch, bias=False)
        self.bn1 = make_norm(norm, ch, groups_gn)
        self.pw  = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn2 = make_norm(norm, ch, groups_gn)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class FeatureFusionBlockLite(nn.Module):
    """Project + (optional) skip add + one DS residual to refine."""
    def __init__(self, ch=160, norm='gn', groups_gn=8, use_skip=True):
        super().__init__()
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn   = make_norm(norm, ch, groups_gn)
        self.act  = nn.GELU()
        self.res  = nn.Sequential(DSConv(ch, ch, norm=norm, groups_gn=groups_gn))
        self.use_skip = use_skip
    def forward(self, x, skip=None):
        x = self.act(self.bn(self.proj(x)))
        if self.use_skip and skip is not None:
            x = x + skip
        x = self.res(x)
        return x

class DepthHeadLite(nn.Module):
    """
    Small DPT-like decoder.
    """
    def __init__(self,
                 in_ch=384,
                 out_size=(640, 640),
                 proj0_ch=32,
                 proj1_ch=64,
                 proj2_ch=96,
                 proj3_ch=128,
                 common_ch=160,
                 dropout=0.2,
                 norm='gn',
                 groups_gn=8,
                 align_corners=False):
        super().__init__()
        self.out_size = out_size
        self.align_corners = align_corners

        self.drop = nn.Dropout2d(p=dropout)

        self.proj0 = nn.Conv2d(in_ch, proj0_ch, 1, bias=False)
        self.proj1 = nn.Conv2d(in_ch, proj1_ch, 1, bias=False)
        self.proj2 = nn.Conv2d(in_ch, proj2_ch, 1, bias=False)
        self.proj3 = nn.Conv2d(in_ch, proj3_ch, 1, bias=False)

        self.down3 = DSDown2(proj3_ch, norm=norm, groups_gn=groups_gn)

        self.to_c0 = DSConv(proj0_ch, common_ch, norm=norm, groups_gn=groups_gn)
        self.to_c1 = DSConv(proj1_ch, common_ch, norm=norm, groups_gn=groups_gn)
        self.to_c2 = DSConv(proj2_ch, common_ch, norm=norm, groups_gn=groups_gn)
        self.to_c3 = DSConv(proj3_ch, common_ch, norm=norm, groups_gn=groups_gn)

        self.fuse3 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=False)
        self.fuse2 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=True)
        self.fuse1 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=True)
        self.fuse0 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=True)

        self.h1 = nn.Conv2d(common_ch, common_ch//2, 3, padding=1)
        self.h2 = nn.Conv2d(common_ch//2, common_ch//4, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(common_ch//4, 1, 1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        self.h1_bn = make_norm(norm, common_ch//2, groups_gn)
        self.h2_bn = make_norm(norm, common_ch//4, groups_gn)

        self.min_depth = 1e-3

    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=self.align_corners)

    def forward(self, feat_1x):
        B, C, Hf, Wf = feat_1x.shape

        b0 = self.proj0(feat_1x)
        b1 = self.proj1(feat_1x)
        b2 = self.proj2(feat_1x)
        b3 = self.proj3(feat_1x)
        b3 = self.down3(b3)

        f0 = self.to_c0(self._upsample(b0, (Hf*4, Wf*4)))
        f1 = self.to_c1(self._upsample(b1, (Hf*2, Wf*2)))
        f2 = self.to_c2(b2)
        f3 = self.to_c3(b3)

        x = self.fuse3(f3, None)
        x = self._upsample(x, f2.shape[-2:]); x = self.fuse2(x, f2)
        x = self._upsample(x, f1.shape[-2:]); x = self.fuse1(x, f1)
        x = self.drop(x)
        x = self._upsample(x, f0.shape[-2:]); x = self.fuse0(x, f0)

        x = self.relu(self.h1_bn(self.h1(x)))
        x = self.drop(x)
        x = self._upsample(x, (x.shape[-2]*2, x.shape[-1]*2))
        x = self.relu(self.h2_bn(self.h2(x)))
        x = self.out(x)

        x = self.softplus(x) + self.min_depth
        x = self._upsample(x, self.out_size)
        return x
