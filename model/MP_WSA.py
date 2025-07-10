from torch import nn
import torch
import math
from model.swish import Swish
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from model.base_function import init_net
from einops import rearrange as rearrange
import numbers

def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=48)
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)


class SimpleMambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # 投影成状态空间的输入
        self.input_proj = nn.Linear(dim, hidden_dim)

        # 状态空间核模拟
        self.state_kernel = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim  # depthwise
        )

        # 动态门控，控制信息流
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Sigmoid()
        )

        # 输出映射
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """
        x: (B, seq_len, dim)
        """
        residual = x
        x = self.norm(x)

        # 1. 输入投影
        u = self.input_proj(x)  # (B, seq_len, hidden_dim)
        gate = self.gate(x)  # (B, seq_len, hidden_dim)

        # 2. 状态卷积核（模拟状态演化）
        u = u.transpose(1, 2)  # (B, hidden_dim, seq_len)
        u = self.state_kernel(u)  # depthwise 卷积
        u = u.transpose(1, 2)  # (B, seq_len, hidden_dim)

        # 3. 门控控制
        y = u * gate  # selective update

        # 4. 映射回原维度 + 残差连接
        out = self.output_proj(y) + residual
        return out
# 图像版 Mamba Block（适配 BCHW 格式）
class VisionMambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = SimpleMambaBlock(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.mamba(self.norm(x)) + x
        x = self.norm2(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

# 主模块：带 Mamba 增强的 PreModule
class PreModule(nn.Module):
    def __init__(self, in_channels=3, ngf=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )

        self.down64 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.GELU(),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.GELU()
        )

        self.feature64 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.GELU()
        )

        # 加入 Mamba 增强模块
        self.mamba_block = VisionMambaBlock(dim=ngf * 4, hidden_dim=ngf * 4)

        self.up128 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.GELU()
        )

        self.up256 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )

    def forward(self, x):
        feature = self.encoder(x)
        feature64 = self.down64(feature)
        prior64 = self.feature64(feature64)
        prior64 = self.mamba_block(prior64)
        prior128 = self.up128(prior64)
        prior256 = self.up256(prior128)
        return prior64, prior128, prior256

class Generator(nn.Module):
    def __init__(self, maps_r_module=None, ngf=48, num_block=[1, 2, 3, 4], num_head=[1, 2, 4, 8], factor=2.66):
        super().__init__()
        self.maps_r = PreModule()

        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )

        self.trane256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0], expansion_factor=factor) for _ in range(num_block[0])])
        self.down128 = Downsample(num_ch=ngf)
        self.trane128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 2, head=num_head[1], expansion_factor=factor) for _ in
              range(num_block[1])])
        self.down64 = Downsample(num_ch=ngf * 2)
        self.trane64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 4, head=num_head[2], expansion_factor=factor) for _ in
              range(num_block[2])])
        self.down32 = Downsample(num_ch=ngf * 4)
        self.trane32 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=num_head[3], expansion_factor=factor) for _ in
              range(num_block[3])])

        self.up64 = Upsample(ngf * 8)
        self.fuse64 = nn.Conv2d(in_channels=ngf * 4 * 3, out_channels=ngf * 4, kernel_size=1, stride=1, bias=False)
        self.trand64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 4, head=num_head[2], expansion_factor=factor) for _ in
              range(num_block[2])])

        self.up128 = Upsample(ngf * 4)

        # 修改 fuse128 的输入通道数（去掉 prior128，原来是 ngf*2*3）
        self.fuse128 = nn.Conv2d(in_channels=ngf * 2 * 2, out_channels=ngf * 2, kernel_size=1, stride=1, bias=False)
        self.trand128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 2, head=num_head[1], expansion_factor=factor) for _ in
              range(num_block[1])])

        self.up256 = Upsample(ngf * 2)

        # 修改 fuse256 的输入通道数（去掉 prior256，原来是 ngf * 3）
        self.fuse256 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, stride=1)
        self.trand256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0], expansion_factor=factor) for _ in range(num_block[0])])

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x, mask=None):
        prior64, prior128, prior256 = self.maps_r(x)

        noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 128.))
        x = x + noise
        feature = torch.cat([x, mask], dim=1)

        feature256 = self.start(feature)
        feature256 = self.trane256(feature256)

        feature128 = self.down128(feature256)
        feature128 = self.trane128(feature128)

        feature64 = self.down64(feature128)
        feature64 = self.trane64(feature64)

        feature32 = self.down32(feature64)
        feature32 = self.trane32(feature32)

        out64 = self.up64(feature32)
        out64 = self.fuse64(torch.cat([feature64, out64, prior64], dim=1))  # 保留 prior64
        out64 = self.trand64(out64)

        out128 = self.up128(out64)




        out128 = self.fuse128(torch.cat([feature128, out128], dim=1))  # 不再使用 prior128
        out128 = self.trand128(out128)

        out256 = self.up256(out128)



        out256 = self.fuse256(torch.cat([feature256, out256], dim=1))  # 不再使用 prior256
        out256 = self.trand256(out256)

        return torch.tanh(self.out(out256))


# class Generator(nn.Module):
#     def __init__(self, ngf=48, num_block=[1,2,3,4], num_head=[1,2,4,8], factor=2.66):
#         super().__init__()
#         self.start = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=4, out_channels=ngf, kernel_size=7, padding=0),
#             nn.InstanceNorm2d(ngf),
#             nn.GELU()
#         )
#         self.trane256 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
#         )
#         self.down128 = Downsample(num_ch=ngf) # B *2ngf * 128, 128
#         self.trane128 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf*2, head=num_head[1],expansion_factor=factor) for i in range(num_block[1])]
#         )
#         self.down64 = Downsample(num_ch=ngf*2) # B *4ngf * 64, 64
#         self.trane64 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf*4, head=num_head[2],expansion_factor=factor) for i in range(num_block[2])]
#         )
#         self.down32 = Downsample(num_ch=ngf*4)  # B *8ngf * 32, 32
#         self.trane32 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf*8, head=num_head[3],expansion_factor=factor) for i in range(num_block[3])]
#         )
#
#         self.up64 = Upsample(ngf*8)  # B *4ngf * 64, 64
#         self.fuse64 = nn.Conv2d(in_channels=ngf*4*2, out_channels=ngf*4, kernel_size=1, stride=1, bias=False)
#         self.trand64 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf*4, head=num_head[2],expansion_factor=factor) for i in range(num_block[2])]
#         )
#
#         self.up128 = Upsample(ngf*4) # B *2ngf * 128, 128
#         self.fuse128 = nn.Conv2d(in_channels=4*ngf, out_channels=2*ngf, kernel_size=1, stride=1, bias=False)
#         self.trand128 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf*2, head=num_head[1],expansion_factor=factor) for i in range(num_block[1])]
#         )
#
#         self.up256 = Upsample(ngf*2) # B *ngf * 256, 256
#         self.fuse256 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, stride=1)
#         self.trand256 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
#         )
#
#         self.trand2562 = nn.Sequential(
#             *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
#         )
#
#         self.out = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
#         )
#
#     def forward(self, x, mask=None):
#         noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 128.))
#         x = x + noise
#         feature = torch.cat([x, mask], dim=1)
#         feature256 = self.start(feature)
#         #m = F.interpolate(mask, size=feature.size()[-2:], mode='nearest')
#         feature256 = self.trane256(feature256)
#         feature128 = self.down128(feature256)
#         feature128 = self.trane128(feature128)
#         feature64 = self.down64(feature128)
#         feature64 = self.trane64(feature64)
#         feature32 = self.down32(feature64)
#         feature32 = self.trane32(feature32)
#
#         out64 = self.up64(feature32)
#         out64 = self.fuse64(torch.cat([feature64, out64], dim=1))
#         out64 = self.trand64(out64)
#         #out128 = torch.nn.functional.interpolate(out64, scale_factor=2, mode='nearest')
#         out128 = self.up128(out64)
#         out128 = self.fuse128(torch.cat([feature128, out128], dim=1))
#         out128 = self.trand128(out128)
#
#         out256 = self.up256(out128)
#         out256 = self.fuse256(torch.cat([feature256, out256], dim=1))
#         out256 = self.trand256(out256)
#         #out256 = self.trand2562(out256)
#         out = torch.tanh(self.out(out256))
#         return out



class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


# (H * W) * C -> (H/2 * C/2) * (4C) -> (H/4 * W/4) * 16C -> (H/8 * W/8) * 64C
class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=256, head=4, expansion_factor=2.66):
        super().__init__()

        self.attn = WSA(dim=in_ch, num_heads=head,bias=False,LayerNorm_type='WithBias')
        self.feed_forward = GDT_FF(dim=in_ch, expansion_factor=expansion_factor,LayerNorm_type='WithBias')

    def forward(self, x):
        x = self.attn(x) + x
        x = self.feed_forward(x) + x
        return x

class Convblock(nn.Module):
    def __init__(self, in_ch=256, out_ch=None, kernel_size=3, padding=1, stride=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.norm = nn.InstanceNorm2d(num_features=out_ch, track_running_stats=False)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU()
        )
        self.linear = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.projection(x)
        x1 = self.conv(x)
        x2 = self.gate(x)
        out = x1 * x2
        out = self.norm(out)
        out = self.linear(out)
        out = out + residual
        return out

class Downsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch*2, track_running_stats=False),
            nn.GELU()
        )

        #self.body = nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, num_ch=32):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch//2, track_running_stats=False),
            nn.GELU()
        )

        #self.body = nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.body(x)

##########################################################################

class WCM(nn.Module):
    def __init__(self, channels):
        super(WCM, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 5, dilation=5)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 7, dilation=7)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 9, dilation=9)

        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)

    def forward(self, x):

        aa =  DWTForward(J=1, mode='zero', wave='db3').cuda()
        yl, yh = aa(x)

        yh_out = yh[0]
        ylh = yh_out[:,:,0,:,:]
        yhl = yh_out[:,:,1,:,:]
        yhh = yh_out[:,:,2,:,:]

        conv_rec1 = self.conv5(yl)
        conv_rec5 = self.conv5(ylh)
        conv_rec7 = self.conv7(yhl)
        conv_rec9 = self.conv9(yhh)

        cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9),dim=2)
        rec_yh = []
        rec_yh.append(cat_all)


        ifm = DWTInverse(wave='db3', mode='zero').cuda()
        Y = ifm((conv_rec1, rec_yh))

        return Y   # #

class WSA(nn.Module):
    def __init__(self, dim, num_heads, bias,LayerNorm_type):
        super(WSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.query = WCM(dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_1 = self.norm1(x)
        g = self.gate(x_1)

        qkv = self.qkv_dwconv(self.qkv(x_1))
        q, k, v = qkv.chunk(3, dim=1)
        q = self.query(x)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #attn = attn.softmax(dim=-1)
        attn = F.relu(attn)
        #attn = torch.square(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out * g
        out = self.project_out(out)
        #out = x+ out
        return out

class GDT_FF(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66,LayerNorm_type='WithBias'):
        super().__init__()

        num_ch = int(dim * expansion_factor)
        #self.norm = LayerNorm(dim, LayerNorm_type)
        self.norm = DynamicTanh(dim, LayerNorm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=num_ch*2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=num_ch*2, out_channels=num_ch*2, kernel_size=3, stride=1, padding=1, groups=num_ch*2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=num_ch, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
        #out = out + x
        return out


# class DynamicGatedFeedForward(nn.Module):
#     def __init__(self, dim=64, expansion_factor=4.0, LayerNorm_type='WithBias'):
#         super().__init__()
#
#         self.expansion_factor = expansion_factor
#         num_ch = int(dim * expansion_factor)
#
#         # 门控分支：生成控制信号 C_hat 和 C_prime
#         self.gate_fc1 = nn.Linear(dim, num_ch)
#         self.gate_fc2 = nn.Linear(dim, num_ch)
#
#         # 归一化层
#         self.norm = LayerNorm(dim, LayerNorm_type)
#
#         # 卷积层
#         self.conv1 = nn.Conv2d(dim, num_ch, kernel_size=1, bias=False)
#         self.dconv3 = nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=1, groups=num_ch, bias=False)
#
#         # 激活函数和最终线性层
#         self.gelu = nn.GELU()
#         self.conv2 = nn.Conv2d(num_ch, dim, kernel_size=1, bias=False)
#
#         # 残差连接
#         self.residual = nn.Identity()
#
#     def forward(self, x):
#         # 提取输入特征维度
#         B, C, H, W = x.shape
#
#         # 动态门控机制
#         Z = x.mean(dim=(2, 3))  # 全局特征，shape: (B, C)
#         C_hat = self.gate_fc1(Z).unsqueeze(-1).unsqueeze(-1)  # shape: (B, num_ch, 1, 1)
#         C_prime = self.gate_fc2(Z).unsqueeze(-1).unsqueeze(-1)  # shape: (B, num_ch, 1, 1)
#
#         # 归一化和特征提取
#         F_norm = self.norm(x)
#         F_conv = self.conv1(F_norm)
#         F_dconv = self.dconv3(F_conv)
#
#         # 门控机制融合
#         F_gate = F_conv * torch.sigmoid(C_hat) + F_dconv * torch.sigmoid(C_prime)
#
#         # 激活和线性变换
#         F_out = self.gelu(F_gate)
#         F_out = self.conv2(F_out)
#
#         # 添加残差连接
#         return self.residual(x) + F_out


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class DynamicTanh(nn.Module):
    def __init__(self, dim, LayerNorm_type, alpha_init_value=0.5):
        super(DynamicTanh, self).__init__()
        self.dim = dim
        self.alpha_init_value = alpha_init_value
        self.LayerNorm_type = LayerNorm_type

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if LayerNorm_type == 'BiasFree':
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = None  # No bias in BiasFree mode
        else:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)  # Convert to 3D
        x = torch.tanh(self.alpha * x)

        if self.bias is not None:  # WithBias mode
            x = x * self.weight + self.bias
        else:  # BiasFree mode
            x = x * self.weight

        return to_4d(x, h, w)  # Convert back to 4D

class GAttn(nn.Module):
    def __init__(self, in_ch=256):
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
            #nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
            #nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.norm = nn.InstanceNorm2d(num_features=in_ch)

    def forward(self, x):
        """
        x: b * c * h * w
        """
        x = self.norm(x)
        B, C, H, W = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        g = self.gate(x)

        q = q.view(B, C, H * W).contiguous().permute(0,2,1).contiguous()  # b * N * C
        k = k.view(B, C, H * W).contiguous()                              # b * C * N
        v = v.view(B, C, H * W).contiguous().permute(0,2,1).contiguous()  # B * N * C
        kv = torch.einsum('bcn, bnd -> bcd', k, v)
        z = torch.einsum('bnc,bc -> bn', q, k.sum(dim=-1)) / math.sqrt(C)
        z = 1.0 / (z + H*W)
        out = torch.einsum('bnc, bcd-> bnd', q, kv)
        out = out / math.sqrt(C)
        out = out + v
        out = torch.einsum('bnc, bn -> bnc', out, z)
        out = out.permute(0,2,1).contiguous().view(B,C,H,W)
        out = out * g
        out = self.output_linear(out)
        return out


class mGAttn(nn.Module):
    def __init__(self, in_ch=256, num_head=4):
        super().__init__()
        self.head = num_head
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
            #nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
            #nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.norm = nn.InstanceNorm2d(num_features=in_ch)

    def forward(self, x):
        """
        x: b * c * h * w
        """
        x = self.norm(x)
        Ba, Ca, He, We = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        g = self.gate(x)
        num_per_head = Ca // self.head

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head)   # B * head * c * N
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head)   # B * head * c * N
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)   # B * head * c * N
        kv = torch.matmul(k, v.transpose(-2, -1))
        # kv = torch.einsum('bhcn, bhdn -> bhcd', k, v)
        z = torch.einsum('bhcn,bhc -> bhn', q, k.sum(dim=-1)) / math.sqrt(num_per_head)
        z = 1.0 / (z + He*We)   # b h n
        out = torch.einsum('bhcn, bhcd-> bhdn', q, kv)
        out = out / math.sqrt(num_per_head)    # b h c n
        out = out + v
        out = out * z.unsqueeze(2)
        out = rearrange(out,'b head c (h w) -> b (head c) h w', h=He)
        out = out * g
        out = self.output_linear(out)
        return out



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
