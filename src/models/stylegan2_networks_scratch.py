# src/models/stylegan2_networks_scratch.py
# Contains from-scratch implementations of StyleGAN2 network architectures.
# This version contains the final architectural fix for the NameError.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

# --- Custom Layers ---

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = (math.sqrt(2) / in_features) * lr_mul
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = math.sqrt(2 / (in_channels * kernel_size**2))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, up=False):
        super().__init__()
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.up = up
        self.padding = kernel_size // 2
        self.modulation = EqualizedLinear(style_dim, in_channels, bias=True)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    def forward(self, x, style):
        batch_size, in_channels, height, width = x.shape
        style = self.modulation(style).view(batch_size, 1, in_channels, 1, 1)
        weight = self.weight.unsqueeze(0) * (style + 1)
        if self.demodulate:
            demod_coeff = (weight.pow(2).sum(dim=[2,3,4]) + 1e-8).rsqrt()
            weight = weight * demod_coeff.view(batch_size, self.out_channels, 1, 1, 1)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            height, width = height * 2, width * 2
        x = x.reshape(1, -1, height, width)
        weight = weight.reshape(-1, in_channels, *self.weight.shape[2:])
        x = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        x = x.reshape(batch_size, self.out_channels, *x.shape[2:])
        x = x + self.bias.view(1, -1, 1, 1)
        return x

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return x + self.weight * noise

class PixelNorm(nn.Module):
    def forward(self, x):
        return x * (x.pow(2).mean(dim=1, keepdim=True) + 1e-8).rsqrt()

class ToRGBLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 1, style_dim, demodulate=True)
    def forward(self, x, style):
        return self.conv(x, style)

class MinibatchStdDevLayer(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
    def forward(self, x):
        N, C, H, W = x.shape
        group_size = min(N, self.group_size)
        y = x.view(group_size, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.square().mean(dim=0) + 1e-8).sqrt().mean(dim=[1,2,3])
        y = y.view(-1, 1, 1, 1).repeat(group_size, 1, H, W)
        return torch.cat([x, y], dim=1)

# --- Main Network Blocks (Order Corrected) ---

# CORRECTED: Moved the Block definitions before the main networks that use them.
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, img_channels):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, w_dim, up=True)
        self.noise1 = NoiseInjection()
        self.activation1 = nn.LeakyReLU(0.2)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, w_dim)
        self.noise2 = NoiseInjection()
        self.activation2 = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGBLayer(out_channels, img_channels, w_dim)
    def forward(self, x, w1, w2, rgb):
        x = self.conv1(x, w1)
        x = self.noise1(x)
        x = self.activation1(x)
        x = self.conv2(x, w2)
        x = self.noise2(x)
        x = self.activation2(x)
        new_rgb = self.to_rgb(x, w2)
        if rgb is not None:
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
            new_rgb = new_rgb + rgb
        return x, new_rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2)
        self.skip = EqualizedConv2d(in_channels, out_channels, 1)
    def forward(self, x):
        skip_x = self.skip(self.downsample(x))
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.downsample(x)
        return (x + skip_x) * (1 / math.sqrt(2))

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers, lr_mul=0.01):
        super().__init__()
        self.w_dim = w_dim
        layers = [PixelNorm()]
        for i in range(num_layers):
            layers.append(EqualizedLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        if z.ndim == 2:
            norm_inv = (z.pow(2).sum(dim=1, keepdim=True) + 1e-8).rsqrt()
            z = z * norm_inv
        return self.net(z)

class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, img_channels, channel_base=32768, channel_max=512):
        super().__init__()
        self.w_dim = w_dim
        self.img_channels = img_channels
        def get_ch(res): return min(int(channel_base / res), channel_max)
        self.input_const = nn.Parameter(torch.randn(1, get_ch(4), 4, 4))
        self.conv1 = ModulatedConv2d(get_ch(4), get_ch(4), 3, w_dim)
        self.noise_init = NoiseInjection()
        self.act_init = nn.LeakyReLU(0.2)
        self.to_rgb_init = ToRGBLayer(get_ch(4), img_channels, w_dim)
        self.blocks = nn.ModuleList()
        in_ch = get_ch(4)
        for res in [2**i for i in range(3, int(np.log2(img_resolution)) + 1)]:
            out_ch = get_ch(res)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim, img_channels))
            in_ch = out_ch
    def forward(self, ws):
        x = self.input_const.repeat(ws.shape[0], 1, 1, 1)
        x = self.conv1(x, ws[:, 0])
        x = self.noise_init(x)
        x = self.act_init(x)
        rgb = self.to_rgb_init(x, ws[:, 0])
        for i, block in enumerate(self.blocks):
            x, rgb = block(x, ws[:, i * 2 + 1], ws[:, i * 2 + 2], rgb)
        return rgb

class Discriminator(nn.Module):
    def __init__(self, img_resolution, img_channels, channel_base=32768, channel_max=512, mbstd_group_size=4):
        super().__init__()
        def get_ch(res): return min(int(channel_base / res), channel_max)
        self.from_rgb = EqualizedConv2d(img_channels, get_ch(img_resolution), 1)
        self.blocks = nn.ModuleList()
        in_ch = get_ch(img_resolution)
        for res_log2 in range(int(np.log2(img_resolution)), 2, -1):
            out_ch = get_ch(2**(res_log2-1))
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch
        self.mbstd = MinibatchStdDevLayer(group_size=mbstd_group_size)
        final_in_ch = in_ch + 1
        self.final_conv = EqualizedConv2d(final_in_ch, in_ch, 3, padding=1)
        self.final_act = nn.LeakyReLU(0.2)
        self.final_dense = EqualizedLinear(in_ch * 4 * 4, in_ch)
        self.output = EqualizedLinear(in_ch, 1)
    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        x = self.final_act(self.final_conv(x))
        x = x.view(x.shape[0], -1)
        x = self.final_act(self.final_dense(x))
        x = self.output(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, num_mapping_layers, mapping_lr_mul,
                 img_resolution, img_channels, channel_base, channel_max, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        num_blocks = int(np.log2(img_resolution)) - 2
        self.num_ws = num_blocks * 2 + 2
        self.mapping = MappingNetwork(z_dim, w_dim, num_mapping_layers, lr_mul=mapping_lr_mul)
        self.synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels, 
                                          channel_base=channel_base, 
                                          channel_max=channel_max)
    def forward(self, z, style_mixing_prob=0.9, truncation_psi=0.7, return_ws=False):
        w = self.mapping(z)
        if self.training and style_mixing_prob > 0 and random.random() < style_mixing_prob:
            z2 = torch.randn_like(z)
            w2 = self.mapping(z2)
            crossover_points = (torch.rand(z.shape[0], device=z.device) * self.num_ws).floor().to(torch.long)
            mask = torch.arange(self.num_ws, device=z.device)[None, :] < crossover_points[:, None]
            ws = torch.where(mask[:, :, None], w.unsqueeze(1), w2.unsqueeze(1))
        else:
            ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)
        if not self.training and truncation_psi < 1.0:
            pass # Truncation would be implemented here
        img = self.synthesis(ws)
        if return_ws:
            return img, ws
        return img
