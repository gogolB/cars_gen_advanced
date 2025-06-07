# src/models/stylegan2_networks_scratch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --- Helper Functions & Building Blocks ---

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0, bias_init=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mul = lr_mul
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        if bias:
            self.bias = nn.Parameter(torch.full([out_features], float(bias_init)) / lr_mul)
        else:
            self.register_parameter('bias', None)
        self.weight_gain = lr_mul / np.sqrt(in_features)
        if self.bias is not None:
            self.bias_gain = lr_mul

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            b = b * self.bias_gain
        if x.ndim == 2:
            return F.linear(x, w, b)
        original_shape = x.shape
        x_reshaped = x.reshape(-1, self.in_features)
        out_reshaped = F.linear(x_reshaped, w, b)
        return out_reshaped.reshape(original_shape[:-1] + (self.out_features,))

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0, bias_init=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr_mul = lr_mul
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) / lr_mul)
        if bias:
            self.bias = nn.Parameter(torch.full([out_channels], float(bias_init)) / lr_mul)
        else:
            self.register_parameter('bias', None)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_gain = lr_mul / np.sqrt(fan_in)
        if self.bias is not None:
            self.bias_gain = lr_mul
            
    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            b = b * self.bias_gain
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        batch, _, height, width = x.shape
        if noise is None:
            noise = torch.randn(batch, 1, height, width, device=x.device, dtype=x.dtype)
        return x + self.weight * noise

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale_transform = EqualizedLinear(w_dim, channels, bias_init=1.0)
        self.style_shift_transform = EqualizedLinear(w_dim, channels, bias_init=0.0)

    def forward(self, x, w): # w is expected to be (N, w_dim)
        if w.ndim != 2 or w.shape[1] != self.style_scale_transform.in_features:
             raise ValueError(f"AdaIN expects w of shape (N, w_dim={self.style_scale_transform.in_features}), got {w.shape}")
        normalized_x = self.norm(x)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        return style_scale * normalized_x + style_shift

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, w_dim,
                 demodulate=True, up=False, down=False, resample_kernel=None, lr_mul=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.lr_mul = lr_mul
        self.resample_kernel = resample_kernel 
        if self.resample_kernel is None and (up or down):
            self.resample_kernel = [1,2,1] 
        self.padding = kernel_size // 2
        self.modulation = EqualizedLinear(w_dim, in_channels, bias_init=1.0, lr_mul=1.0) 
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) / lr_mul
        )
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_gain = lr_mul / np.sqrt(fan_in) 

    def forward(self, x, w): # w is expected to be (N, w_dim)
        if w.ndim != 2 or w.shape[1] != self.modulation.in_features:
             raise ValueError(f"ModulatedConv2d expects w of shape (N, w_dim={self.modulation.in_features}), got {w.shape}")
        batch_size, in_channels, height, width = x.shape
        style = self.modulation(w).view(batch_size, 1, in_channels, 1, 1) 
        weights = self.weight * self.weight_gain 
        weights = weights.unsqueeze(0) 
        modulated_weights = weights * style 
        if self.demodulate:
            demod_scale = (modulated_weights.square().sum(dim=[2,3,4], keepdim=True) + 1e-8).rsqrt()
            modulated_weights = modulated_weights * demod_scale
        x = x.reshape(1, batch_size * in_channels, height, width) 
        modulated_weights = modulated_weights.reshape(batch_size * self.out_channels, in_channels, self.kernel_size, self.kernel_size)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) 
        out = F.conv2d(x, modulated_weights, bias=None, stride=1, padding=self.padding, groups=batch_size)
        out = out.reshape(batch_size, self.out_channels, out.shape[2], out.shape[3])
        if self.down:
            out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        return out

class Blur(nn.Module):
    def __init__(self, kernel=[1,2,1], normalize=True, flip=False, stride=1):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = kernel[:, None] * kernel[None, :] 
        if normalize:
            kernel /= kernel.sum()
        if flip:
            kernel = kernel.flip([0,1])
        self.register_buffer('kernel', kernel[None, None, :, :]) 
        self.stride = stride
        self.padding = (kernel.shape[2] - 1) // 2 

    def forward(self, x):
        kernel_expanded = self.kernel.expand(x.size(1), -1, -1, -1) 
        x = F.conv2d(x, kernel_expanded, stride=self.stride, padding=self.padding, groups=x.size(1))
        return x

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8, lr_mul=0.01, hidden_dim=None):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        if hidden_dim is None:
            hidden_dim = w_dim 
        layers = [PixelNorm()] 
        for i in range(num_layers):
            in_f = z_dim if i == 0 else hidden_dim
            out_f = w_dim if i == num_layers - 1 else hidden_dim
            layers.append(EqualizedLinear(in_f, out_f, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        if x.ndim == 2: # (N, C) for z in MappingNetwork
            return x * (x.square().mean(dim=1, keepdim=True) + self.epsilon).rsqrt()
        elif x.ndim == 4: # (N, C, H, W) for feature maps (if used, not typical in StyleGAN2 synthesis blocks)
            return x * (x.square().sum(dim=1, keepdim=True) + self.epsilon).rsqrt() 
        else:
            raise ValueError(f"PixelNorm expects 2D or 4D input, got {x.ndim}D")

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resolution,
                 is_first_block=False, use_noise=True,
                 conv_clamp=None, resample_kernel=None): 
        super().__init__()
        self.is_first_block = is_first_block
        self.use_noise = use_noise
        self.resolution = resolution 
        if is_first_block:
            self.const_input = nn.Parameter(torch.randn(1, out_channels, 4, 4))
            self.conv1 = ModulatedConv2d(out_channels, out_channels, kernel_size=3, w_dim=w_dim, demodulate=True)
        else:
            self.conv0_up = ModulatedConv2d(in_channels, out_channels, kernel_size=3, w_dim=w_dim, up=True, resample_kernel=resample_kernel, demodulate=True)
            self.conv1 = ModulatedConv2d(out_channels, out_channels, kernel_size=3, w_dim=w_dim, demodulate=True)
        if use_noise:
            self.noise_injector0 = NoiseInjection(out_channels)
            self.noise_injector1 = NoiseInjection(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.ada_in0 = AdaIN(out_channels, w_dim) 
        self.ada_in1 = AdaIN(out_channels, w_dim)

    def forward(self, x, w0, w1, noise_inputs=None): 
        # w0 and w1 are style vectors of shape (N, w_dim) for ada_in0 and ada_in1 respectively
        noise0, noise1 = None, None
        if noise_inputs is not None:
            if len(noise_inputs) >= 1: noise0 = noise_inputs[0]
            if len(noise_inputs) >= 2: noise1 = noise_inputs[1]

        if self.is_first_block:
            x = self.const_input.repeat(w0.shape[0], 1, 1, 1) 
            if self.use_noise:
                x = self.noise_injector0(x, noise=noise0)
            x = self.activation(x)
            x = self.ada_in0(x, w0) 
            x_main = self.conv1(x, w0) # First conv also uses w0 for modulation
        else:
            x = self.conv0_up(x, w0) # Upsampling conv uses w0
            if self.use_noise:
                x = self.noise_injector0(x, noise=noise0)
            x = self.activation(x)
            x = self.ada_in0(x, w0)
            x_main = self.conv1(x, w1) # Second conv uses w1

        if self.use_noise:
            x_main = self.noise_injector1(x_main, noise=noise1)
        x_main = self.activation(x_main)
        x_main = self.ada_in1(x_main, w1) # Second AdaIN uses w1
        return x_main

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, lr_mul=1.0):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=kernel_size, w_dim=w_dim, demodulate=False, lr_mul=lr_mul)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, w_rgb, prev_rgb=None): # w_rgb is (N, w_dim)
        y = self.conv(x, w_rgb)
        y = y + self.bias 
        if prev_rgb is not None:
            if prev_rgb.shape[2:] != y.shape[2:]:
                 prev_rgb = F.interpolate(prev_rgb, size=y.shape[2:], mode='bilinear', align_corners=False)
            y = y + prev_rgb
        return y

class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, img_channels,
                 channel_base=32768, channel_max=512, num_fp16_res=0, conv_clamp=None,
                 resample_kernel=None): 
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4
        def nf(stage): 
            return min(int(channel_base / (2.0 ** stage)), channel_max)

        self.blocks = nn.ModuleList()
        self.torgbs = nn.ModuleList()
        self.num_ws = 0
        
        # Initial block (4x4)
        self.blocks.append(SynthesisBlock(nf(2), nf(2), w_dim, resolution=4, is_first_block=True, resample_kernel=resample_kernel))
        self.torgbs.append(ToRGB(nf(2), img_channels, w_dim))
        self.num_ws += 2 # Two AdaINs (and their convs' modulations) in the first block
        self.num_ws += 1 # For the ToRGB layer

        # Subsequent blocks
        for res_log2 in range(3, resolution_log2 + 1): 
            in_ch = nf(res_log2 - 1)
            out_ch = nf(res_log2)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim, resolution=2**res_log2, resample_kernel=resample_kernel))
            self.torgbs.append(ToRGB(out_ch, img_channels, w_dim))
            self.num_ws += 2 # Two AdaINs (and their convs' modulations) per block
            self.num_ws += 1 # One ToRGB layer
            
    def forward(self, ws, noise_mode='random', force_fp32=False, **block_kwargs):
        # ws: (N, self.num_ws, w_dim)
        if ws.ndim != 3 or ws.shape[1] != self.num_ws or ws.shape[2] != self.w_dim:
            raise ValueError(f"Expected ws of shape (N, {self.num_ws}, {self.w_dim}), got {ws.shape}")

        x = None 
        rgb_image = None
        w_idx = 0 

        for i, (block, torgb) in enumerate(zip(self.blocks, self.torgbs)):
            # Each SynthesisBlock now takes w0, w1 for its two main style application points
            w_style0 = ws[:, w_idx, :]
            w_idx += 1
            w_style1 = ws[:, w_idx, :]
            w_idx += 1
            
            if i == 0: 
                x = block(x, w_style0, w_style1) 
            else:
                x = block(x, w_style0, w_style1)

            w_style_rgb = ws[:, w_idx, :]
            w_idx += 1
            current_rgb = torgb(x, w_style_rgb, prev_rgb=rgb_image) 
            rgb_image = current_rgb 
        return rgb_image

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, num_mapping_layers, img_resolution, img_channels,
                 mapping_lr_mul=0.01, **synthesis_kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.mapping = MappingNetwork(z_dim, w_dim, num_mapping_layers, lr_mul=mapping_lr_mul)
        self.synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws 

    def forward(self, z, c=None, truncation_psi=1.0, truncation_cutoff=None, noise_mode='random', force_fp32=False, return_ws=False):
        ws_single = self.mapping(z) 
        # TODO: Implement w_avg tracking for truncation
        # For now, truncation is effectively disabled if w_avg is not present.
        
        # Expand ws_single to match self.num_ws for the synthesis network
        ws = ws_single.unsqueeze(1).repeat(1, self.num_ws, 1) # (N, num_ws, w_dim)

        img = self.synthesis(ws, noise_mode=noise_mode, force_fp32=force_fp32)
        if return_ws:
            return img, ws_single 
        return img

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resample_kernel=None): 
        super().__init__()
        self.conv0 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.blur = None # TODO: Implement FIR downsampling using Blur if resample_kernel
        self.conv1_down = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2) 
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=2)

    def forward(self, x):
        y = self.skip(x) 
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1_down(x)) 
        return (x + y) * (1 / np.sqrt(2)) 

class MinibatchStdDevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features # Must be 1 for this simplified version

    def forward(self, x):
        N, C, H, W = x.shape
        if self.num_new_features != 1:
            raise NotImplementedError("MinibatchStdDevLayer for num_new_features > 1 not fully implemented in this scratch version.")
        
        actual_group_size = min(self.group_size, N)
        if N % actual_group_size != 0: 
            actual_group_size = N // (N // actual_group_size) if (N // actual_group_size) > 0 else N
        if actual_group_size <= 0: return x

        y = x.view(actual_group_size, N // actual_group_size, C, H, W) 
        y = y - y.mean(dim=0, keepdim=True)    
        y = y.square().mean(dim=0)             
        y = (y + 1e-8).sqrt()                  
        y = y.mean(dim=[1,2,3], keepdim=True) # (G, 1, 1, 1) where G = N // actual_group_size
        y = y.repeat_interleave(actual_group_size, dim=0) # (N, 1, 1, 1)
        y = y.repeat(1, 1, H, W) # (N, 1, H, W)
        return torch.cat([x, y], dim=1)

class Discriminator(nn.Module):
    def __init__(self, img_resolution, img_channels,
                 channel_base=32768, channel_max=512, num_fp16_res=0, conv_clamp=None,
                 mbstd_group_size=4, mbstd_num_features=1,
                 resample_kernel=None, block_kwargs={}, mapping_kwargs={}, epilogue_kwargs={}):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4
        def nf(stage):
            return min(int(channel_base / (2.0 ** stage)), channel_max)

        self.fromrgb = EqualizedConv2d(img_channels, nf(resolution_log2 -1), kernel_size=1) 
        self.activation = nn.LeakyReLU(0.2)
        self.blocks = nn.ModuleList()
        for res_log2_current in range(resolution_log2, 2, -1): 
            in_ch = nf(res_log2_current - 1) 
            out_ch = nf(res_log2_current - 2)
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch, resample_kernel=resample_kernel))
        final_features_channels = nf(2) 
        self.mbstd = MinibatchStdDevLayer(group_size=mbstd_group_size, num_new_features=mbstd_num_features)
        final_conv_in_channels = final_features_channels + mbstd_num_features
        self.final_conv = EqualizedConv2d(final_conv_in_channels, nf(1), kernel_size=3, padding=1) 
        self.final_dense = EqualizedLinear(nf(1) * 4 * 4, nf(0), lr_mul=1.0) 
        self.output_layer = EqualizedLinear(nf(0), 1, lr_mul=1.0) 

    def forward(self, img, c=None, **block_kwargs):
        x = self.activation(self.fromrgb(img))
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        x = self.activation(self.final_conv(x))
        x = x.view(x.size(0), -1) 
        x = self.activation(self.final_dense(x))
        out = self.output_layer(x)
        return out

if __name__ == '__main__':
    print("Running basic network tests...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Z_DIM = 128; W_DIM = 128; IMG_RESOLUTION = 32; IMG_CHANNELS = 1
    N_MAPPING_LAYERS = 4; BATCH_SIZE = 2
    
    # Test Generator with correct ws handling for SynthesisNetwork
    gen = Generator(
        z_dim=Z_DIM, w_dim=W_DIM, num_mapping_layers=N_MAPPING_LAYERS,
        img_resolution=IMG_RESOLUTION, img_channels=IMG_CHANNELS,
        mapping_lr_mul=0.01,
        channel_base=512, # Adjusted for smaller test res
        channel_max=128,  # Adjusted for smaller test res
    ).to(device)
    
    test_z = torch.randn(BATCH_SIZE, Z_DIM).to(device)
    with torch.no_grad():
        fake_images, mapped_ws = gen(test_z, return_ws=True)
    print(f"Generator output shape: {fake_images.shape}") 
    assert fake_images.shape == (BATCH_SIZE, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION)
    print(f"Generator mapped_ws shape: {mapped_ws.shape}")
    assert mapped_ws.shape == (BATCH_SIZE, W_DIM)
    
    # Test SynthesisNetwork directly with correctly shaped ws
    synth_net = gen.synthesis
    test_ws_for_synth = torch.randn(BATCH_SIZE, synth_net.num_ws, W_DIM).to(device)
    with torch.no_grad():
        fake_images_from_synth = synth_net(test_ws_for_synth)
    print(f"SynthesisNetwork direct output shape: {fake_images_from_synth.shape}")
    assert fake_images_from_synth.shape == (BATCH_SIZE, IMG_CHANNELS, IMG_RESOLUTION, IMG_RESOLUTION)


    disc = Discriminator(
        img_resolution=IMG_RESOLUTION, img_channels=IMG_CHANNELS,
        channel_base=512, channel_max=128,
        mbstd_group_size=min(4, BATCH_SIZE) if BATCH_SIZE > 0 else 1
    ).to(device)
    output_logits = disc(fake_images)
    print(f"Discriminator output shape: {output_logits.shape}") 
    assert output_logits.shape == (BATCH_SIZE,1)
    print("Basic G/D network tests completed.")

    print("\nTesting ModulatedConv2d...")
    mod_conv = ModulatedConv2d(in_channels=3, out_channels=8, kernel_size=3, w_dim=W_DIM).to(device)
    test_x_conv = torch.randn(BATCH_SIZE, 3, 16, 16).to(device)
    test_w_conv = torch.randn(BATCH_SIZE, W_DIM).to(device)
    out_mod_conv = mod_conv(test_x_conv, test_w_conv)
    print(f"ModulatedConv2d output shape: {out_mod_conv.shape}") 
    assert out_mod_conv.shape == (BATCH_SIZE, 8, 16, 16)
    print("ModulatedConv2d test completed.")
