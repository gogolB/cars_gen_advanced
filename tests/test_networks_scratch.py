# tests/test_networks_scratch.py

import torch
import pytest
import numpy as np 

try:
    from src.models.stylegan2_networks_scratch import (
        EqualizedLinear, EqualizedConv2d, NoiseInjection, AdaIN,
        ModulatedConv2d, PixelNorm, MappingNetwork, SynthesisBlock,
        ToRGB, SynthesisNetwork, Generator, DiscriminatorBlock,
        MinibatchStdDevLayer, Discriminator
    )
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.models.stylegan2_networks_scratch import (
        EqualizedLinear, EqualizedConv2d, NoiseInjection, AdaIN,
        ModulatedConv2d, PixelNorm, MappingNetwork, SynthesisBlock,
        ToRGB, SynthesisNetwork, Generator, DiscriminatorBlock,
        MinibatchStdDevLayer, Discriminator
    )

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size(): return 2
@pytest.fixture
def z_dim(): return 64 # Smaller for tests
@pytest.fixture
def w_dim(): return 64 
@pytest.fixture
def img_channels(): return 1
@pytest.fixture
def base_img_resolution(): return 32 # Test with 32x32

def test_equalized_linear(device, batch_size):
    in_f, out_f = 16, 32
    layer = EqualizedLinear(in_f, out_f).to(device)
    x = torch.randn(batch_size, in_f).to(device)
    out = layer(x)
    assert out.shape == (batch_size, out_f)
    x_multi_dim = torch.randn(batch_size, 4, in_f).to(device)
    out_multi_dim = layer(x_multi_dim)
    assert out_multi_dim.shape == (batch_size, 4, out_f)

def test_equalized_conv2d(device, batch_size):
    in_c, out_c, k = 3, 8, 3
    layer = EqualizedConv2d(in_c, out_c, kernel_size=k, padding=1).to(device)
    x = torch.randn(batch_size, in_c, 16, 16).to(device)
    out = layer(x)
    assert out.shape == (batch_size, out_c, 16, 16)

def test_pixel_norm(device, batch_size):
    layer = PixelNorm().to(device)
    num_features_2d = 64
    x_2d = torch.randn(batch_size, num_features_2d, device=device) * 10 
    out_2d = layer(x_2d)
    assert out_2d.shape == x_2d.shape
    norm_2d = torch.linalg.norm(out_2d, dim=1)
    # PixelNorm with mean reduction should result in L2 norm of sqrt(num_features)
    expected_norm_2d = torch.full_like(norm_2d, float(np.sqrt(num_features_2d)))
    assert torch.allclose(norm_2d, expected_norm_2d, atol=1e-1), \
        f"Expected L2 norm ~sqrt(C)={np.sqrt(num_features_2d):.2f}, got {norm_2d.cpu().numpy()}"

    num_channels_4d = 3
    x_4d = torch.randn(batch_size, num_channels_4d, 16, 16, device=device) * 10
    out_4d = layer(x_4d) # This uses sum over channels for normalization
    assert out_4d.shape == x_4d.shape
    norm_per_pixel_vector = torch.linalg.norm(out_4d, dim=1) 
    assert torch.allclose(norm_per_pixel_vector, torch.ones_like(norm_per_pixel_vector), atol=1e-5), \
        "L2 norm of pixel vectors (across channels) after 4D PixelNorm should be ~1."
    assert not torch.allclose(out_4d, x_4d)

def test_noise_injection(device, batch_size):
    channels = 16
    layer = NoiseInjection(channels).to(device)
    x = torch.randn(batch_size, channels, 8, 8, device=device)
    
    with torch.no_grad(): # Ensure weight modification is not tracked by autograd
        layer.weight.fill_(1.0) 
    out_auto_noise = layer(x, noise=None)
    assert out_auto_noise.shape == x.shape
    # Check if any element is different, more robust than allclose for small noise
    assert not torch.equal(out_auto_noise, x), "Output should be different with non-zero noise weight"

    custom_noise = torch.randn(batch_size, 1, 8, 8, device=device)
    out_custom_noise = layer(x, noise=custom_noise) 
    assert out_custom_noise.shape == x.shape
    assert not torch.equal(out_custom_noise, x)

    with torch.no_grad():
        layer.weight.fill_(0.0)
    out_zero_weight = layer(x, noise=None)
    assert torch.allclose(out_zero_weight, x), "Output should be same with zero noise weight"

def test_adain(device, batch_size, w_dim):
    channels = 16
    layer = AdaIN(channels, w_dim).to(device)
    x = torch.randn(batch_size, channels, 8, 8, device=device)
    w = torch.randn(batch_size, w_dim, device=device)
    out = layer(x, w)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)

def test_modulated_conv2d(device, batch_size, w_dim):
    in_c, out_c, k = 4, 8, 3
    layer = ModulatedConv2d(in_c, out_c, kernel_size=k, w_dim=w_dim).to(device)
    x = torch.randn(batch_size, in_c, 16, 16, device=device)
    w = torch.randn(batch_size, w_dim, device=device)
    out = layer(x, w)
    assert out.shape == (batch_size, out_c, 16, 16)

    layer_up = ModulatedConv2d(in_c, out_c, kernel_size=k, w_dim=w_dim, up=True).to(device)
    out_up = layer_up(x, w)
    assert out_up.shape == (batch_size, out_c, 32, 32)

    layer_down = ModulatedConv2d(in_c, out_c, kernel_size=k, w_dim=w_dim, down=True).to(device)
    x_down_input = torch.randn(batch_size, in_c, 32, 32, device=device) 
    out_down = layer_down(x_down_input, w)
    assert out_down.shape == (batch_size, out_c, 16, 16)

def test_mapping_network(device, batch_size, z_dim, w_dim):
    num_layers = 4
    net = MappingNetwork(z_dim, w_dim, num_layers=num_layers).to(device)
    z = torch.randn(batch_size, z_dim, device=device)
    w = net(z)
    assert w.shape == (batch_size, w_dim)

def test_synthesis_block(device, batch_size, w_dim):
    res_4x4 = 4; ch_4x4 = 64
    res_8x8 = 8; ch_8x8 = 32
    
    w_style0 = torch.randn(batch_size, w_dim, device=device)
    w_style1 = torch.randn(batch_size, w_dim, device=device)

    block_first = SynthesisBlock(ch_4x4, ch_4x4, w_dim, res_4x4, is_first_block=True).to(device)
    out_first = block_first(None, w_style0, w_style1) 
    assert out_first.shape == (batch_size, ch_4x4, res_4x4, res_4x4)

    block_next = SynthesisBlock(ch_4x4, ch_8x8, w_dim, res_8x8).to(device)
    out_next = block_next(out_first, w_style0, w_style1)
    assert out_next.shape == (batch_size, ch_8x8, res_8x8, res_8x8)

def test_torgb(device, batch_size, w_dim, img_channels):
    in_c, res = 64, 8
    layer = ToRGB(in_c, img_channels, w_dim).to(device)
    x = torch.randn(batch_size, in_c, res, res, device=device)
    w_rgb = torch.randn(batch_size, w_dim, device=device)
    
    rgb_out = layer(x, w_rgb, prev_rgb=None)
    assert rgb_out.shape == (batch_size, img_channels, res, res)

    prev_rgb_img = torch.randn(batch_size, img_channels, res, res, device=device)
    rgb_out_combined = layer(x, w_rgb, prev_rgb=prev_rgb_img)
    assert rgb_out_combined.shape == (batch_size, img_channels, res, res)
    
    prev_rgb_smaller = torch.randn(batch_size, img_channels, res // 2, res // 2, device=device)
    rgb_out_upsampled_prev = layer(x, w_rgb, prev_rgb=prev_rgb_smaller)
    assert rgb_out_upsampled_prev.shape == (batch_size, img_channels, res, res)

def test_synthesis_network(device, batch_size, w_dim, base_img_resolution, img_channels):
    test_channel_base = 512 
    test_channel_max = 64   
    net = SynthesisNetwork(
        w_dim=w_dim, img_resolution=base_img_resolution, img_channels=img_channels,
        channel_base=test_channel_base, channel_max=test_channel_max 
    ).to(device)
    
    ws_input_correct_shape = torch.randn(batch_size, net.num_ws, w_dim, device=device) 
    img = net(ws_input_correct_shape) 
    assert img.shape == (batch_size, img_channels, base_img_resolution, base_img_resolution)

def test_generator(device, batch_size, z_dim, w_dim, base_img_resolution, img_channels):
    test_channel_base = 512
    test_channel_max = 64
    gen = Generator(
        z_dim=z_dim, w_dim=w_dim, num_mapping_layers=4,
        img_resolution=base_img_resolution, img_channels=img_channels,
        channel_base=test_channel_base, channel_max=test_channel_max
    ).to(device)
    
    z = torch.randn(batch_size, z_dim, device=device)
    img = gen(z)
    assert img.shape == (batch_size, img_channels, base_img_resolution, base_img_resolution)

    img_ws, ws_out_single = gen(z, return_ws=True)
    assert img_ws.shape == (batch_size, img_channels, base_img_resolution, base_img_resolution)
    assert ws_out_single.shape == (batch_size, w_dim) 

def test_discriminator_block(device, batch_size):
    block1 = DiscriminatorBlock(in_channels=32, out_channels=32).to(device)
    x1 = torch.randn(batch_size, 32, 16, 16, device=device)
    out1 = block1(x1)
    assert out1.shape == (batch_size, 32, 8, 8) 

    block2 = DiscriminatorBlock(in_channels=32, out_channels=64).to(device)
    out2 = block2(x1) # Use x1 as input
    assert out2.shape == (batch_size, 64, 8, 8)

def test_minibatch_stddev_layer(device, batch_size):
    if batch_size == 0: pytest.skip("Batch size is 0")
    
    group_s = min(4, batch_size)
    test_batch_size = batch_size
    # Ensure test_batch_size is compatible with group_s for this simplified test
    if batch_size % group_s != 0:
        test_batch_size = group_s * (batch_size // group_s)
        if test_batch_size == 0: test_batch_size = group_s
    
    in_c = 16
    layer = MinibatchStdDevLayer(group_size=group_s, num_new_features=1).to(device)
    x = torch.randn(test_batch_size, in_c, 8, 8, device=device)
    out = layer(x)
    assert out.shape == (test_batch_size, in_c + 1, 8, 8)

def test_discriminator(device, batch_size, base_img_resolution, img_channels):
    if batch_size == 0: pytest.skip("Batch size is 0")
    
    test_channel_base = 512
    test_channel_max = 64
    current_batch_size = batch_size
    mbstd_group = min(4, current_batch_size)
    if current_batch_size % mbstd_group != 0:
        current_batch_size = mbstd_group * (current_batch_size // mbstd_group)
        if current_batch_size == 0: current_batch_size = mbstd_group

    if current_batch_size == 0: pytest.skip("Adjusted batch size is 0")

    disc = Discriminator(
        img_resolution=base_img_resolution, img_channels=img_channels,
        channel_base=test_channel_base, channel_max=test_channel_max,
        mbstd_group_size=mbstd_group 
    ).to(device)
    
    img = torch.randn(current_batch_size, img_channels, base_img_resolution, base_img_resolution, device=device)
    logits = disc(img)
    assert logits.shape == (current_batch_size, 1)
