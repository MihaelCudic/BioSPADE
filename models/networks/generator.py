"""
Functions sourced from https://github.com/NVlabs/SPADE and adjusted accordingly for bioSPADE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.normalization import get_nonspade_norm_layer

def compute_latent_vector_size(opt):
    if opt.num_upsampling_layers == 'small':
        num_up_layers = 4
    elif opt.num_upsampling_layers == 'normal':
        num_up_layers = 5
    elif opt.num_upsampling_layers == 'more':
        num_up_layers = 6
    elif opt.num_upsampling_layers == 'most':
        num_up_layers = 7
    else:
        raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                         opt.num_upsampling_layers)
    return num_up_layers

# Declare final layer of G
class Acquisition_Layer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.acquire_Gaussian_noise = opt.acquire_Gaussian_noise # Inject Gaussian noise
        self.acquire_Frechet_noise  = opt.acquire_Frechet_noise # Inject Frechet noise
        
    def forward(self, input, frames=1):
        acquired = torch.zeros_like(input[:,:1])
        with torch.no_grad():
            frames_npy = frames.view(-1).cpu().numpy().astype(int)
        sz = list(input.shape)
        sz[1] = max(frames_npy)
        frames = torch.ones(sz).cuda() * frames # Create multiple frames of existing G output
        
        # Aquisition
        bias = F.relu(input[:,1:2].mean(axis=[1,2,3],keepdims=True))
        
        sampled_px = torch.zeros_like(frames) + input[:,:1] + bias
        center_px = input[:,:1] + bias
        
        n_ch = 2
        if self.acquire_Frechet_noise: # Add Frechet noise
            frechet_weight = F.relu(input[:,n_ch:n_ch+1])
            frechet_alpha = F.relu(input[:,n_ch+1:n_ch+2])
            
            noise = torch.clamp(torch.rand_like(frames), 1e-6, 1-1e-6)
            log_noise = torch.log(noise)
            pow_noise = (-1*log_noise)**(-frechet_alpha)
            frechet_noise = frechet_weight*pow_noise
            sampled_px = sampled_px + frechet_noise
            pow_center = (1/(frechet_alpha+1))**frechet_alpha
            frechet_center = frechet_weight*pow_center
            center_px = center_px+frechet_center
            
            n_ch += 2
            
        if self.acquire_Gaussian_noise: # Add Gaussian noise
            gaussian_weight = F.relu(input[:,n_ch:n_ch+1])
            gaussian_noise = gaussian_weight * torch.randn_like(frames)
            sampled_px = sampled_px + gaussian_noise
            
            
        # Go through acquired frames and average accordingly
        acquired_frames = torch.clamp(sampled_px,0,1)
        for i in range(len(acquired_frames)):
            n_frames = frames_npy[i]
            acquired[i] = acquired_frames[i,:n_frames].mean(0, keepdims=True)
        center = torch.clamp(center_px, 0, 1)
        
        return acquired, center

class SPADEGenerator(BaseNetwork):
    def __init__(self, opt, network):
        super().__init__()
        self.opt = opt
        self.network = network
        self.aquisition = opt.aquisition
        self.mid_slice = opt.in_Gslices//2
        nf = getattr(opt, 'nf_'+network)
        
        first_norm = opt.first_norm_G
        if opt.first_norm_G is None:
            first_norm = opt.norm_G
            
        out_nc = 1
        if self.aquisition:
            out_nc += opt.acquire_Gaussian_noise + 1
            if opt.acquire_Frechet_noise:
                out_nc += 2
            self.acq_layer = Acquisition_Layer(opt)
            
        self.condition_on_power = opt.condition_on_power
        self.condition_on_frames = opt.condition_on_frames
        self.condition_on_z = opt.condition_on_z
        self.conditions = opt.condition_on_power + opt.condition_on_frames + opt.condition_on_z
        
        self.add_channel_noise = opt.add_channel_noise
        
        add_channels = self.conditions + self.opt.in_Gslices 
        input_nc = self.opt.in_Gslices + add_channels 

        self.num_up_layers = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(input_nc, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, network, first_norm)

        if opt.num_upsampling_layers != 'small':
            self.G_middle_0 = SPADEResnetBlock(16 * nf + add_channels, 16 * nf, opt, network, first_norm)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, network, first_norm)

        self.up_0 = SPADEResnetBlock(16 * nf + add_channels, 8 * nf, opt, network)
        self.up_1 = SPADEResnetBlock(8 * nf + add_channels, 4 * nf, opt, network)
        self.up_2 = SPADEResnetBlock(4 * nf + add_channels, 2 * nf, opt, network)
        self.up_3 = SPADEResnetBlock(2 * nf + add_channels, 1 * nf, opt, network)
        
        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf + add_channels, nf // 2, opt, network)
            final_nc = nf // 2

        self.up = nn.Upsample(scale_factor=2)
        self.conv_img = nn.Conv2d(final_nc+2, out_nc, 3, padding=1)


    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'small':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        return num_up_layers
    
    # Concatenate input with noise and optical configurations
    def concat(self, input, noise, power=None, frames=None, z_pos=None):
        x = input
        
        if self.condition_on_power: # concat power
            power_mat = torch.ones([input.size(0), 1, *input.shape[-2:]], 
                                   dtype=torch.float32, device=input.get_device()) * power
            x = torch.cat([x,power_mat],1)
        if self.condition_on_frames: # concat frames
            frames_mat = torch.ones([input.size(0), 1, *input.shape[-2:]],
                                   dtype=torch.float32, device=input.get_device()) * frames
            x = torch.cat([x,frames_mat],1)
        if self.condition_on_z: # concat z position
            z_mat = torch.ones([input.size(0), 1, *input.shape[-2:]],
                                    dtype=torch.float32, device=input.get_device()) * z_pos
            x = torch.cat([x,z_mat],1)
            
            
        x = torch.cat([x,noise],1) 
        return x

    def forward(self, input, noise, power=None, frames=None, z_pos=None):
        sw = input.shape[-2] // (2**self.num_up_layers)
        sh = input.shape[-1] // (2**self.num_up_layers)
        
        # SPADE
        seg = input
        mid = input[:,self.mid_slice:self.mid_slice+1]
        
        x = F.interpolate(seg, size=(sh, sw))
        x = self.concat(x, noise[-1], power, frames, z_pos)
        
        x = self.fc(x)
            
        x = self.head_0(x, seg)
        
        ind_adj = 1
        if self.opt.num_upsampling_layers != 'small':
            x = self.up(x)
            x = self.G_middle_0(self.concat(x, noise[-2], power, frames, z_pos), seg)
            ind_adj = 0

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(self.concat(x, noise[-3+ind_adj], power, frames, z_pos), seg)
        x = self.up(x)
        x = self.up_1(self.concat(x, noise[-4+ind_adj], power, frames, z_pos), seg)
        x = self.up(x)
        x = self.up_2(self.concat(x, noise[-5+ind_adj], power, frames, z_pos), seg)
        x = self.up(x)
        x = self.up_3(self.concat(x, noise[-6+ind_adj], power, frames, z_pos), seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(self.concat(x, noise[-7+ind_adj], power, frames, z_pos), seg)
        
        x = torch.cat([x,mid,torch.randn_like(x[:,:1])],1)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        
        if self.aquisition:
            return self.acq_layer(x, frames)
        else:
            x = torch.clamp(x,0,1)
            return x, x