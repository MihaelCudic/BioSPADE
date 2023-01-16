"""
Functions sourced from https://github.com/NVlabs/SPADE and adjusted accordingly for bioSPADE
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.layers import FlattenGram, GramBlock, spatial_stddev_layer
import util.util as util
import torch

from util.util import tensorize_dict, mkdir, save_gif, save_image

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt, network):
        super().__init__()
        self.opt = opt
        self.network = network
        
        num_D = 1
        self.dim = getattr(opt, 'dim_'+network)
        self.num = getattr(opt, 'num_'+network)
        
        self.down_stride = 2
        if self.dim == '3D':
            self.down_stride = (1,2,2)

        for i in range(self.num):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = getattr(opt,'subarch_'+self.network)
        
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, self.network)
        elif subarch == 'gram':
            netD = GramDiscriminator(opt, self.network)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        pool = F.avg_pool2d
        if self.dim=='3D':
            pool = F.avg_pool3d
        return pool(input, kernel_size=3, stride=self.down_stride, padding=1, count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = True
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            
            if len(result)<self.num:
                input = self.downsample(input)
        return result
    
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt, network):
        super().__init__()
        self.opt = opt
        
        self.dim = getattr(opt, 'dim_'+network)
        D_layers = getattr(opt, 'n_layers_'+network)

        nf = getattr(opt, 'nf_'+network)
        norm = getattr(opt, 'norm_'+network)
        norm_layer = get_nonspade_norm_layer(opt, norm)
        
        do_instancenorm = getattr(opt, 'do_instancenorm_'+network)
        instancenorm_layer = get_nonspade_norm_layer(opt, 'instance')
        
        do_spatial_stddev = opt.spatial_stddev_layer

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        stride = 2
        
        in_slices = self.opt.in_Dslices
        conv = nn.Conv2d
        if self.dim  == '3D':
            in_slices = 1
            conv = nn.Conv3d
            stride = (1,2,2)

        input_nc = in_slices + opt.condition_on_power + opt.condition_on_frames + opt.condition_on_z
        if opt.paired_translation:
            input_nc += self.opt.in_Dslices
        
        first_layer = conv(input_nc, nf, kernel_size=kw, stride=stride, padding=padw)
        if do_instancenorm:
            first_layer = instancenorm_layer(first_layer, dim=self.dim)

        sequence = [[first_layer,
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, D_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            
            pre_layer = []
            if n == D_layers - 1:
                stride = 1
                if do_spatial_stddev:
                    nf_prev *= 2
                    pre_layer = [spatial_stddev_layer()]
                
            sequence += [pre_layer+
                         [norm_layer(conv(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw), dim=self.dim),
                          nn.LeakyReLU(0.2, False)
                         ]]

        sequence += [[conv(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        return results[1:]
    
# Define Gramian-based Discriminator
class GramDiscriminator(BaseNetwork):
    def __init__(self, opt, network):
        super().__init__()
        self.opt = opt
        
        self.dim = getattr(opt, 'dim_'+network)
        D_layers = getattr(opt, 'n_layers_'+network)

        nf = getattr(opt, 'nf_'+network)
        norm = getattr(opt, 'norm_'+network)
        norm_layer = get_nonspade_norm_layer(opt, norm)
        
        do_instancenorm = getattr(opt, 'do_instancenorm_'+network)
        instancenorm_layer = get_nonspade_norm_layer(opt, 'instance')

        in_slices = self.opt.in_Dslices
            
        conv = nn.Conv2d
        stride = 2
        if self.dim  == '3D':
            in_slices = 1
            conv = nn.Conv3d
            stride = (1,2,2) 
        
        input_nc = in_slices + opt.condition_on_power + opt.condition_on_frames + opt.condition_on_z
        if opt.paired_translation:
            input_nc += 1
            
        # Initialize first layer
        first_layer = conv(input_nc, nf, kernel_size=3, stride=1, padding=1)
        if do_instancenorm:
            first_layer = instancenorm_layer(first_layer, dim=self.dim)
        
        sequence = [[first_layer,
                     nn.ReLU(), 
                     norm_layer(conv(nf, nf, kernel_size=4, stride=stride, padding=2), dim=self.dim),
                     nn.ReLU()]]
        
        for n in range(1, D_layers):
            nf_prev = nf
            nf = 2*nf
            sequence += [[norm_layer(conv(nf_prev, nf, kernel_size=3, stride=1, padding=1), dim=self.dim),
                          nn.ReLU(),
                          norm_layer(conv(nf, nf, kernel_size=4, stride=stride, padding=1), dim=self.dim),
                          nn.ReLU()]]
        
        # Add Gramian block
        gram_nc = (nf)*(nf+1)//2
        sequence += [[GramBlock(),
                      FlattenGram(),
                      norm_layer(nn.Linear(gram_nc, gram_nc//2), dim='1D'),
                      nn.ReLU(),
                      nn.Linear(gram_nc//2, 1)]]
        
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for i, submodel in enumerate(self.children()):
            x = results[-1]
            intermediate_output = submodel(x)
            results.append(intermediate_output)
        return results[1:]