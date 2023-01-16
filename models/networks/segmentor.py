import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer

# UNet Segmentation Architecture
class UNETSegmentor(BaseNetwork):
    def __init__(self, opt, mode):
        super().__init__()
        self.opt = opt
        nf = 32 if opt.train_mode=='GAN' else opt.nf_Seg
        self.n_downsampling_layers = 0 if opt.train_mode=='GAN' else self.opt.n_down_layers_Seg-1
        input_nc = self.opt.in_Dslices if opt.train_mode=='GAN' else self.opt.in_Sslices
        if opt.train_mode!='GAN':
            input_nc += opt.condition_on_power + opt.condition_on_frames + opt.condition_on_z
        
        instance_layer = get_nonspade_norm_layer(opt, 'instance')
        
        norm = getattr(opt, 'norm_Seg')
        norm_layer = get_nonspade_norm_layer(opt, norm)

        sequence = [[instance_layer(nn.Conv2d(input_nc, nf, 3, padding=1)),
                     nn.ReLU(),
                     norm_layer(nn.Conv2d(nf, nf, 3, padding=1)),
                     nn.ReLU()]]
        
        nf_ls = [nf]
        for i in range(self.n_downsampling_layers):
            nf_prev = nf
            nf = min(2*nf, 512)
            nf_ls += [nf]
            
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf_prev, 4, stride=2, padding=1)), 
                          nn.ReLU(),
                          norm_layer(nn.Conv2d(nf_prev, nf, 3, padding=1)), 
                          nn.ReLU(),
                          norm_layer(nn.Conv2d(nf, nf, 3, padding=1)),
                          nn.ReLU()]]
        
        sequence += [[norm_layer(nn.Conv2d(nf, nf, 4, stride=2, padding=1)),
                      nn.ReLU(),]]
        for i, nf_prev in enumerate(nf_ls[::-1]):
            
            nf_last = nf_prev
            nf = min(2*nf_prev, 512)
            nf_prev = nf*2 if i>0 else nf_prev
            
            
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, 3, padding=1)), 
                          nn.ReLU(),
                          norm_layer(nn.Conv2d(nf, nf_last, 3, padding=1)),
                          nn.ReLU(),
                          nn.Upsample(scale_factor=2)]]

        sequence += [[norm_layer(nn.Conv2d(2*nf_last, nf_last, 3, padding=1)), 
                      nn.ReLU()]]
        
        sequence += [[nn.Conv2d(nf_last+input_nc, 2, 1),
                      nn.Softmax(dim=1)]]
        
        self.n_seq = len(sequence)
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))
        
    def forward(self, input):
        
        results = [input]
        x = input
        for i, submodel in enumerate(self.children()):
            i_rev = self.n_seq-i
            
            if i<self.n_downsampling_layers+1:
                x = submodel(x)
                results = [x]+results
            elif abs(i_rev)<=self.n_downsampling_layers+2:
                pre = results[-i_rev]
                
                x = torch.cat([pre, x],1)
                x = submodel(x)
            else:
                x = submodel(x)
                
        return x