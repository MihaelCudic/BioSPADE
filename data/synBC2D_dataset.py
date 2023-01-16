"""
Loader for the 2DSyn_BC dataset
"""

import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.sampler.synBC2D_sampler import SynBC2DSampler
import torch
import numpy as np
import random

class SynBC2DDataset(BaseDataset):
    
    def initialize(self, opt, mode):
        self.opt = opt
        
        # Useful global variable
        self.half_Gs = opt.in_Gslices//2
        self.Ds = opt.in_Dslices
        
        # Get paths for data
        mesh_paths  = self.get_paths(opt)
        
        # Partition data
        strt = 0
        end = len(mesh_paths)
        if mode=='train':
            end = int(end*opt.data_split[0])
        elif mode=='val':
            strt = int(end*opt.data_split[0])
            end = strt+int(end*opt.data_split[1])
        elif mode=='test':
            strt = end-int(end*opt.data_split[2])
            
        mesh_paths = mesh_paths[strt:end]
        
        self.stack_paths = mesh_paths
        self.mesh_paths = mesh_paths
        
        self.stack_dataset_size = len(self.mesh_paths)
        self.mesh_dataset_size = len(self.mesh_paths)

        # Initialize Samplers
        self.sampler = SynBC2DSampler(opt)

    def get_paths(self, opt):
        mesh_dir = opt.mesh_root
        mesh_paths = make_dataset(mesh_dir, recursive=True)

        return mesh_paths

    def __getitem__(self, index):
        mesh_path = self.mesh_paths[index]
        
        # Load file
        samplers_per_interest = self.opt.samples_per_instance
        self.sampler.load(mesh_path)
        real_data, mesh_data, powers, frames, z_pos = self.sampler()
        
        real_stack = torch.Tensor(real_data['stack'])
        real_semantics = torch.Tensor(real_data['semantics'])
        real_slices = real_semantics.clone()[:, self.half_Gs:self.half_Gs+self.Ds]
        
        mesh_semantics = torch.Tensor(mesh_data['semantics'])
        mesh_slices = mesh_semantics.clone()[:, self.half_Gs:self.half_Gs+self.Ds]
        
        powers = torch.Tensor(powers)
        frames = torch.Tensor(frames)
        z_pos = torch.Tensor(z_pos)
        
        
        data_dict = {'real_stack': real_stack,
                     'real_slices': real_slices,
                     'real_semantics': real_semantics,
                     'mesh_slices': mesh_slices,
                     'mesh_semantics': mesh_semantics,
                     'power': powers,
                     'frames': frames, 
                     'z_pos': z_pos, 
                     'stack_path': mesh_path,
                     'mesh_path': mesh_path
                     }
        
        return data_dict
    
    def __len__(self):
        return self.stack_dataset_size