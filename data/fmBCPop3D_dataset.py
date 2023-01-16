"""
Loader for the 3DFM_BCPop dataset
"""

import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.sampler.fmBCPop3D_sampler import FMBCPop3DSampler
import torch
import numpy as np
import random

class FMBCPop3DDataset(BaseDataset):
    
    def initialize(self, opt, mode):
        self.opt = opt
        
        # Useful global variable
        self.half_Gs = opt.in_Gslices//2
        self.Ds = opt.in_Dslices
        
        # Get paths for data
        stack_paths, mesh_paths  = self.get_paths(opt)

        # Partition Data
        random_mesh = False
        if opt.train_mode == 'GAN':
            self.stack_path = stack_paths[2]
            self.mesh_path = mesh_paths[2]
            random_mesh = True
        elif mode=='train':
            self.stack_path = stack_paths[2]
            self.mesh_path = mesh_paths[2]
            random_mesh = False
        elif mode=='valid':
            self.stack_path = stack_paths[1]
            self.mesh_path = mesh_paths[1]
        elif mode=='test':
            self.stack_path = stack_paths[0]
            self.mesh_path = mesh_paths[0]
        
        self.stack_dataset_size = (2048//opt.crop_xy_sz[0])**2
        self.mesh_dataset_size = self.stack_dataset_size

        # Initialize Samplers
        self.sampler = FMBCPop3DSampler(opt, self.stack_path, self.mesh_path, random_mesh)

    def get_paths(self, opt):
        stack_dir = opt.stack_root
        stack_paths = make_dataset(stack_dir, recursive=True)
        
        mesh_dir = opt.mesh_root
        mesh_paths = make_dataset(mesh_dir, recursive=True)

        return stack_paths, mesh_paths

    def __getitem__(self, index):
        
        # Load file
        real_data, mesh_data, power, frames, z_pos = self.sampler()
        
        real_stack = torch.Tensor(real_data['stack']).div(255.0)
        real_semantics = torch.Tensor(real_data['semantics'])
        real_slices = real_semantics.clone()[:, self.half_Gs:self.half_Gs+self.Ds]
        
        mesh_stack = torch.Tensor(mesh_data['stack']).div(255.0)
        mesh_semantics = torch.Tensor(mesh_data['semantics'])
        mesh_slices = mesh_semantics.clone()[:, self.half_Gs:self.half_Gs+self.Ds]
        
        powers = torch.Tensor(power)
        frames = torch.Tensor(frames)
        z_pos = torch.Tensor(z_pos)*self.opt.delta_z
        
        data_dict = {'real_stack': real_stack,
                     'real_slices': real_slices,
                     'real_semantics': real_semantics,
                     'mesh_stack': mesh_stack,
                     'mesh_slices': mesh_slices,
                     'mesh_semantics': mesh_semantics,
                     'power': powers,
                     'frames': frames,
                     'z_pos': z_pos, 
                     'stack_path': self.stack_path,
                     'mesh_path': self.mesh_path
                     }
        
        return data_dict
    
    def __len__(self):
        return self.stack_dataset_size