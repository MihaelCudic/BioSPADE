"""
Loader for the 3DFM_AC dataset
"""

import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.sampler.fmAC3D_sampler import FMAC3DSampler
import torch
import torch.nn.functional as F
import numpy as np
import random

class FMAC3DDataset(BaseDataset):
    
    def initialize(self, opt, mode):
        self.opt = opt
        self.prob_new_mesh = opt.prob_new_mesh
        no_pairing_check = False
        
        # Get paths for data
        stack_paths, gt_paths, mesh_paths  = self.get_paths(opt)
        
        strt = 0
        end = len(stack_paths)
        
        if opt.train_mode == 'GAN':
            end = int(end*(opt.data_split[0]+opt.data_split[1]))
        elif mode=='train':
            end = int(end*opt.data_split[0])
        elif mode=='valid':
            strt = int(end*opt.data_split[0])
            end = strt+int(end*opt.data_split[1])
        elif mode=='test':
            strt = end-int(end*opt.data_split[2])
            
        self.gt_paths = gt_paths[strt:end]
        self.stack_paths = stack_paths[strt:end]
        self.mesh_paths = mesh_paths
        
        self.stack_dataset_size = len(self.gt_paths)
        self.mesh_dataset_size = len(self.mesh_paths)

        if not no_pairing_check:
            for path1, path2 in zip(self.gt_paths, self.stack_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
                
        # Useful global variable
        self.half_Gs = opt.in_Gslices//2
        self.Ds = opt.in_Dslices
        self.delta_slice = opt.delta_slice
        
        # Initialize Samplers
        self.sampler = FMAC3DSampler(opt)

    def get_paths(self, opt):
        stack_dir = os.path.join(opt.stack_root, 'Stacks')
        stack_paths = make_dataset(stack_dir, recursive=True)
        
        gt_dir = os.path.join(opt.stack_root, 'GT')
        gt_paths = make_dataset(gt_dir, recursive=True)
        
        mesh_dir = opt.mesh_root
        mesh_paths = make_dataset(mesh_dir, recursive=True)

        return stack_paths, gt_paths, mesh_paths
    
    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext
    
    def __getitem__(self, index):
        # Get file
        gt_path = self.gt_paths[index]
        stack_path = self.stack_paths[index]
        assert self.paths_match(gt_path, stack_path), \
            "The label_path %s and image_path %s don't match." % \
            (gt_path, stack_path)
        mesh_path = random.choice(self.mesh_paths)
        
        # Load file
        self.sampler.load_stack(stack_path, gt_path)
        if not hasattr(self.sampler, 'vox'):
            self.sampler.load_mesh(mesh_path)
        elif random.random()<self.prob_new_mesh:
            self.sampler.load_mesh(mesh_path)
        
        # Sample stack data
        stack, gt, vox, powers, frames, z_pos = self.sampler()
        
        real_stack =  torch.Tensor(stack[:,::self.delta_slice]).div(255.0-18.0) # subtract 18 to normalize image to 0 and 1
        real_semantics =  torch.Tensor(gt[:,::self.delta_slice]).div(255.0)
        real_slices = real_semantics.clone() #[:, self.half_Gs:self.half_Gs+self.Ds]
        real_semantics = F.pad(real_semantics, (0,0,0,0,self.half_Gs,self.half_Gs), "constant", 0)

        powers = torch.Tensor(powers)
        frames = torch.Tensor(frames)
        z_pos = torch.Tensor(z_pos)
        
        # Sample mesh data
        mesh_semantics = torch.Tensor(vox[:,::self.delta_slice])
        mesh_slices = mesh_semantics.clone()[:, self.half_Gs:self.half_Gs+self.Ds]

        data_dict = {'real_stack': real_stack,
             'real_slices': real_slices,
             'real_semantics': real_semantics,
             'mesh_slices': mesh_slices,
             'mesh_semantics': mesh_semantics,
             'power': powers,
             'frames': frames, 
             'z_pos': z_pos,
             'stack_path': stack_path,
             'mesh_path': mesh_path
             }

        return data_dict
    
    def __len__(self):
        return self.stack_dataset_size