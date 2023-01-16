'''
Sampler for 3DFM_BCPop dataset
'''

import random
import numpy as np
import tifffile
from scipy import stats
from util.mesh_handler import *

class FMBCPop3DSampler(object):
    def __init__(self, opt, stack_file, mesh_file, random_mesh=False, crop_size=None):
        self.z_pad = opt.delta_slice*(opt.in_Gslices//2)
        
        if crop_size is None:
            z_span = opt.delta_slice*(opt.in_Dslices-1)+1
            crop_sz = (z_span, *opt.crop_xy_sz)
        self.crop_sz = list(crop_sz)
        self.crop_sz[0] = self.crop_sz[0]+2*self.z_pad
        
        self.dim = opt.in_dim

        self.samples_per_instance = opt.samples_per_instance
        
        stack = tifffile.imread(stack_file)
        stack_mode = stats.mode(stack[:,:,:128,:128].flatten()).mode[0]
        
        self.stack = (255.0/(255.0-stack_mode))*np.maximum(stack.astype(float)-stack_mode, 0) # Normalize image between 0 and 1
        
        self.gt = tifffile.imread(mesh_file)>0
        
        self.stack_sz = self.stack.shape
        
        self.powers = opt.powers
        self.frames = opt.frames
        self.random_mesh = random_mesh
        self.avg_frames = False
        if not random_mesh:
            self.avg_frames = True
        
    def sample(self, f=None, z=None, x=None, y=None):
        
        if x is None:
            x = random.randint(0,self.stack_sz[2]-self.crop_sz[1])
        if y is None:
            y = random.randint(0,self.stack_sz[3]-self.crop_sz[2])
        if z is None:
            z = random.randint(0,self.stack_sz[0]-self.crop_sz[0])
        if f is None:
            f = self.frames[np.random.randint(0,len(self.frames))]
            f = int(f)
        
        mesh = self.gt[z:z+self.crop_sz[0],
                       x:x+self.crop_sz[1],
                       y:y+self.crop_sz[2]]
        
        stack = self.stack[z+self.z_pad:z+self.crop_sz[0]-self.z_pad,:f,
                           x:x+self.crop_sz[1],
                           y:y+self.crop_sz[2]]
        stack = stack.mean(1)
                
        return stack, mesh, f, z, x, y
        
        
    def __call__(self):

        powers = []
        frames = []
        z_pos = []
        
        real_stack = np.zeros([self.samples_per_instance,self.crop_sz[0]-2*self.z_pad,*self.crop_sz[1:]], dtype=np.float)
        mesh_stack = np.zeros_like(real_stack)
        
        real_slices = np.zeros([self.samples_per_instance,*self.crop_sz], dtype=np.float)
        mesh_slices = np.zeros_like(real_slices)
        for i in range(self.samples_per_instance):
            real_stack_, real_slices_, f_, z_, x_, y_ = self.sample()
            
            if self.random_mesh:
                x_ = None
                y_ = None
                
            if self.avg_frames:
                f_ = 8
                
            mesh_stack_, mesh_slices_, f_, z_, x_, y_ = self.sample(f_, z_, x_, y_)
            
            real_stack[i] =  real_stack_
            real_slices[i] = real_slices_
            
            mesh_stack[i] = mesh_stack_
            mesh_slices[i] = mesh_slices_
            
            powers += [self.powers]
            frames += [f_]
            z_pos += [z_]
            
        real_data = {'stack': real_stack, 'semantics': real_slices}
        mesh_data = {'stack': mesh_stack, 'semantics': mesh_slices}
        
        return real_data, mesh_data, powers, frames, z_pos