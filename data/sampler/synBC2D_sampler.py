'''
Sampler for 2DSyn_BC dataset
'''

import random
import numpy as np
from util.mesh_handler import *

class SynBC2DSampler(object):
    def __init__(self, opt, crop_size=None):
        
        self.z_pad = opt.delta_slice*(opt.in_Gslices//2)
        
        if crop_size is None:
            z_span = opt.delta_slice*(opt.in_Dslices-1)+1
            crop_sz = (z_span, *opt.crop_xy_sz)
        self.crop_sz = list(crop_sz)
        self.crop_sz[0] = self.crop_sz[0]+2*self.z_pad
        
        self.res = np.asarray(opt.mesh_res)
        self.rot = np.asarray(opt.mesh_rot)
        self.max_xy_sz = opt.max_xy_sz
        self.dim = opt.in_dim

        self.sigmas = opt.sigmas
        self.background = opt.background
        self.p_blobs = opt.p_blobs
        self.std_scalar = opt.std_scalar
        
        self.powers = opt.powers
        self.frames = opt.frames

        self.rand_trans = opt.rand_trans
        self.samples_per_instance = opt.samples_per_instance
        
        self.rand_flip = opt.train_mode != 'GAN'
        
    def generate(self, input, power, frames, std_scalar=None, bin_mat=None):
        if std_scalar is None:
            std_scalar = self.std_scalar
        
        x = np.maximum(input, self.background) # Inject background pixel value
        x = add_blobs(x, self.p_blobs, self.background, bin_mat) # Add random blobs
        x = add_gauss_noise(x, power, frames, std_scalar) # Blur image
        return x

    def load(self, mesh_file):
        if mesh_file is None:
            mesh_file = self.mesh_file
        else:
            self.mesh_file = mesh_file
        
        # Load mesh
        self.mesh = Mesh(mesh_file) 
        #self.mesh.flatten()
        self.mesh.scale_z(min(self.res[1:])/self.res[0])
        self.mesh.rand_rotation(self.rot)     
        
        if self.rand_flip:
            self.mesh.rand_flip(1)
            
        # Voxelize mesh
        self.vox = self.mesh.voxelize(min(self.res), self.z_pad, None, self.max_xy_sz, self.dim, False)
        
        # Flip vox
        self.vox_flipped = self.vox[:,::-1,:]

        # Stylize Vox
        self.stack = blur_vox(self.vox, sigmas=self.sigmas)
        
    def sample(self, z, x, y, vox, power=None, frames=None):
        # Randomly samply patch
        
        out = np.zeros(self.crop_sz, dtype=np.float)
        
        z_strt = max(0,z)
        x_strt = max(0,x)
        y_strt = max(0,y)
        
        sample = vox[z_strt:z_strt+self.crop_sz[0],
                     x_strt:x_strt+self.crop_sz[1],
                     y_strt:y_strt+self.crop_sz[2]]
        
        out[:sample.shape[0],
            :sample.shape[1],
            :sample.shape[2]] = sample
        
        if power is not None and frames is not None:
            out = self.generate(out, power, frames)
                
        return out
        
        
    def __call__(self):
        
        vox = self.vox
        stack = self.stack
        vox_flipped = self.vox_flipped
        
        # Choose XYZ locations
        inds = np.transpose(vox.nonzero())
        rand_inds = inds[np.random.choice(len(inds), size=self.samples_per_instance, replace=False)]
            
        # Randomize XYZ 
        rand_z = np.random.randint(-self.rand_trans[0], self.rand_trans[0]+1,
                                   size=[self.samples_per_instance,2])-self.crop_sz[0]//2
        rand_x = np.random.randint(-self.rand_trans[1], self.rand_trans[1]+1,
                                   size=[self.samples_per_instance,2])-self.crop_sz[1]//2
        rand_y = np.random.randint(-self.rand_trans[2], self.rand_trans[2]+1,
                                   size=[self.samples_per_instance,2])-self.crop_sz[2]//2

        vox_inds = rand_inds+np.concatenate([rand_z[:,:1], rand_x[:,:1], rand_y[:,:1]],1)
        
        power = []
        frames = []
        z_pos = []
        
        real_stack = np.zeros([self.samples_per_instance,self.crop_sz[0]-2*self.z_pad,*self.crop_sz[1:]], dtype=np.float)
        real_slices = np.zeros([self.samples_per_instance,*self.crop_sz], dtype=np.float)
        mesh_slices = np.zeros([self.samples_per_instance,*self.crop_sz], dtype=np.float)
        for i in range(self.samples_per_instance):
            z,x,y = vox_inds[i]
            
            z_pos += [z]
            
            # Choose power
            power_ = random.choice(self.powers)
            power += [power_]
            
            # Choose averaging
            frames_ = int(random.choice(self.frames))
            frames += [frames_]
            
            real_slices[i] = self.sample(z, x, y, vox)
            real_stack[i] = self.sample(z, x, y, stack, power_, frames_)
            mesh_slices[i]  = self.sample(z, x, y, vox_flipped)
        
        real_data = {'stack': real_stack, 'semantics': real_slices}
        mesh_data = {'semantics': mesh_slices}
        
        return real_data, mesh_data, power, frames, z_pos