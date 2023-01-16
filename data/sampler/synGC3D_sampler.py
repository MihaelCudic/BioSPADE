'''
Sampler for 3DSyn_GC dataset
'''

import random
import numpy as np
from skimage.measure import block_reduce
from util.mesh_handler import *

class SynGC3DSampler(object):
    def __init__(self, opt, crop_size=None):
        
        self.z_pad = opt.delta_slice*(opt.in_Gslices//2)
        
        if crop_size is None:
            z_span = opt.delta_slice*(opt.in_Dslices-1)+1
            crop_sz = (z_span, *opt.crop_xy_sz)
        self.crop_sz = np.asarray(crop_sz)

        self.delta_z = opt.delta_z
        
        self.res = np.asarray(opt.mesh_res)
        self.rot = np.asarray(opt.mesh_rot)
        self.max_xy_sz = opt.max_xy_sz
        self.dim = opt.in_dim

        self.sigmas = opt.sigmas
        self.background = opt.background
        self.std_scalar = opt.std_scalar
        
        self.powers = opt.powers
        self.frames = opt.frames

        self.samples_per_instance = opt.samples_per_instance
        
        self.bounds = self.crop_sz//2

    def generate(self, input, ind, power, frames):
        x = input.sum(0) + 3*input[ind] # Amplify signal of chosen neuron
        x = np.maximum(x, self.background) # Inject background pixel value
        x = add_gauss_noise(x, power, frames, self.std_scalar) # Gaussian blur
        return x
        
    def load(self, files):
        self.n_samples = len(files)
        
        # Find mesh bounds of entire Ganglion Cell population
        mesh_ls = []
        bounds = np.zeros([len(files),2,3])
        for i,file in enumerate(files):
            mesh = Mesh(file)
            mesh.scale_z(min(self.res[1:])/self.res[0])
            bounds[i] = mesh.bounds
            mesh_ls += [mesh]
        bounds = np.concatenate([np.min(bounds[:,0,:],0)[None],np.max(bounds[:,1,:],0)[None]],0)
        
        # Load entire Ganglion Cell population
        sz = np.ceil((bounds[1,:]-bounds[0,:])/min(self.res))+1
        sz = sz.astype(int)
        vox = np.zeros([self.n_samples,*sz])
        vox_flipped = np.zeros_like(vox)
        stack = np.zeros_like(vox)
        for i, mesh in enumerate(mesh_ls):
            vox_i = mesh.voxelize(min(self.res), 0, bounds=bounds, dim=self.dim)
            vox[i] = vox_i
            vox_flipped[i] = vox_i[:,:,::-1]
            stack[i] = blur_vox(vox_i, sigmas=self.sigmas)
            
        self.vox = vox # Meshes used to produce stack
        self.vox_flipped = np.pad(vox_flipped,((0,0),(self.z_pad,self.z_pad),(0,0),(0,0)),"constant") # Meshes with no corresponding stack
        self.stack = stack

    def sample(self, power=None, frames=None):
        pad = 30
        
        end_sample = self.vox.shape[1:]-self.crop_sz
        
        # Randomly choose sample region to sample from
        has_vox = False
        z = None
        X_patch = Y_patch = stack_patch = None
        X_block = Y_block = None
        while(not has_vox): # Make sure there are neurons in sampled spatial region
            x = random.randint(pad,end_sample[1]-pad)
            y = random.randint(pad,end_sample[2]-pad)
            
            z_dist = self.vox_flipped[:,:,x:x+self.crop_sz[1], y:y+self.crop_sz[2]].sum(0).sum(-1).sum(-1)
            z_dist = z_dist[self.bounds[0]+self.z_pad:-(self.bounds[0]+self.z_pad)]
            
            nonzero_z = z_dist.nonzero()
            strt = np.min(nonzero_z)
            end = np.max(nonzero_z)
            z = random.choices(list(range(len(z_dist))), weights=z_dist/z_dist.sum())[0]

            X_patch = self.vox_flipped[:,
                                       z:z+2*self.z_pad+self.crop_sz[0], 
                                       x:x+self.crop_sz[1], 
                                       y:y+self.crop_sz[2]]
            XY_patch = self.vox[:,
                                 z:z+self.crop_sz[0], 
                                 x:x+self.crop_sz[1], 
                                 y:y+self.crop_sz[2]]

            Y_patch = self.stack[:,
                                 z:z+self.crop_sz[0], 
                                 x:x+self.crop_sz[1],
                                 y:y+self.crop_sz[2]]

            has_vox = X_patch[:,self.z_pad:-self.z_pad].sum()>0 and XY_patch.sum()>0
            if X_patch.shape[1] != self.crop_sz[0]+2*self.z_pad:
                has_vox=False
        
        # Find flipped mesh that overlaps to "real" neuron in question"
        # This is done to ensure that there is some correspondence between stack and mesh              
        block = (1,self.crop_sz[0],self.crop_sz[1]//4,self.crop_sz[2]//4)
        X_block = block_reduce(X_patch, block, np.mean) # downsample stack block
        XY_block = block_reduce(XY_patch, block, np.mean) # downsample flipped stack
            
        X_inds = X_block.sum(-1).sum(-1).sum(-1).nonzero()[0]

        XY_weight = XY_block.sum(-1).sum(-1).sum(-1)
        XY_ind = random.choices(list(range(self.n_samples)), weights=XY_weight/XY_weight.sum())[0] # Randomly choose overlapping mesh

        
        diff = ((X_block[:,0]-XY_block[XY_ind,0])**2).sum(-1).sum(-1)
        X_ind = X_inds[np.argmin(diff[X_inds])]
        
        XY_patch = XY_patch[XY_ind].copy()
        Y_patch = self.generate(Y_patch, XY_ind, power, frames)
        X_patch = X_patch[X_ind]
                
        return X_patch, XY_patch, Y_patch, (z-strt)*self.delta_z
        
        
    def __call__(self):
        powers = []
        frames = []
        z_pos = []
        
        real_stack = np.zeros([self.samples_per_instance,*self.crop_sz], dtype=np.float)
        real_slices = np.zeros([self.samples_per_instance,*self.crop_sz], dtype=np.float)
        mesh_slices = np.zeros([self.samples_per_instance,self.crop_sz[0]+2*self.z_pad,*self.crop_sz[1:]], dtype=np.float)

        for i in range(self.samples_per_instance):
            # Choose power
            power_ = random.choice(self.powers)
            powers += [power_]
            
            # Choose averaging
            frames_ = int(random.choice(self.frames))
            frames += [frames_]
            
            mesh_slices[i], real_slices[i], real_stack[i], z_ = self.sample(power_, frames_)
            z_pos += [z_]
        
        real_data = {'stack': real_stack, 'slices': real_slices}
        mesh_data = {'semantics': mesh_slices}
        
        return real_data, mesh_data, powers, frames, z_pos