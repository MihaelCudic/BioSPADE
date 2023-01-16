'''
Sampler for 3DFM_AC dataset
'''

import random
import numpy as np
import tifffile
from util.mesh_handler import *

class FMAC3DSampler(object):
    def __init__(self, opt, crop_size=None):
        self.isTrain = opt.isTrain
        
        self.z_pad = opt.delta_slice*(opt.in_Gslices//2)
        if crop_size is None:
            z_span = opt.delta_slice*(opt.in_Dslices-1)+1
            crop_sz = (z_span, *opt.crop_xy_sz)

        self.crop_sz = np.asarray(crop_sz)
        self.max_xy_sz = opt.max_xy_sz
        self.delta_z = opt.delta_z

        self.res = np.asarray(opt.mesh_res)
        self.rot = np.asarray(opt.mesh_rot)
        
        self.powers = opt.powers
        self.frames = opt.frames

        self.rand_trans = opt.rand_trans
        self.samples_per_instance = opt.samples_per_instance
        
        self.bounds = self.crop_sz//2+self.rand_trans
    
    def load_stack(self, stack_file, gt_file):
        # Load stack and corresponding ground truth 
        self.stack_file = stack_file
        self.stack = tifffile.imread(stack_file)
        gt = tifffile.imread(gt_file)
        self.gt = gt
        
        mean_stack = np.mean(self.stack[-1,:],axis=1)
        
        self.zprof_stack = np.mean(mean_stack, axis=(1,2))
        self.xprof_stack = np.mean(mean_stack[:1], axis=(0,2))
        self.yprof_stack = np.mean(mean_stack[:1], axis=(0,1))
        
    def load_mesh(self, mesh_file, rot=None, flip=None):
        # Load mesh
        self.mesh_file = mesh_file
        self.mesh = Mesh(mesh_file)
        self.mesh.flatten()
        self.mesh.flip(0)
        self.mesh.scale_z(min(self.res[1:])/self.res[0])
        
        if rot is None:
            self.mesh.rand_rotation(self.rot) # Randomly rotate mesh
        if flip is None:
            self.mesh.rand_flip(1) # Randomly flip mesh
        
        # Voxelize mesh
        self.vox = self.mesh.voxelize(min(self.res), 0, max_xy_sz=self.max_xy_sz, dim='3D', sub_sample=True)
        
        self.zprof_mesh = np.mean(self.vox, axis=(1,2))
        self.xprof_mesh = np.mean(self.vox[:5], axis=(0,2))
        self.yprof_mesh = np.mean(self.vox[:5], axis=(0,1))
        
    def overlay(self):
        # Overlay mesh with stack so that there is rough correspondence between the mesh
        
        stack = self.stack
        gt = self.gt
        vox = self.vox
        
        # Overlap mesh and vox        
        offset = []
        for dim_str in ['z', 'x', 'y']:
            prof_stack = getattr(self, dim_str+'prof_stack')
            prof_mesh = getattr(self, dim_str+'prof_mesh')
            conv = np.convolve(prof_stack[::-1],prof_mesh)
            offset += [-(np.argmax(conv)-len(prof_stack)+1)]
        z_offset, x_offset, y_offset = offset
                
        # Overlap Z
        z_pad_beg = self.z_pad
        if z_offset>0:
            stack = stack[:,z_offset:]
            gt = gt[z_offset:]
        elif z_offset<0:
            z_pad_beg -= abs(z_offset)
            if z_pad_beg<0:
                vox = vox[abs(z_pad_beg):]
                z_pad_beg=0
        vox = np.pad(vox,((z_pad_beg,0),(0,0),(0,0)),"constant")  
        
        z_pad_end = self.z_pad
        z_add = gt.shape[0]-vox.shape[0]+self.z_pad
        if z_add>0:
            z_pad_end += z_add
        elif z_add<0:
            z_pad_end -= abs(z_add)
            if z_pad_end<0:
                vox = vox[:-(abs(z_pad_end))]
                z_pad_end = 0
        vox = np.pad(vox,((0,z_pad_end),(0,0),(0,0)),"constant")
        
        non_vox = vox.copy()
            
        # Overlap X
        if x_offset<0:
            vox = np.pad(vox[:,abs(x_offset):],((0,0),(0,abs(x_offset)),(0,0)),"constant")
        elif x_offset>0:
            vox = np.pad(vox[:,:-x_offset],((0,0),(x_offset,0),(0,0)),"constant")
            
        # Overlap Y
        if y_offset<0:
            vox = np.pad(vox[:,:,abs(y_offset):],((0,0),(0,0),(0,abs(y_offset))),"constant")
        elif y_offset>0:
            vox = np.pad(vox[:,:,:-y_offset],((0,0),(0,0),(y_offset,0)),"constant")
        
        return vox, stack, gt

    def __call__(self):

        vox, stack, gt = self.overlay()

        # Choose XYZ locations
        bound_vox = vox[self.bounds[0]+self.z_pad:-(self.bounds[0]+self.z_pad),
                                           self.bounds[1]:-(self.bounds[1]),
                                           self.bounds[2]:-(self.bounds[2])]
        
        random_crop = False
        if np.prod(bound_vox.shape) == 0:
            random_crop = True
        else:
            bound_inds = np.transpose(bound_vox.nonzero())
            inds = bound_inds[random.choices(range(len(bound_inds)), k=self.samples_per_instance)]
        
        powers = []
        frames = []
        z_pos = []
        stack_patch = np.zeros((self.samples_per_instance, *self.crop_sz), dtype=np.uint8)
        if not self.isTrain:
            stack_patch = np.zeros((self.samples_per_instance, len(self.powers), 
                                    self.crop_sz[0], len(self.frames), *self.crop_sz[1:]),
                                    dtype=np.uint8)
        gt_patch = np.zeros((self.samples_per_instance, self.crop_sz[0], *self.crop_sz[1:]), dtype=np.uint8)
        vox_patch = np.zeros([self.samples_per_instance, self.crop_sz[0]+2*self.z_pad, *self.crop_sz[1:]], dtype=np.float)
        for i in range(self.samples_per_instance):
            z = random.randint(0, vox.shape[0]-2*(self.bounds[0]+self.z_pad))
            x = 0
            y = 0
            if not random_crop:
                z,x,y = inds[i]
                z += random.randint(0, 2*self.rand_trans[0]+1)
                x += random.randint(0, 2*self.rand_trans[1]+1)
                y += random.randint(0, 2*self.rand_trans[2]+1)
            
            z_pos += [z*self.delta_z]
            
            # Choose power
            powers_i = np.random.randint(len(self.powers))
            powers_ = self.powers[powers_i]
            powers += [powers_]
            
            # Choose averaging
            frames_i = np.random.randint(len(self.frames))+1
            frames_ = self.frames[frames_i-1]
            frames += [frames_]
            
            # Crop patch
            stack_patch_ = None
            if self.isTrain:
                stack_patch_ = stack[int(powers_),z:z+self.crop_sz[0],
                     :int(frames_),x:x+self.crop_sz[1], y:y+self.crop_sz[2]]
                stack_patch_ = np.mean(stack_patch_,axis=1)     
            else:
                stack_patch_ = stack[:,z:z+self.crop_sz[0],:,x:x+self.crop_sz[1], y:y+self.crop_sz[2]]
            
            stack_patch[i] = stack_patch_                 
          
            gt_patch_ = gt[z:z+self.crop_sz[0],
                           x:x+self.crop_sz[1],y:y+self.crop_sz[2]]
            gt_patch[i] = gt_patch_
            
            vox_patch_ = vox[z:z+self.crop_sz[0]+2*self.z_pad,
                             x:x+self.crop_sz[1],y:y+self.crop_sz[2]]
            vox_patch[i] = vox_patch_
            
        return stack_patch, gt_patch, vox_patch, powers, frames, z_pos