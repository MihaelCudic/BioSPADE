import numpy as np
import random
import math

import trimesh
from scipy.ndimage import gaussian_filter, maximum_filter, zoom
from trimesh.voxel.ops import sparse_to_matrix
from PIL import Image,ImageDraw

# Blockwise averaging for np.array
def blockwise_average_3D(A,S):    
    # A is the 3D input array
    # S is the blocksize on which averaging is to be performed
    pad_z = A.shape[0]%S[0]
    pad_x = A.shape[1]%S[1]
    pad_y = A.shape[2]%S[2]
    
    A = np.pad(A, ((0,pad_z), (0, pad_x), (0, pad_y)))

    m,n,r = np.array(A.shape)//S
    return A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))

# Create rotation matrix given x,y,z rotation in radians
def rotation_matrix(x_rad, y_rad, z_rad):
    rot_mat = np.eye(4)

    Rx = np.asarray([[1, 0, 0], 
         [0, np.cos(x_rad), -np.sin(x_rad)], 
         [0, np.sin(x_rad), np.cos(x_rad)]])

    Ry = np.asarray([[np.cos(y_rad), 0, np.sin(y_rad)], 
         [0, 1, 0], 
         [-np.sin(y_rad), 0, np.cos(y_rad)]])

    Rz = np.asarray([[np.cos(z_rad), -np.sin(z_rad), 0], 
         [np.sin(z_rad), np.cos(z_rad), 0], 
         [0, 0, 1]])

    Rxyz = np.dot(np.dot(Rx,Ry),Rz)
    rot_mat[:3,:3] = Rxyz

    return rot_mat
    
# Create scaling matrix
def scale_matrix(x_scale, y_scale, z_scale):
    scale_matrix = np.eye(4)
    S =  np.asarray([[x_scale, 0, 0],
                     [0, y_scale, 0], 
                     [0, 0, z_scale]])
    scale_matrix[:3,:3] = S
    return scale_matrix

# Flip matrix on designated axis
def flip_matrix(axis):
    flip_mat = np.eye(4)
    flip_mat[axis,axis] = -1
    return flip_mat
 
# Collapse matrix on designated axis
def collapse_matrix(axis):
    collapse_matrix =  scale_matrix = np.eye(4)
    collapse_matrix[axis,axis] = 0
    return collapse_matrix
    
# Code to load, manipulate, and voxelize mesh
class Mesh(object):
    def __init__(self, file):
        if isinstance(file, str):
            self.mesh = trimesh.load(file)
        else:
            self.mesh = trimesh.load(file[0])
            for i in range(1,len(file)):
                self.mesh += trimesh.load(file[i])
        self.bounds = self.mesh.bounds
        
    def transform(self, mat):
        self.mesh.apply_transform(mat)
        
    def flatten(self):
        flat_mat = self.mesh.principal_inertia_transform.copy()
        self.transform(flat_mat)
    
    def flip(self, axis):
        self.transform(flip_matrix(axis))
        
    def scale_z(self, z_scale):
        self.transform(scale_matrix(z_scale, 1.0, 1.0))
        
    def rand_rotation(self, rot):
        Z = random.uniform(rot[0], rot[1])
        X = random.uniform(rot[2], rot[3])
        Y = random.uniform(rot[4], rot[5])
        
        rot_mat = rotation_matrix(np.radians(X), np.radians(Y), np.radians(Z))
        self.transform(rot_mat)
    
    def rand_flip(self, axis):
        if random.random() < .5:
            self.transform(flip_matrix(axis))
     
    # Voxelize mesh i.e. turn into binary matrix
    def voxelize(self, pitch, z_pad=0, bounds=None, max_xy_sz=None, dim='3D', sub_sample=False):
        if dim=='2D':
            self.transform(collapse_matrix(0))
        
        if sub_sample:
            pitch /= 2
        vox_grid = self.mesh.voxelized(pitch)
        vox = sparse_to_matrix(vox_grid.sparse_indices).astype(np.float)
        
        shape_V = vox.shape
        
        if sub_sample: # Multiple by 10 to deal with partial volume in voxelized mesh
            vox = 10*blockwise_average_3D(vox, (2,2,2)) 
        
        if bounds is not None: # Keep boundaries
            extents = bounds[1,:] - bounds[0,:]
            sz = np.ceil(extents/pitch).astype(int)+1
            new_vox = np.zeros(sz)
            
            pos = (self.mesh.bounds[0,:] - bounds[0,:])/pitch
            pos = np.round(pos).astype(int)
            
            new_vox[pos[0]:pos[0]+vox.shape[0],
                    pos[1]:pos[1]+vox.shape[1],
                    pos[2]:pos[2]+vox.shape[2]] = vox
            return new_vox
                
        
        if max_xy_sz is not None:
            x_pad = max((max_xy_sz[0]-vox.shape[1])//2, 0)
            y_pad = max((max_xy_sz[1]-vox.shape[2])//2, 0)
            vox = np.pad(vox,((z_pad,z_pad),(x_pad,x_pad+vox.shape[1]%2),(y_pad,y_pad+vox.shape[2]%2)),"constant")
            return vox

# Blur voxelized mesh
def blur_vox(vox, sigmas):
    stack = gaussian_filter(vox, sigmas)
    return stack

# Add 2x2x2 blobs to background of voxelized mesh
def add_blobs(vox, p, thresh, bin_mat=None):
    blobs = bin_mat
    if bin_mat is None:
        blobs = np.random.choice([0.0, 1.0], size=vox.shape, p=[1-p,p])
    blobs[vox>thresh+.05] = 0.0
    blobs = maximum_filter(blobs, [2,2,2])
    vox += blobs*thresh*1.5
    return vox

# Add gaussian noise to voxelized mesh
def add_gauss_noise(vox, power, frames, std_scalar):
    if power is None and frames is None:
        return vox
    X = np.zeros_like(vox)
    
    lam = vox * power
    
    for i in range(frames):
        X_ = np.random.normal(lam, std_scalar*np.sqrt(lam)/frames)
        X += np.clip(X_, 0, 1)
    X /= frames
    
    return X

