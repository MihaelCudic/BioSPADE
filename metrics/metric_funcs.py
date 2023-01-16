import numpy as np
import math
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
from numpy.linalg import norm
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix
from cooccurrence2Dand3D.cooccur3D import cooccur3D
from util.util import tensorize_dict
import torch

def blur_all(input):
    out = np.zeros_like(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                out[i,j,k] = gaussian_filter(input[i,j,k],1)
    return out

def MSE(y_pred, y_true, normalize=False, axis=(-3,-2,-1)):
    err = (y_pred - y_true)**2
    mse = np.mean(err, axis=axis)
    if normalize:
        mse /= (np.mean(y_true**2, axis=axis)+1e-6)
    return mse.mean(-1)
        
def JSD(P, Q): # Calculate John-Shannon Divergence
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def mult_JSDs(Ps, Qs):
    jsd = np.zeros((len(Ps),))
    for i in range(len(Ps)):
        jsd[i] = JSD(Ps[i], Qs[i])
    return jsd
 
def calc_single_auto(img, sz=np.array([1,5,5])): # Calculate auto-correlation
    z_dim = img.shape[-3]
    xy_dim = img.shape[-1]
    
    auto = np.zeros((2*sz+1))
    for z in range(-sz[0],sz[0]+1,1):
        for x in range(-sz[1],sz[1]+1,1):
            for y in range(-sz[2],sz[2]+1,1):
                zbeg1 = max(0, z) 
                zend1 = min(z_dim, z_dim+z) 
                zbeg2 = max(0, -z)
                zend2 = min(z_dim, z_dim-z)
                
                xbeg1 = max(0, x) 
                xend1 = min(xy_dim, xy_dim+x) 
                xbeg2 = max(0, -x)
                xend2 = min(xy_dim, xy_dim-x)

                ybeg1 = max(0, y) 
                yend1 = min(xy_dim, xy_dim+y) 
                ybeg2 = max(0, -y)
                yend2 = min(xy_dim, xy_dim-y)
                
                corr = np.mean(img[zbeg1:zend1,xbeg1:xend1,ybeg1:yend1]*img[zbeg2:zend2,xbeg2:xend2,ybeg2:yend2])
                
                auto[z+sz[0], x+sz[1], y+sz[2]] = corr
                
    return auto

def calc_auto(imgs, sz=np.array([1,5,5])):
    n_styles = imgs.shape[0]
    n_imgs = imgs.shape[1]
    
    auto = np.zeros([n_styles,*(2*sz+1)])
    for style_id in range(n_styles):
        for i in range(n_imgs):
            #auto[style_id] += calc_single_auto(imgs[style_id,i], sz)/n_imgs
            img_ = imgs[style_id,i]
            auto_i = signal.correlate(img_, img_[sz[0]:-sz[0],sz[1]:-sz[1],sz[2]:-sz[2]], mode='valid')
            auto[style_id] += auto_i/n_imgs
    return auto

# Calculate Local Binary Pattern
def calc_LBP(imgs, mask=None, radius=3, n_points=None, n_points_mult=8):
    if mask is None:
        mask = np.ones_like(imgs[0])
    
    if n_points is None:
        n_points = n_points_mult*radius
        
    n_styles = imgs.shape[0]
    n_imgs = imgs.shape[1]
    z_slices = imgs.shape[2]
    
    n_bins = 256
    hist = np.zeros((n_styles,n_bins))
    for style_id in range(n_styles):
        for i in range(n_imgs):
            for z in range(z_slices):
                lbp = local_binary_pattern(imgs[style_id,i,z], n_points, radius)
                hist_,_ = np.histogram(lbp[mask[i,z]>.5], bins=n_bins, range=(0, n_bins))
                hist[style_id] += hist_
            
    return hist/hist.sum(1,keepdims=True)

# calculate Gray-Level Co-Occurrence Matrix
def calc_GLCM(imgs, mask=None, levels=256, dist_ls=[1,3], angle_ls=None):
    if mask is None:
        mask = np.ones_like(imgs[0])
    
    if angle_ls is None:
        angle_ls = np.linspace(0,math.pi,5)[:-1]
        
    n_styles = imgs.shape[0]
    n_imgs = imgs.shape[1]
    z_slices = imgs.shape[2]
    
    imgs = (imgs*(levels-.001)).astype(np.uint16)
    imgs += 1
    imgs[:,mask<.5] = 0
    glcm = np.zeros((n_styles,levels+1,levels+1,len(dist_ls),len(angle_ls)))

    for style_id in range(n_styles):
        for i in range(n_imgs):
            for z in range(z_slices):
                glcm[style_id] += greycomatrix(imgs[style_id,i,z],
                                               distances=dist_ls, angles=angle_ls, 
                                               levels=levels+1, symmetric=True, 
                                               normed=False)
    glcm = glcm[:,1:,1:,:,:]
    glcm = glcm/np.sum(glcm, axis=(1,2), keepdims=True)
    return glcm

# Calculate 3D Co-Occurrence Matrixß
def calc_COOC(imgs, mask=None, i_bins=50, i_range=(0,1), g_bins=4, a_bins=7, dists=(1,3), angle_ls=None):
    if angle_ls is None:
        np.linspace(0,math.pi,5)[:-1]
    
    n_styles = imgs.shape[0]
    n_imgs = imgs.shape[1]
    
    cooc = np.zeros((n_styles, len(dists), a_bins, g_bins, g_bins, i_bins, i_bins))

    for style_id in range(n_styles):
        for i in range(n_imgs):
            if mask is None:
                mask_ = None
            else:
                mask_ = mask[i]
                if mask_.sum() == 0:
                    continue
            cooc[style_id] += cooccur3D(imgs[style_id,i], i_bins=i_bins, i_range=i_range, 
                                        g_bins=g_bins, a_bins=a_bins, dists=dists, econ=True, mask=mask_)
        
    cooc = cooc/np.sum(cooc, axis=(-6,-5,-4,-3,-2,-1), keepdims=True)
    return cooc
  
# Calculate PSNR
def calc_PSNR(imgs, axis=(1,2,3,4,5)):
    err = (imgs-imgs.mean(1,keepdims=True))**2
    err = np.mean(err, axis=axis)
    return 20*np.log10(1/np.sqrt(err))
        
# Compare histograms
def compare_hist(P_imgs, Q_imgs, bins=100, range_=(0,1)):
    vals = np.zeros((len(P_imgs),))
    
    for i in range(len(P_imgs)):
        P,_ = np.histogram(P_imgs[i], bins=bins, range=range_, density=True)
        Q,_ = np.histogram(Q_imgs[i], bins=bins, range=range_, density=True)
        
        P = P / norm(P, ord=1)
        Q = Q / norm(Q, ord=1)
        
        vals[i] = JSD(P, Q)
    return vals

# Calculate Intersection of Union Across batch
def IoU_Loss_Batch( pred, target, ignore_index=255):
    pred_argmax = torch.argmax(pred, dim=1)

    intersection = ((target*pred_argmax)==1).sum()

    union = pred_argmax.clone()
    union[target==1] = 1
    union[target==ignore_index] = 0
    union = union.sum()

    return intersection, union

def SEG(model, images, target, powers, frames, z_pos, return_images=False, batch_sz=None):
    n_styles = images.shape[0]
    n_imgs = images.shape[1]
    
    if batch_sz is None:
        batch_sz = n_imgs
    
    iou = np.zeros((n_styles,))
    img_target = np.zeros((n_styles, n_imgs, *images.shape[-2:]))
    img = np.zeros((n_styles, n_imgs, *images.shape[-3:]))
    seg = np.zeros((n_styles, n_imgs, 2, *images.shape[-2:]))
    
    for style_id in range(n_styles):
        intersection = 0
        union = 0
        for i in range(n_imgs//batch_sz):
            data = {'real_stack': images[style_id, i*batch_sz:(i+1)*batch_sz],
                    'real_slices': target[:,i*batch_sz:(i+1)*batch_sz],
                    'power': n_imgs*[powers[style_id]],
                    'frames': n_imgs*[frames[style_id]],
                    'z_pos': z_pos[i*batch_sz:(i+1)*batch_sz],
                    'noise': [0]}
            data = tensorize_dict(data)
            
            seg_loss, seg_iou, img_, img_target_, seg_ = model(data,'test')

            intersection_, union_ = IoU_Loss_Batch(seg_, img_target_)
            intersection += intersection_
            union += union_
        
            img[style_id:, i*batch_sz:(i+1)*batch_sz] = img_.cpu().numpy()
            img_target[style_id:, i*batch_sz:(i+1)*batch_sz] = img_target_.cpu().numpy()
            seg[style_id:, i*batch_sz:(i+1)*batch_sz] = seg_.cpu().numpy()
        iou[style_id] = intersection/union #seg_iou['test_IoU']
        
    if return_images:
        return img, img_target, seg
    return iou