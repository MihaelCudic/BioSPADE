import torch
import torch.nn as nn
import torch.nn.functional as F

# Compute Gram Matrix
def GramMatrix(input):
    a = input.size(0)
    b = input.size(1)
    other = input[0,0].numel()
    
    x = input[:,:].view(a,b,other)
    G = torch.matmul(x,x.permute(0,2,1)).div(b * other)
    return G

# Gram Matrix layer
class GramBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        G = GramMatrix(input)
        return G

# Flattened Gram Matrix layer
class FlattenGram(nn.Module):
    def forward(self, input):
        inds = torch.tril_indices(input.shape[-2], input.shape[-1]) # because gram matrix is symmetric, only take unique values
        vec = input[:,inds[0], inds[1]]
        return vec   
    
# Spatial StdDev Layer
class spatial_stddev_layer(nn.Module):
    def __init__(self, collapse=False):
        super(spatial_stddev_layer, self).__init__()
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
        self.collapse = collapse
        
    def forward(self, x):
        dim_axis = [-1,-2]
        if x.dim() ==  5:
            dim_axis = [-1,-2, -3]
        vals = self.adjusted_std(x, dim=dim_axis, keepdim=True) # compute std across every channel
        if self.collapse:
            vals = vals.mean(dim=1, keepdim=True)
            vals = vals.expand(*x[:,:1].shape)
        else: 
            vals = vals.expand(*x.shape)
            
        return torch.cat([x, vals], 1)