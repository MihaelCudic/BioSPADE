import torch
import models.networks as networks
from models.networks.loss import IoU_Loss
import util.util as util
import torch.nn.functional as F

# Code to train basic segmentor
class SegmentModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        
        self.net_G, self.net_Seg = self.initialize_networks(opt)
        if self.net_G is not None:
            self.net_G.eval()
            self.num_up_layers = self.net_G.compute_latent_vector_size(self.opt)
        
        self.loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.iou_criterion = IoU_Loss(ignore_index=255)

    def forward(self, data, mode, input=None):
        
        # Process input (i.e. reshape input, send to gpu, create target, etc.)
        data = self.preprocess_input(data, mode)
        if mode != 'test':
            with torch.no_grad():
                mask = data['mesh_semantics']
                noise = data['noise']
                power = data['power']
                frames = data['frames']
                z_pos = data['z_pos']
                
                self.input = input
                if input is None:
                    self.input, _ = self.generate_fake(mask, noise, power, frames, z_pos)
                self.input.requires_grad_()
                self.target = data['target']
        else:
            self.input = data['real_stack']
            self.target = data['target']
            
        if mode == 'train':
            self.net_Seg.train()
            seg_loss, seg_iou, seg = self.compute_Seg_loss(self.input, self.target, data, mode)
            return seg_loss, seg_iou, self.input, self.target, seg
        elif mode=='val' or mode=='test':
            with torch.no_grad():
                self.net_Seg.eval()
                seg_loss, seg_iou, seg = self.compute_Seg_loss(self.input, self.target, data, mode)
                return seg_loss, seg_iou, self.input, self.target, seg
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        S_params = list(self.net_Seg.parameters())
        optimizer_S = torch.optim.Adam(S_params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        return optimizer_S

    def save(self, epoch):
        util.save_network(self.net_Seg, 'Seg(inst'+str(self.opt.seg_instance)+')', epoch, self.opt)

    def initialize_networks(self, opt):
        net_G = None
        if opt.isTrain:
            net_G = networks.define_model(opt, 'G')
            net_G = util.load_network(net_G, 'G', opt.which_epoch, opt) if opt.isTrain else None
        
        net_Seg = networks.define_model(opt, 'Seg')
        if not opt.isTrain:
            net_Seg = util.load_network(net_Seg, 'Seg(inst'+str(self.opt.seg_instance)+')', opt.which_epoch, opt)
            
        return net_G, net_Seg

    def preprocess_input(self, data, mode, do_target=False):
        sz = data['real_stack'].shape
        dim_concat = sz[0]*sz[1]
        z_sz = sz[-3]
        xy_sz = sz[-2:]
        
        self.X_2D = (dim_concat, z_sz+self.opt.in_Gslices-1, *xy_sz)
        self.X_3D = (dim_concat, 1, z_sz+self.opt.in_Gslices-1, *xy_sz)

        self.Y_2D = (dim_concat, z_sz, *xy_sz)
        self.Y_3D = (dim_concat, 1, z_sz, *xy_sz)
        self.GT = (dim_concat, *xy_sz)
        
        if self.use_gpu():
            if 'real_stack' in data:
                data['real_stack'] = data['real_stack'].cuda().view(self.Y_2D)
            if 'real_slices' in data:
                data['real_slices'] = data['real_slices'].cuda().view(self.Y_2D)
            if 'real_semantics' in data:
                data['real_semantics'] = data['real_semantics'].cuda().view(self.X_2D)
            if 'mesh_slices' in data:
                data['mesh_slices'] = data['mesh_slices'].cuda().view(self.Y_2D)
            if 'mesh_semantics' in data:
                data['mesh_semantics'] = data['mesh_semantics'].cuda().view(self.X_2D)
            
            if 'noise' not in data and self.net_G is None:
                data['noise'] = None
            elif 'noise' not in data:
                noise = []
                for i in range(self.num_up_layers+1):
                    x_sz = xy_sz[0]//(2**i)
                    y_sz = xy_sz[1]//(2**i)

                    noise += [torch.randn([*self.X_2D[:-2], x_sz, y_sz]).cuda()]
                data['noise'] = noise
            
            # Create target
            target_str = 'mesh_slices'
            if mode ==  'test':
                target_str = 'real_slices'
            
            target_input = data[target_str]
            target = torch.zeros(self.GT, dtype=torch.long).cuda()

            max_proj = torch.zeros(self.GT)
            if self.opt.in_dim == '3D' and self.opt.in_Dslices>1:
                max_proj = target_input.max(1)[0]
            elif self.opt.in_dim == '3D' and self.opt.in_Dslices==1:
                max_proj = target_input[:,self.opt.in_Gslices//2]
            else:
                max_proj = target_input[:,0]
                
            target_ignore = F.max_pool2d(max_proj.clone(), 3, 1, 1)
            if self.opt.gt_ignore_neighbour_pxs:
                target[target_ignore>1e-6] = 255
            target[max_proj>1e-6] = 1
            data['target'] = target
            
            if 'power' in data:
                min_val = min(self.opt.powers)
                if min_val == 0:
                    min_val = 1.0
                    data['power'] += 1.0
                data['power'] = data['power'].cuda()/min_val
            if 'frames' in data:
                data['frames'] = data['frames'].cuda()
            if 'z_pos' in data:
                data['z_pos'] = data['z_pos'].cuda()
        return data

    def compute_Seg_loss(self, input, target, data, mode):
        Seg_losses = {}
        Seg_iou = {}
        
        pred = self.segment(input, data['power'], data['frames'], data['z_pos'])

        Seg_losses[mode+'_Loss'] = self.loss_criterion(pred, target)
        Seg_iou[mode+'_IoU'] = self.iou_criterion(pred, target)

        return Seg_losses, Seg_iou, pred  
    
    def generate_fake(self, mask, noise, power, frames, z_pos):
        mask = mask.view(self.X_2D)
        in_Dslices = mask.shape[1]-self.opt.in_Gslices+1
        
        power = power.view(-1,1,1,1) 
        frames = frames.view(-1,1,1,1)
        z_pos = z_pos.view(-1,1,1,1)
        
        fake = torch.zeros_like(mask[:,:in_Dslices])
        fake_mu = torch.zeros_like(mask[:,:in_Dslices])
        for i in range(in_Dslices):
            sub_mask = mask[:,i:(i+self.opt.in_Gslices)]
            sub_noise = [n[:,i:(i+self.opt.in_Gslices)] for n in noise]
            z_pos_i = z_pos+(i*self.opt.delta_z)

            fake_, fake_mu_ = self.net_G.forward(sub_mask, sub_noise, power, frames, z_pos_i)

            fake[:,i,None], fake_mu[:,i,None] = fake_, fake_mu_

        fake = fake.view(self.Y_3D)
        fake_mu = fake_mu.view(self.Y_3D)
        
        return fake, fake_mu
    
    def segment(self, input, power, frames, z_pos):
        input = input.view(self.Y_2D)
        
        power = power.view(-1,1,1,1) 
        frames = frames.view(-1,1,1,1)
        z_pos = z_pos.view(-1,1,1,1)
        
        input_concat = input
        if self.opt.condition_on_power:
            power = torch.ones_like(input[:,:1]) * power
            input_concat = torch.cat([input_concat, power], dim=1)
        if self.opt.condition_on_frames:
            frames = torch.ones_like(input[:,:1]) * frames
            input_concat = torch.cat([input_concat, frames], dim=1)
        if self.opt.condition_on_z:
            z_pos = torch.ones_like(input[:,:1]) * z_pos
            input_concat = torch.cat([input_concat, z_pos], dim=1)
        
        pred = self.net_Seg(input_concat)
        return pred

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
