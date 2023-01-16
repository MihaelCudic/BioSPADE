import torch
import models.networks as networks
import util.util as util
import torch.nn.functional as F

# Code to train BioSPADE
class BioSPADEModel(torch.nn.Module):
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
        
        self.paired_translation = opt.paired_translation
        
        # set models
        self.models = opt.models_to_train.keys()
        for model in self.models:
            setattr(self, model, None)
            
        self.generate_for_D = False
        self.initialize_networks(opt)
        self.num_up_layers = networks.generator.compute_latent_vector_size(self.opt)
        
        # set loss functions
        if opt.isTrain:
            self.criterionGAN_conv = networks.GANLoss(opt.gan_conv_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionGAN_gram = networks.GANLoss(opt.gan_gram_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionSegment = torch.nn.CrossEntropyLoss(ignore_index=255)
            
            self.criterionFeat = networks.FeatLoss()
            self.criterionStyle = networks.StyleLoss()
            
            if 'VGG_style' in self.opt.losses:
                self.criterionVGG_style = networks.VGGLoss(self.opt.gpu_ids, do_gram=True)
            if 'VGG_feat' in self.opt.losses:
                self.criterionVGG_feat = networks.VGGLoss(self.opt.gpu_ids)

    def forward(self, data, mode):
        
        # Process input (i.e. reshape input, send to gpu, create target, etc.)
        if mode == 'inference':
            data = self.preprocess_input(data, False)
        else:
            data = self.preprocess_input(data, True)

        if self.generate_for_D:
            with torch.no_grad():
                mask = data['mesh_semantics']
                noise = data['noise']
                power = data['power']
                frames = data['frames']
                z_pos = data['z_pos']
                self.fake, self.fake_mu = self.generate_fake(mask, noise, power, frames, z_pos)
                self.fake_mu.requires_grad_()
                self.fake.requires_grad_()
                self.generate_for_D = False
                
        if mode == 'G':
            g_loss, fake, fake_mu = self.compute_G_loss(data) # Compute G loss
            return g_loss, fake, fake_mu
        elif mode == 'Dconv':
            dconv_loss = self.compute_Dconv_loss(self.fake, data) # Compute Dconv loss
            return dconv_loss
        elif mode == 'Dgram':
            dgram_loss = self.compute_Dgram_loss(self.fake, data) # Compute Dgram loss
            return dgram_loss
        elif mode == 'Seg':
            seg_loss, seg = self.compute_Seg_loss(self.fake_mu, data) # Compute Seg loss
            return seg_loss, seg
        elif mode == 'inference':
            with torch.no_grad():
                fake, fake_mu = self.generate_fake(data['mesh_semantics'], 
                                                   data['noise'], 
                                                   data['power'],
                                                   data['frames'], 
                                                   data['z_pos'])
            return fake, fake_mu
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        # Create optimizers for models only if models required training
        optimizer = dict.fromkeys(self.models)
        for model in opt.models_to_train.keys():
            if not opt.models_to_train[model]:
                continue
            
            if model not in optimizer:
                raise(model + 'not a listed model')
                    
            if 'G'==model[0]: # if generator
                G_params = list(getattr(self,'net_'+model).parameters())
                optimizer[model] = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
            elif 'D'==model[0]: # if discriminator
                D_params = list(getattr(self,'net_'+model).parameters())
                optimizer[model] = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
            elif 'S'==model[0]: # if segmentor
                S_params = list(getattr(self,'net_'+model).parameters())
                optimizer[model] = torch.optim.Adam(S_params, lr=D_lr, betas=(beta1, beta2))
                
        return optimizer

    def save(self, epoch):
        # Save models that are being trained
        for model in self.models:
            if not self.opt.models_to_train[model]:
                continue
            util.save_network(getattr(self, 'net_'+model), model, epoch, self.opt)

    def initialize_networks(self, opt):
        # Initialize networks and load pretrained models if specified 
        nets  = []
        for model in self.models:
            if not opt.isTrain and not 'G'==model[0]:
                continue
            load_model = (not opt.isTrain and 'G'==model[0]) or model not in opt.models_to_train
            net = networks.define_model(opt, model)
            net = util.load_network(net, model, opt.which_epoch, opt) if load_model else net
            setattr(self, 'net_'+model, net)
        return

    def preprocess_input(self, data, do_target=False):
        sz = data['mesh_semantics'].shape
        dim_concat = sz[0]*sz[1] # batch size
        z_sz = sz[-3]
        xy_sz = sz[-2:]
        
        self.X_2D = (dim_concat, z_sz, *xy_sz) #(batch size, z size, x size, y size) for input/semantic map
        self.X_3D = (dim_concat, 1, z_sz, *xy_sz) #(batch size, 1, z size, x size, y size) for input/semantic map

        self.Y_2D = (dim_concat, z_sz-self.opt.in_Gslices+1, *xy_sz) #(batch size, 1, z size, x size, y size) for output/stack
        self.Y_3D = (dim_concat, 1, z_sz-self.opt.in_Gslices+1, *xy_sz) #(batch size, 1, z size, x size, y size) for output/stack
        self.GT = (dim_concat, *xy_sz)
        
        if self.use_gpu():
            # Send tensors to GPU and reshape
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
                
            if self.paired_translation: # if paired data is required (only true for control experiment)
                if 'real_slices' in data:
                    data['mesh_slices'] = data['real_slices'] # then ensure correspondence between semantic map input and stack
                data['mesh_semantics'] = data['real_semantics']
            
            if 'noise' not in data: # Add noise to every layer
                noise = []
                for i in range(self.num_up_layers+1):
                    x_sz = xy_sz[0]//(2**i)
                    y_sz = xy_sz[1]//(2**i)

                    noise += [torch.randn([*self.X_2D[:-2], x_sz, y_sz]).cuda()]
                data['noise'] = noise
            
            if do_target: # Create target
                target = torch.zeros(self.GT, dtype=torch.long).cuda()
                
                max_proj = torch.zeros(self.GT)
                if self.opt.in_dim == '3D' and self.opt.in_Dslices>1:
                    max_proj = data['mesh_semantics'][:,self.opt.in_Gslices//2:-self.opt.in_Gslices//2].max(1)[0]
                elif self.opt.in_dim == '3D' and self.opt.in_Dslices==1:
                    max_proj = data['mesh_semantics'][:,self.opt.in_Gslices//2]
                else:
                    max_proj = data['mesh_semantics'][:,0]
                
                target_ignore = F.max_pool2d(max_proj.clone(), 3, 1, 1)
                if self.opt.gt_ignore_neighbour_pxs:
                    target[target_ignore>1e-6] = 255
                target[max_proj>1e-6] = 1
                
                data['mesh_target'] = target
            
            if 'power' in data: # ensure minimum power = 1
                min_val = min(self.opt.powers)
                if min_val == 0:
                    min_val = 1.0
                    data['power'] += 1.0
                data['power'] = data['power'].cuda()/min_val
            if 'frames' in data:
                data['frames'] = data['frames'].cuda()
            if 'z_pos' in data:
                data['z_pos'] = data['z_pos'].cuda()
            else:
                data['z_pos'] = None
                
        return data

    def compute_G_loss(self, data):
        G_losses = {}
        
        mask = data['mesh_semantics']
        noise = data['noise']
        real = data['real_stack']
        target = data['mesh_target']
        power = data['power']
        frames = data['frames']
        z_pos = data['z_pos']
        semantic_map = None
        if self.paired_translation:
            semantic_map = data['real_slices']

        fake, fake_mu = self.generate_fake(mask, noise, power, frames, z_pos)

        if 'Dconv' in  self.models: # Pass fake stack to Dconv
            pred_fake_conv, pred_real_conv = self.conv_discriminate(fake, real, 
                                                                    power=power, frames=frames, 
                                                                    z_pos=z_pos, semantic_map=semantic_map)
        if 'Dgram' in  self.models: # Pass fake stack to Dgram
            pred_fake_gram, pred_real_gram = self.gram_discriminate(fake, real, 
                                                                    power=power, frames=frames, 
                                                                    z_pos=z_pos, semantic_map=semantic_map)
        if 'Seg' in  self.models: # Pass fake stack to Seg
            pred_semantics = self.segment(fake_mu)
            
        if 'GAN_conv' in self.opt.losses and 'Dconv' in  self.models: # Compute discrim. loss from Dconv
            G_losses['GAN_conv'] = self.criterionGAN_conv(pred_fake_conv, True, for_discriminator=False) * \
                                            self.opt.losses['GAN_conv']
        if 'Dconv_feat' in self.opt.losses and 'Dconv' in  self.models: # Compute perceptual loss from Dconv
            G_losses['Dconv_feat'] = self.criterionFeat(pred_fake_conv, pred_real_conv) * \
                                        self.opt.losses['Dconv_feat']
        if 'Dconv_style' in self.opt.losses and 'Dconv' in  self.models: # Compute style loss from Dconv
            G_losses['Dconv_style'] = self.criterionStyle(pred_fake_conv, pred_real_conv) * \
                                        self.opt.losses['Dconv_style']

        if 'GAN_gram' in self.opt.losses and 'Dgram' in  self.models: # Compute discrim. loss from Dstyle
            G_losses['GAN_gram'] = self.criterionGAN_gram(pred_fake_gram, True, for_discriminator=False) * \
                                            self.opt.losses['GAN_gram']
        if 'Dgram_style' in self.opt.losses and 'Dgram' in  self.models: # Compute style loss from Dgram
            G_losses['Dgram_style'] = self.criterionStyle(pred_fake_gram, pred_real_gram) * \
                                        self.opt.losses['Dgram_style']

        if 'Rec' in self.opt.losses and 'Seg' in  self.models: # Compute reconstruction loss from Seg
            G_losses['Rec'] = self.criterionSegment(pred_semantics, target) * \
                                            self.opt.losses['Rec']
       
        if 'VGG_feat' in self.opt.losses: # Compute style loss from VGG
            if self.opt.in_dim == '3D':
                real = real.view(self.Y_3D)
            G_losses['VGG_feat'] = self.criterionVGG_feat(fake, real) \
                * self.opt.losses['VGG_feat']
        if 'VGG_style' in self.opt.losses: # Compute style loss from VGG
            if self.opt.in_dim == '3D':
                real = real.view(self.Y_3D)
            G_losses['VGG_Style'] = self.criterionVGG_style(fake, real) \
                * self.opt.losses['VGG_style']
            
        return G_losses, fake, fake_mu

    def compute_Dconv_loss(self, fake, data):
        Dconv_losses = {}
        power = data['power']
        frames = data['frames']
        z_pos = data['z_pos']
        real = data['real_stack']
        semantic_map = None
        if self.paired_translation:
            semantic_map = data['real_slices']
            
        pred_fake, pred_real = self.conv_discriminate(fake, real, power, frames, z_pos, semantic_map)

        Dconv_losses['Dconv_fake'] = self.criterionGAN_conv(pred_fake, False, for_discriminator=True)
        Dconv_losses['Dconv_real'] = self.criterionGAN_conv(pred_real, True, for_discriminator=True)

        return Dconv_losses
    
    def compute_Dgram_loss(self, fake, data):
        Dgram_losses = {}
        power = data['power']
        frames = data['frames']
        z_pos = data['z_pos']
        real = data['real_stack']
        semantic_map = None
        if self.paired_translation:
            semantic_map = data['real_slices']
        
        pred_fake, pred_real = self.gram_discriminate(fake, real, power, frames, z_pos, semantic_map)

        Dgram_losses['Dgram_fake'] = self.criterionGAN_gram(pred_fake, False, for_discriminator=True)
        Dgram_losses['Dgram_real'] = self.criterionGAN_gram(pred_real, True, for_discriminator=True)

        return Dgram_losses
    
    def compute_Seg_loss(self, fake, data):
        Seg_losses = {}
        target = data['mesh_target']
        
        pred_fake = self.segment(fake)

        Seg_losses['Seg_Loss'] = self.criterionSegment(pred_fake, target)

        return Seg_losses, pred_fake  
    
    def generate_fake(self, mask, noise, power, frames, z_pos):
        # Reshape to 2D
        mask = mask.view(self.X_2D)
        in_Dslices = mask.shape[1]-self.opt.in_Gslices+1
        
        power = power.view(-1,1,1,1)
        frames = frames.view(-1,1,1,1)
        if z_pos is not None:
            z_pos = z_pos.view(-1,1,1,1)
        
        # Successively generate individual 2D slices to create 3D sub stack
        fake = torch.zeros_like(mask[:,:in_Dslices])
        fake_mu = torch.zeros_like(mask[:,:in_Dslices])
        context = torch.zeros_like(mask[:,:self.opt.in_Gslices//2])
        for i in range(in_Dslices):
            sub_mask = mask[:,i:(i+self.opt.in_Gslices)]
            sub_noise = [n[:,i:(i+self.opt.in_Gslices)] for n in noise]
            z_pos_i = z_pos+(i*self.opt.delta_z)
            
            fake_, fake_mu_ = self.net_G.forward(sub_mask, sub_noise, power, frames, z_pos_i)

            fake[:,i,None], fake_mu[:,i,None] = fake_, fake_mu_

        # Reshape to 3D
        fake = fake.view(self.Y_3D)
        fake_mu = fake_mu.view(self.Y_3D)
    
        return fake, fake_mu
    
    def conv_discriminate(self, fake, real, power, frames, z_pos, semantic_map=None):
        
        fake_concat, real_concat = self.reshape_data(self.opt.dim_Dconv, fake, real, power, frames, z_pos, semantic_map) #reshape
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0) # combine real and fake to be fed to Dconv
        discriminator_out = self.net_Dconv(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        
        return pred_fake, pred_real
    
    def gram_discriminate(self, fake, real, power, frames, z_pos, semantic_map=None):
 
        fake_concat, real_concat = self.reshape_data(self.opt.dim_Dgram, fake, real, power, frames, z_pos, semantic_map) #reshape
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0) # combine real and fake to be fed to Dgram
        
        discriminator_out = self.net_Dgram(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    def segment(self, fake):
        fake = fake.view(self.Y_2D)
        pred_mask = self.net_Seg(fake)
        return pred_mask
    
    # Reshape data to either 2D and 3D; concatenate power, frames, z_pos to input
    def reshape_data(self, dim, fake, real, power, frames, z_pos, semantic_map=None):
        
        with torch.no_grad():
            frames = max(self.opt.frames)/frames
        
        if dim == '3D':
            fake = fake.view(self.Y_3D)
            real = real.view(self.Y_3D)
            if semantic_map is not None:
                semantic_map = semantic_map.view(self.Y_3D)
            power = power.view(-1,1,1,1,1) 
            frames = frames.view(-1,1,1,1,1)
            z_pos = z_pos.view(-1,1,1,1,1)
        else:
            fake = fake.view(self.Y_2D)
            real = real.view(self.Y_2D)
            if semantic_map is not None:
                semantic_map = semantic_map.view(self.Y_2D)
            power = power.view(-1,1,1,1) 
            frames = frames.view(-1,1,1,1)
            z_pos = z_pos.view(-1,1,1,1)
        
        
        fake_concat = fake
        real_concat = real
        
        if self.opt.condition_on_power:
            power = torch.ones_like(fake[:,:1]) * power
            fake_concat = torch.cat([fake_concat, power], dim=1)
            real_concat = torch.cat([real_concat, power], dim=1)
        if self.opt.condition_on_frames:
            frames = torch.ones_like(fake[:,:1]) * frames
            fake_concat = torch.cat([fake_concat, frames], dim=1)
            real_concat = torch.cat([real_concat, frames], dim=1)
        if self.opt.condition_on_z:
            z_pos = torch.ones_like(fake[:,:1]) * z_pos
            fake_concat = torch.cat([fake_concat, z_pos], dim=1)
            real_concat = torch.cat([real_concat, z_pos], dim=1)
        if self.paired_translation and semantic_map is not None:
            fake_concat = torch.cat([fake_concat, semantic_map], dim=1)
            real_concat = torch.cat([real_concat, semantic_map], dim=1)
        
        return fake_concat, real_concat


    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
