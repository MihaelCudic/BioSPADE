"""
Functions sourced from https://github.com/NVlabs/SPADE and adjusted for bioSPADE
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.biospade_model import BioSPADEModel

class BioSPADETrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.models = opt.models_to_train.keys()
        self.biospade_model = BioSPADEModel(opt)
        
        if len(opt.gpu_ids) > 0:
            self.biospade_model = DataParallelWithCallback(self.biospade_model,
                                                          device_ids=opt.gpu_ids)
            self.biospade_model_on_one_gpu = self.biospade_model.module
        else:
            self.biospade_model_on_one_gpu = self.biospade_model

        if opt.isTrain:
            optimizer = self.biospade_model_on_one_gpu.create_optimizers(opt)
            for model in self.models:
                setattr(self, 'optimizer_'+model, optimizer[model])
            self.old_lr = opt.lr
            
    def run_G_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, fake, fake_mu = self.biospade_model(data, 'G')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.G_losses = g_losses
        self.fake = fake
        self.fake_mu = fake_mu
        
    def run_D_one_step(self, data):
        self.biospade_model_on_one_gpu.generate_for_D = True
        if 'Dconv' in self.models:
            self.run_Dconv_one_step(data)
        if 'Dgram' in self.models:
            self.run_Dgram_one_step(data)
        if 'Seg' in self.models:
            self.run_Seg_one_step(data)
        
    def run_Dconv_one_step(self, data):
        self.optimizer_Dconv.zero_grad()
        dconv_losses = self.biospade_model(data, mode='Dconv')
        dconv_loss = sum(dconv_losses.values()).mean()
        dconv_loss.backward()
        self.optimizer_Dconv.step()
        self.Dconv_losses = dconv_losses
        
    def run_Dgram_one_step(self, data):
        self.optimizer_Dgram.zero_grad()
        dgram_losses = self.biospade_model(data, mode='Dgram')
        dgram_loss = sum(dgram_losses.values()).mean()
        dgram_loss.backward()
        self.optimizer_Dgram.step()
        self.Dgram_losses = dgram_losses
        
    def run_Seg_one_step(self, data):
        self.optimizer_Seg.zero_grad()
        seg_losses, seg = self.biospade_model(data, mode='Seg')
        seg_loss = sum(seg_losses.values()).mean()
        seg_loss.backward()
        self.optimizer_Seg.step()
        self.Seg_losses = seg_losses
        self.seg = seg

    def get_latest_losses(self):
        losses = {}
        if 'G' in self.opt.models_to_train:
            losses.update(**self.G_losses)
        if 'Dconv' in self.opt.models_to_train:
            losses.update(**self.Dconv_losses)
        if 'Dgram' in self.opt.models_to_train:
            losses.update(**self.Dgram_losses)
        if 'Seg' in self.opt.models_to_train:
            losses.update(**self.Seg_losses)
        return losses
            
    def get_latest_generated(self):
        return self.fake, self.fake_mu

    def get_latest_segment(self):
        return self.seg

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.biospade_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for model in self.opt.models_to_train:
                if model[0]=='D':
                    optimizer = getattr(self, 'optimizer_'+model)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr_D
                if model[0]=='G':
                    optimizer = getattr(self, 'optimizer_'+model)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
