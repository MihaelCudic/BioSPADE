from models.networks.sync_batchnorm import DataParallelWithCallback
from models.segment_model import SegmentModel

class SegmentTrainer():

    def __init__(self, opt):
        self.opt = opt
        self.segment_model = SegmentModel(opt)
        if len(opt.gpu_ids) > 0:
            self.segment_model = DataParallelWithCallback(self.segment_model,
                                                          device_ids=opt.gpu_ids)
            self.segment_model_on_one_gpu = self.segment_model.module
        else:
            self.segment_model_on_one_gpu = self.segment_model

        if opt.isTrain:
            self.optimizer_S = self.segment_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            
        self.reset_losses()
         
    def reset_losses(self):
        self.train_counter = 0
        self.val_counter = 0
        self.test_counter = 0
        
        self.train_loss = {'train_Loss': 0}
        self.val_loss = {'val_Loss': 0}
        self.test_loss = {'test_Loss': 0}
        
        self.train_iou = {'train_IoU': 0}
        self.val_iou = {'val_IoU': 0}
        self.test_iou = {'test_IoU': 0}
            
    def run_Seg_train_one_step(self, data, input=None):
        self.train_counter += 1
        
        self.optimizer_S.zero_grad()
        s_losses, s_iou, fake, target, pred = self.segment_model(data, 'train', input)
        s_loss = sum(s_losses.values()).mean()
        s_loss.backward()
        self.optimizer_S.step()
        
        self.train_loss['train_Loss'] += s_losses['train_Loss']
        self.train_iou['train_IoU'] += s_iou['train_IoU']
        self.fake = fake
        self.target = target
        self.pred = pred

    def run_Seg_val_one_step(self, data, input=None):
        self.val_counter += 1
        
        s_losses, s_iou, fake, target, pred = self.segment_model(data, 'val', input)
        s_loss = sum(s_losses.values()).mean()
        
        self.s_val_losses = s_losses
        self.val_loss['val_Loss'] += s_losses['val_Loss']
        self.val_iou['val_IoU'] += s_iou['val_IoU']
        self.fake = fake
        self.target = target
        self.pred = pred
        
    def run_Seg_test_one_step(self, data, input=None):
        self.test_counter += 1
        s_losses, s_iou, fake, target, pred = self.segment_model(data, 'test', input)
        return fake, target, pred
    
    def get_latest_losses(self):
        losses = {}
        
        self.train_loss['train_Loss'] /= self.train_counter
        self.val_loss['val_Loss'] /= self.val_counter
        
        self.train_iou['train_IoU'] /= self.train_counter
        self.val_iou['val_IoU'] /= self.val_counter
        
        losses.update(**self.train_loss, **self.train_iou, **self.val_loss, **self.val_iou)
        return losses
            
    def get_latest_data(self):
        return self.fake, self.target, self.pred
    
    def save(self, epoch):
        self.segment_model_on_one_gpu.save(epoch)

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

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
