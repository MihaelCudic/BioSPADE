"""
Functions sourced from https://github.com/NVlabs/SPADE and adjusted accordingly for bioSPADE
"""

import sys
import data
import torch

from collections import OrderedDict

from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.biospade_trainer import BioSPADETrainer

def train_network(opt):
    # load the dataset
    dataloader,_ = data.create_dataloader(opt, 'all')

    # create trainer for our model
    trainer = BioSPADETrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    # save options used for expeirment
    opt.save_options()
    
    data_i = None
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # Training
            if i % opt.D_steps_per_G == 0:
                trainer.run_G_one_step(data_i)
            trainer.run_D_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                fake, fake_mu = trainer.get_latest_generated()
                seg = None
                if 'Seg' in opt.models_to_train:
                    seg = trainer.get_latest_segment()

                stack_mid = opt.in_Dslices//2
                slice_mid = (opt.in_Dslices+opt.in_Gslices-1)//2

                real = data_i['real_stack'].view(-1,opt.in_Dslices,*opt.crop_xy_sz)
                sem = data_i['mesh_semantics'].view(-1,opt.in_Gslices+opt.in_Dslices-1,*opt.crop_xy_sz)
                sem = sem.max(1)[0][:,None]

                visuals = [('real', real[:,stack_mid:stack_mid+1]),
                           ('sem', sem),
                           ('fake_mu', fake_mu[:,:,stack_mid]),
                           ('fake', fake[:,:,stack_mid])]
                if seg is not None:
                    visuals += [('seg', seg[:,1:2])]
                    
                visuals = OrderedDict(visuals)
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
           epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
    print('Training was successfully finished.')