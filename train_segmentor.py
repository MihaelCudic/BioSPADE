"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import data
import torch

from collections import OrderedDict

from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.segment_trainer import SegmentTrainer

def train_network(opt):
    # load the dataset
    trainloader,_ = data.create_dataloader(opt, 'train')
    valloader = trainloader
    if hasattr(opt, 'seg_train_is_valid'):
        if not opt.seg_train_is_valid:
            valloader,_ = data.create_dataloader(opt, 'valid')

    # create trainer for our model
    trainer = SegmentTrainer(opt)

    # create tool for counting iterations
    train_iter_counter = IterationCounter(opt, len(trainloader))
    val_iter_counter = IterationCounter(opt, len(valloader))

    # create tool for visualization
    visualizer = Visualizer(opt)
    
    data_i = None
    min_loss = 1e8
    moniter = 0
    for epoch in train_iter_counter.training_epochs():
        trainer.reset_losses()
        train_iter_counter.record_epoch_start(epoch)
        for i, train_data in enumerate(trainloader, start=train_iter_counter.epoch_iter):
            train_iter_counter.record_one_iteration()
            trainer.run_Seg_train_one_step(train_data)

        val_iter_counter.record_epoch_start(epoch)
        for i, val_data in enumerate(valloader, start=val_iter_counter.epoch_iter):
            val_iter_counter.record_one_iteration()
            trainer.run_Seg_val_one_step(val_data)

        losses = trainer.get_latest_losses()
        visualizer.print_current_errors(epoch, train_iter_counter.epoch_iter,
                                        losses, train_iter_counter.time_per_iter)

        if (epoch%5) == 0:
            fake_input, fake_target, fake_pred = trainer.get_latest_data()
            real_input, real_target, real_pred = trainer.run_Seg_test_one_step(val_data)
            
            stack_mid = opt.in_Dslices//2
            real = val_data['real_stack'].view(-1,opt.in_Dslices,*opt.crop_xy_sz)
            n_samps = 16
            
            visuals = [('fake_input', fake_input[:n_samps,:,stack_mid]),
                       ('fake_pred', fake_pred[:n_samps,1:]),
                       ('fake_target', fake_target[:n_samps,None]),
                       ('real_input', real_input[:n_samps,stack_mid:stack_mid+1]),
                       ('real_pred', real_pred[:n_samps,1:]),
                       ('real_target', real_target[:n_samps,None]),]          
            '''
            
            visuals = [('fake_input', fake_input[:n_samps,stack_mid:stack_mid+1]),
                       ('fake_pred', fake_pred[:n_samps,1:]),
                       ('fake_target', fake_target[:n_samps,None]),
                       ('real_input', real_input[:n_samps,stack_mid:stack_mid+1]),
                       ('real_pred', real_pred[:n_samps,1:]),
                       ('real_target', real_target[:n_samps,None]),]    
            '''
            
            visuals = OrderedDict(visuals)
            visualizer.display_current_results(visuals, epoch, train_iter_counter.total_steps_so_far)

        if losses['train_Loss']<min_loss:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, train_iter_counter.total_steps_so_far))
            trainer.save('latest')
            min_loss = losses['train_Loss']
            moniter = 0
        elif moniter<opt.early_stopping:
            moniter += 1
        else:
            break

        trainer.update_learning_rate(epoch)
        train_iter_counter.record_epoch_end()
    print('Training was successfully finished.')