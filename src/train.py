import os
from os import listdir
from os.path import isfile, join
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict
import cv2
from torchvision.utils import save_image
from options import *
from model.hidden import Hidden
from average_meter import AverageMeter

def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    block_size = hidden_config.block_size
    block_number = int(hidden_config.H/hidden_config.block_size)
    val_folder = train_options.validation_folder
    loss_type = train_options.loss_mode
    m_length = hidden_config.message_length
    alpha = train_options.alpha
    img_names = listdir(val_folder+"/valid_class")
    img_names.sort()
    out_folder = train_options.output_folder
    default = train_options.default
    beta = train_options.beta
    crop_width = int(beta*block_size)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)
    icount = 0
    plot_block = []

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        #train
        for image, _ in train_data:
            image = image.to(device)
            #crop imgs into blocks
            imgs, modified_imgs, entropies = cropImg(block_size,image,crop_width,alpha)
            bitwise_arr=[]
            main_losses = None
            encoded_imgs = []
            batch = 0 
            for img, modified_img, entropy in zip(imgs,modified_imgs, entropies):
                img=img.to(device)
                modified_img = modified_img.to(device)
                entropy = entropy.to(device)
                
                message = torch.Tensor(np.random.choice([0, 1], (img.shape[0], m_length))).to(device)
                losses, (encoded_images, noised_images, decoded_messages) = \
                    model.train_on_batch([img, message, modified_img, entropy,loss_type])
                encoded_imgs.append(encoded_images)
                batch = encoded_images.shape[0]
                #get loss in the last block
                if main_losses is None:
                    main_losses = losses
                    for k in losses:
                        main_losses[k] = losses[k]/len(imgs)
                else:
                    for k in main_losses:
                        main_losses[k] += losses[k]/len(imgs)

            #blocking effect loss calculation
            blocking_loss = blocking_value(encoded_imgs,batch,block_size,block_number)
          
            #update bitwise training loss
            for name, loss in main_losses.items():
                if(default == False  and name == 'blocking_effect'):
                    training_losses[name].update(blocking_loss)
                else:
                    training_losses[name].update(loss) 
            #statistic
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))

        #validation
        ep_blocking = 0
        ep_total = 0
     
        for image, _ in val_data:
            image = image.to(device)
            #crop imgs
            imgs, modified_imgs, entropies = cropImg(block_size,image,crop_width,alpha)
            bitwise_arr=[]
            main_losses = None
            encoded_imgs = []
            batch = 0
          
            for img, modified_img, entropy in zip(imgs,modified_imgs, entropies):
                img=img.to(device)
                modified_img = modified_img.to(device)
                entropy = entropy.to(device)
                
                message = torch.Tensor(np.random.choice([0, 1], (img.shape[0], m_length))).to(device)
                losses, (encoded_images, noised_images, decoded_messages) = \
                    model.train_on_batch([img, message, modified_img, entropy,loss_type])
                encoded_imgs.append(encoded_images)
                batch = encoded_images.shape[0]
                #get loss in the last block
                if main_losses is None:
                    main_losses = losses
                    for k in losses:
                        main_losses[k] = losses[k]/len(imgs)
                else:
                    for k in main_losses:
                        main_losses[k] += losses[k]/len(imgs)
                
            #blocking value for plotting
            blocking_loss = blocking_value(encoded_imgs,batch,block_size,block_number)
            ep_blocking = ep_blocking+ blocking_loss
            ep_total = ep_total+1

            for name, loss in main_losses.items():
                if(default == False  and name == 'blocking_effect'):
                    validation_losses[name].update(blocking_loss)
                else:
                    validation_losses[name].update(loss) 
            #concat image
            encoded_images = concatImgs(encoded_imgs,block_number)
            #save_image(encoded_images,"enc_img"+str(epoch)+".png")
            #save_image(image,"original_img"+str(epoch)+".png")
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False
            #save validation in the last epoch
            if(epoch == train_options.number_of_epochs):
                if(train_options.ats):
                    for i in range(0,batch):
                        image = encoded_images[i].cpu()
                        image = (image + 1) / 2
                        f_dst = out_folder+"/"+img_names[icount]
                        save_image(image,f_dst)
                        icount = icount+1
        #append block effect for plotting
        plot_block.append(ep_blocking/ep_total)
    
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
