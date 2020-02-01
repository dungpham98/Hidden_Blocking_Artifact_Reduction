import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter

def cropImg(size,img_tensor):
    imgs=[]
    batch = int(img_tensor.shape[0])
    channel = int(img_tensor.shape[1])
    h = int(img_tensor.shape[2])
    w = int(img_tensor.shape[3])
    n = int(h/size)
    i = 0
    while(i*size < h):
        j = 0
        while(j*size < w):
            i_n =int(i*size)
            j_n = int(j*size)
            img = img_tensor[0:batch,0:channel,i_n:(i_n+size),j_n:(j_n+size)]
            imgs.append(img)
            #torchvision.utils.save_image(img,"cropped"+str(i_n)+str(j_n)+".jpg")
            j = j + 1 
        i = i + 1
    return imgs

def concatImgs(imgs):
    img_len = len(imgs)
    i = 0
    img_cat =[]
    while(i < 16):
        img_cat.append(torch.cat([imgs[0+i],imgs[1+i],imgs[2+i],imgs[3+i]],dim=3))
        i = i + 4
    img = torch.cat([img_cat[0],img_cat[1],img_cat[2],img_cat[3]],2)
    return img

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
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        #train
        for image, _ in train_data:
            image = image.to(device)
            """
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, _ = model.train_on_batch([image, message])
            print(losses)
            """
            #crop imgs
            imgs = cropImg(32,image)
            #iterate img
            bitwise_arr=[]
            main_losses = None
            for img in imgs:
                img=img.to(device)
                message = torch.Tensor(np.random.choice([0, 1], (img.shape[0], hidden_config.message_length))).to(device)
                losses, _ = model.train_on_batch([img, message])
                main_losses = losses
                for name, loss in losses.items():
                    if(name == 'bitwise-error  '):
                        bitwise_arr.append(loss)

            bitwise_arr = np.array(bitwise_arr)
            bitwise_avg = np.average(bitwise_arr)

            for name, loss in main_losses.items():
                if(name == 'bitwise-error  '):
                    training_losses[name].update(bitwise_avg)
                else:
                    training_losses[name].update(loss)    

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

        #val
        for image, _ in val_data:
            image = image.to(device)
            #crop imgs
            imgs = cropImg(32,image)
            #iterate img
            bitwise_arr=[]
            main_losses = None
            encoded_imgs=[]
            for img in imgs:
                img=img.to(device)
                message = torch.Tensor(np.random.choice([0, 1], (img.shape[0], hidden_config.message_length))).to(device)
                losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([img, message])
                encoded_imgs.append(encoded_images)
                main_losses = losses
                for name, loss in losses.items():
                    if(name == 'bitwise-error  '):
                        bitwise_arr.append(loss)

            bitwise_arr = np.array(bitwise_arr)
            bitwise_avg = np.average(bitwise_arr)

            for name, loss in main_losses.items():
                if(name == 'bitwise-error  '):
                    validation_losses[name].update(bitwise_avg)
                else:
                    validation_losses[name].update(loss)
            #concat image
            encoded_images = concatImgs(encoded_imgs)

            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False

        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
