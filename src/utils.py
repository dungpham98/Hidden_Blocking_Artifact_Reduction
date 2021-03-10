import numpy as np
import os
import re
import csv
import time
import pickle
import logging
import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(options_file_name) -> (TrainingOptions, HiDDenConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test_gray': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
                                               drop_last=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size, drop_last=True,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)

def calculate_image_entropy(imgs):
    BINS = 256
    ret = []
    for img in imgs:
        marg = np.histogramdd(np.ravel(img), bins=BINS)[0] / img.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))

        ret.append(entropy / 8)

    return np.array(ret)


def cropImg(size, img_tensor, WIDTH, ALPHA):
    imgs = []
    modified_imgs = []
    entropies = []

    batch = int(img_tensor.shape[0])
    channel = int(img_tensor.shape[1])
    h = int(img_tensor.shape[2])
    w = int(img_tensor.shape[3])
    n = int(h / size)
    i = 0

    img_tensor1 = img_tensor.cpu().detach().numpy()
    img_entropy = calculate_image_entropy(img_tensor1)
    while (i * size < h):
        j = 0
        while (j * size < w):
            i_n = int(i * size)
            j_n = int(j * size)

            img = img_tensor[0:batch, 0:channel, i_n:(i_n + size), j_n:(j_n + size)]
            modified_img = img_tensor1[0:batch, 0:channel, i_n:(i_n + size), j_n:(j_n + size)]

            if j_n + size + WIDTH <= w:
                modified_img[0:batch, 0:channel, :, size - WIDTH:size] *= (1 - ALPHA)
                modified_img[0:batch, 0:channel, :, size - WIDTH:size] += \
                    ALPHA * img_tensor1[0:batch, 0:channel, i_n:i_n + size, j_n + size:j_n + size + WIDTH]

            if i_n + size + WIDTH <= h:
                modified_img[0:batch, 0:channel, size - WIDTH:size, :] *= (1 - ALPHA)
                modified_img[0:batch, 0:channel, size - WIDTH:size, :] += \
                    ALPHA * img_tensor1[0:batch, 0:channel, i_n + size:i_n + size + WIDTH, j_n:j_n + size]

            imgs.append(img)
            # print(np.sum(img.cpu().detach().numpy() - modified_img))
            modified_imgs.append(torch.tensor(modified_img))
            entropies.append(torch.tensor(img_entropy[0:len(modified_img)]))

            # torchvision.utils.save_image(img,"cropped"+str(i_n)+str(j_n)+".jpg")
            j = j + 1
        i = i + 1

    return imgs, modified_imgs, entropies


def concatImgs(imgs, block_number):
    img_len = len(imgs)
    i = 0
    img_cat = []
    block_num = block_number * block_number
    while (i < block_num):
        img_col = torch.cat([imgs[0 + i], imgs[1 + i]], 3)
        for j in range(2, block_number):
            img_col = torch.cat([img_col, imgs[j + i]], 3)
        img_cat.append(img_col)
        i = i + block_number
    img = torch.cat([img_cat[0], img_cat[1]], 2)
    # print(img.shape)
    for i in range(2, len(img_cat)):
        img = torch.cat([img, img_cat[i]], 2)

    # img = torch.cat([img_cat[0],img_cat[1],img_cat[2],img_cat[3]],2)
    return img


def blocking_value(encoded_imgs, batch, block_size, block_number):
    # blocking effect value
    Total = 0
    Vcount = 0
    Hcount = 0
    V_average = 0
    H_average = 0
    for idx in range(0, batch):
        V_average = 0
        H_average = 0
        for i in range(0, len(encoded_imgs) - 1):
            if ((i + 1) % block_number != 0):
                img = encoded_imgs[i][idx][0].cpu().detach().numpy()
                img_next = encoded_imgs[i + 1][idx][0].cpu().detach().numpy()
                for j in range(0, block_size):
                    distinct = np.abs(img[j][block_size - 1] - img_next[j][0])
                    V_average = V_average + distinct
                    Total = Total + 1
                    if (distinct > 0.25):
                        Vcount = Vcount + 1

        for i in range(0, len(encoded_imgs) - block_number):
            img = encoded_imgs[i][idx][0].cpu().detach().numpy()
            img_next = encoded_imgs[i + block_number][idx][0].cpu().detach().numpy()
            for j in range(0, block_size):
                distinct = np.abs(img[block_size - 1][j] - img_next[0][j])
                H_average = H_average + distinct
                Total = Total + 1
                if (distinct > 0.25):
                    Hcount = Hcount + 1

    blocking_loss = (Vcount + Hcount) / (Total)
    return blocking_loss
