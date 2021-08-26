import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

config = yaml.safe_load(open('config.yaml'))

def plot_from_batch_generator(batch_gen):
    '''Plot 3x3 images from batch'''
    data_batch, label_batch = next(iter(batch_gen))
    grid_size = (3, 3)
    f, axarr = plt.subplots(*grid_size)
    f.set_size_inches(15,10)
    class_names = batch_gen.dataset.classes
    for i in range(grid_size[0] * grid_size[1]):
        
        # read images from batch to numpy.ndarray and change axes order [H, W, C] -> [H, W, C]
        batch_image_ndarray = np.transpose(data_batch[i].numpy(), [1, 2, 0])
        
        # inverse normalization for image data values back to [0,1] and clipping the values for correct pyplot.imshow()
        src = np.clip(config['image_std'] * batch_image_ndarray + config['image_mean'], 0, 1)
        
        # display batch samples with labels
        sample_title = 'Label = %d (%s)' % (label_batch[i], class_names[label_batch[i]])
        axarr[i // grid_size[0], i % grid_size[0]].imshow(src)
        axarr[i // grid_size[0], i % grid_size[0]].set_title(sample_title)

def load_data(path_to_data, transformer):
    '''Load data, transform it with transformer 
    and return torch's train and validation batches'''
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config['DATA_PATH'], 'train'), transform=transformer)
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config['DATA_PATH'], 'val'), transform=transformer)
    train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                        batch_size=config['BATCH_SIZE'],
                                        shuffle=True,
                                        num_workers=config['NUM_WORKERS'])
    val_batch_gen = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=config['BATCH_SIZE'],
                                        num_workers=config['NUM_WORKERS'])
    return train_batch_gen, val_batch_gen