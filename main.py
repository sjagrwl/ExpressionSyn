#Options
base_path = '/home/sahaj/Desktop/UG-3/DIP/group5/'
config = {
    'c_dim' : 8,
    'c2_dim': 6,
    'celeba_crop_size': 178,
    'fera_crop_size': 256,
    'image_size': 128,
    'g_conv_dim': 64,
    'd_conv_dim': 64,
    'g_repeat_num': 6,
    'd_repeat_num': 6,
    'lambda_cls': 1,
    'lambda_rec': 10,
    'lambda_gp': 10,
    
    # Training configuration.
    'dataset' : 'FERA',
    'batch_size': 16,
    'num_iters': 200000,
    'num_iters_decay': 100000,
    'g_lr': 0.0001,
    'd_lr': 0.0001,
    'n_critic': 5,
    'beta1': 0.5,
    'beta2': 0.999,
    'resume_iters': 120000,
    'selected_attrs' : ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],

    'test_iters': 120000,

    # Miscellaneous.
    'num_workers': 16,
    'mode': 'test', 
    'use_tensorboard': False, 

    # Directories.
    'celeba_image_dir': base_path + 'data/CelebA_nocrop/images/',
    'attr_path': base_path + 'data/list_attr_celeba.txt',
    'fera_image_dir': base_path + 'data/FERA/test/',
    'log_dir': base_path + 'logs/',
    'model_save_dir': base_path + 'models/',
    'sample_dir': base_path + 'samples/',
    'result_dir': base_path + 'results/',

    # Step size.
    'log_step': 10,
    'sample_step': 1000,
    'model_save_step' : 10000,
    'lr_update_step' : 1000,
}

import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

cudnn.benchmark = True

# Create directories if not exist.
os.makedirs(config['log_dir'], exist_ok = True)
os.makedirs(config['model_save_dir'], exist_ok = True)
os.makedirs(config['sample_dir'], exist_ok = True)
os.makedirs(config['result_dir'], exist_ok = True)

# Data loader.
celeba_loader = None
rafd_loader = None

if config['dataset'] in ['FERA']:
    fera_loader = get_loader(config['fera_image_dir'], None, None,
                             config['fera_crop_size'], config['image_size'], config['batch_size'],
                             'FERA', config['mode'], config['num_workers'])

# Solver for training and testing StarGAN.
solver = Solver(fera_loader, config)

if config['mode'] == 'train':
    if config['dataset'] in ['FERA']:
        solver.train()
elif config['mode'] == 'test':
    if config['dataset'] in ['FERA']:
        solver.test()