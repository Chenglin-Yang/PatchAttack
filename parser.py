import argparse
import os
import numpy as np
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='PatchAttack --- Chenglin Yang')

parser.add_argument('--gpu', default='0', type=str, 
                    help = 'specify which gpus to use (default: ''0'' )')
parser.add_argument('--torch-cuda', default=0, type=int, 
                    help='cuda device to use for torch model (default: 0)')
parser.add_argument('--data', default='ImageNet', type=str, 
                    choices=['ImageNet', 'ImageNet-1000'],
                    help='data name (default: ImageNet)')
parser.add_argument('--tdict-dir', default='TextureDict_ImageNet', type=str, 
                    help='dir to the texture dictionary (default: TextureDict_ImageNet)')
parser.add_argument('--t-labels', default=[], type=int, nargs='+', 
                    help='specify the labels to build texture dictionary or to attack images (default: [])')
parser.add_argument('--t-labels-range', default=[0, 1000], type=int, nargs=2, 
                    help='if t_labels is [], then np.arange(*t_labels_range).tolist() is used (default:0, 1000)')
parser.add_argument('--arch', default='ResNet', type=str, 
                    help='the architecture of the neural network (default: ResNet)')
parser.add_argument('--depth', default=50, type=int, 
                    help='the depth of the neural network (default: 50)')
# main_build-dict
parser.add_argument('--t-data', default='ImageNet', type=str, 
                    help = 'dataset to build texture dict upon (default: ImageNet)')
parser.add_argument('--dict', default='Texture', type=str, choices=['Texture', 'AdvPatch'], 
                    help='type of dictionary to build (default: Texture)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU organization

# cfg
cfg = edict()
# convert args to cfg
for key, value in args.__dict__.items():
    cfg[key] = value

if cfg.t_labels == []:
    cfg.t_labels = np.arange(*cfg.t_labels_range).tolist()

# ImageNet path
cfg.ImageNet_train_dir = '/CUSTOM_PATH_TO_TRAIN/'  # Please provide the path to ImageNet train folder
cfg.ImageNet_val_dir = '/CUSTOM_PATH_TO_VAL/'  # Please provide the path to ImageNet val folder
