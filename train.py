import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
from tqdm import tqdm
import os.path as osp
#from networks.gcnet import Res_Deeplab
from dataset.datasets import CSDataSet
#import matplotlib.pyplot as plt
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from utils.utils import decode_labels, inv_preprocess, decode_predictions
from utils.criterion import CriterionCrossEntropy, CriterionOhemCrossEntropy, CriterionDSN, CriterionOhemDSN
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.utils import fromfile
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
'''
BATCH_SIZE = 8
DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/train.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '769,769'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 60000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = 'snapshots/'
WEIGHT_DECAY = 0.0005
'''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the model with large input size.")
    parser.add_argument('--config', help='train config file path')

    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000,
                        help="choose the samples with correct probability underthe threshold.")
    return parser.parse_args()

args = get_arguments()
cfg=fromfile(args.config)

if cfg.model.type == 'basenet':
    from networks.basenet import Res_Deeplab

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(cfg.train_cfg.learning_rate, i_iter, cfg.train_cfg.num_steps, cfg.train_cfg.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def main():
    """Create the model and start the training."""
    writer = SummaryWriter(cfg.train_cfg.snapshot_dir)
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.data_dir is not None:
        cfg.data_cfg.data_dir = args.data_dir
    if args.restore_from is not None:
        cfg.train_cfg.restore_from = args.restore_from
    if args.start_iters is not None:
        cfg.train_cfg.start_iters = args.start_iters

    h, w = map(int, cfg.data_cfg.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.
    deeplab = Res_Deeplab(cfg.model,cfg.data_cfg.num_classes)
    print(deeplab)

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 
    
    deeplab.load_state_dict(new_params)

    model = DataParallelModel(deeplab)
    model.train()
    model.float()
    # model.apply(set_bn_momentum)
    model.cuda()    

    if args.ohem:
        criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
    else:
        criterion = CriterionDSN()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(cfg.train_cfg.snapshot_dir):
        os.makedirs(cfg.train_cfg.snapshot_dir)


    trainloader = data.DataLoader(CSDataSet(cfg.data_cfg.data_dir, cfg.data_cfg.data_list, max_iters=cfg.train_cfg.num_steps*cfg.train_cfg.batch_size,
                                            crop_size=input_size,scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                                  batch_size=cfg.train_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': cfg.train_cfg.learning_rate }],
                lr=cfg.train_cfg.learning_rate, momentum=cfg.train_cfg.momentum,weight_decay=cfg.train_cfg.weight_decay)
    optimizer.zero_grad()

    for i_iter, batch in enumerate(trainloader):
        i_iter += cfg.train_cfg.start_iters
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        if torch_ver == "0.3":
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        preds = model(images)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        # if i_iter % 5000 == 0:
        #     images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
        #     labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
        #     if isinstance(preds, list):
        #         preds = preds[0]
        #     preds_colors = decode_predictions(preds, args.save_num_images, args.num_classes)
        #     for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
        #         writer.add_image('Images/'+str(index), img, i_iter)
        #         writer.add_image('Labels/'+str(index), lab, i_iter)
        #         writer.add_image('preds/'+str(index), preds_colors[index], i_iter)

        print('iter = {} of {} completed, loss = {}'.format(i_iter, cfg.train_cfg.num_steps, loss.data.cpu().numpy()))

        if i_iter >= cfg.train_cfg.num_steps-1:
            print('save model ...')
            torch.save(deeplab.state_dict(),osp.join(cfg.train_cfg.snapshot_dir, 'CS_scenes_'+str(cfg.train_cfg.num_steps)+'.pth'))
            break

        if i_iter % cfg.train_cfg.save_pred_every == 0 and i_iter > cfg.train_cfg.save_from:
            print('taking snapshot ...')
            torch.save(deeplab.state_dict(),osp.join(cfg.train_cfg.snapshot_dir, 'CS_scenes_'+str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
