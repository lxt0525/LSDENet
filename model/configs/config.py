import sys
# sys.path.append('../lightNDF')
import numpy as np
import os
import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    """
    Parse input arguments
    """
    parser.add_argument('--epochs', dest='num_epochs',
                        help='number of epochs to train',
                        default=1000, type=int)
    parser.add_argument('--train_batch_size', dest='train_batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=1, type=int)
    parser.add_argument('--modeldir', dest='modeldir',
                        help='model output directory',
                        default='/log', type=str)

    parser.add_argument('--model_name', default='REDE_V1', type=str)
    parser.add_argument('--img_channels', default=1, type=int)
    parser.add_argument('--dataset', default='ACDC', type=str)
    parser.add_argument('--half', default=True, type=bool)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--loss_dl_weight', default=0.01, type=float)

    parser.add_argument("--train_dir", type=str, default='data/piccolo/train', help='input train data directory')
    parser.add_argument("--val_dir", type=str, default='data/piccolo/validation', help='input val data directory')
    parser.add_argument("--test_dir", type=str, default='data/piccolo/test', help='input val video data directory')

    parser.add_argument('--optimizer', dest='optimizer',
                        help='training optimizer',
                        default="Adam", type=str)
    parser.add_argument('--scheduler', type=str, default="StepLR")
    parser.add_argument('--gpu', type=str, default='all', help='GPU to use [default: all].')

    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=20, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.5, type=float)


    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    return cfg
