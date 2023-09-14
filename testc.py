from __future__ import print_function
import argparse
import os
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging

from data_utils.data_util import PointcloudScaleAndTranslate
from data_utils.ModelNetDataLoader import ModelNetDataLoaderC

from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2 import PointNet2ClsMsg
from models.dgcnn import DGCNN
from models.pointcnn import PointCNNCls

from utils import progress_bar, log_row
import sys
sys.path.append("./emd/")
import emd_module as emd

MAP = ['uniform',
        'gaussian',
       'background',
       'impulse',
    #    'scale',
       'upsampling',
       'shear',
       'rotation',
       'cutout',
       'density',
       'density_inc',
       'distortion',
       'distortion_rbf',
       'distortion_rbf_inv',
       'occlusion',
       'lidar',
       'original'
]

# def test(model, test_loader, criterion):
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader, 0):
        points, label = data
        points, label = points.to(device), label.to(device)

        if args.model == 'rscnn_kcutmix':
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                              2).contiguous()  # (B, N, 3)
        points = points.transpose(2, 1)  # to be shape batch_size*3*N

        pred, trans_feat = model(points)

        # loss = criterion(pred, label.long())

        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(label.data).cpu().sum()
        total += label.size(0)
        # progress_bar(j, len(test_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    #  % (loss.item() / (j + 1), 100. * correct.item() / total, correct, total))

    return 100. * correct.item() / total

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointnet_kcutmix', help='choose model type')
    parser.add_argument('--data', type=str, default='modelnetc', help='choose data set')
    parser.add_argument('--seed', type=int, default=0, help='manual random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--resume', type=str, default='/', help='resume path')
    parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")
    parser.add_argument('--lambda_ft', type=float, default=0.001, help="lambda for feature transform")
    parser.add_argument('--augment', type=int, default=1, help='data argment to increase robustness')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--note', type=str, default='', help='notation of the experiment')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--beta', default=1, type=float, help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')
    args = parser.parse_args()
    args.feature_transform, args.augment = bool(args.feature_transform), bool(args.augment)
    ### Set random seed
    args.seed = args.seed if args.seed > 0 else random.randint(1, 10000)

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    ###
    # gen_test_log(args)
    # logname = ('logs_test/%s_%s_%s.csv' % (args.data, args.model, args.name))

    if not os.path.isdir('logs_test'):
        os.mkdir('logs_test')
    logname = ('%s_%s_%s' % (args.data, args.model, args.name))
    experiment_dir = 'logs_test/'

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #
    file_handler = logging.FileHandler('%s/%s.txt' % (experiment_dir, logname))

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # dataset path
    DATA_PATH = '/home/user_tp/workspace/data/modelnet40_normal_resampled/'

    ########################################
    ## Intiate model
    ########################################
    
    num_classes = 40

    ##model
    if args.model == 'dgcnn_kcutmix':
        model = DGCNN(num_classes)
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        if args.model == 'pointnet_kcutmix':
            model = PointNetCls(num_classes, args.feature_transform)
            model = model.to(device)
        elif args.model == 'pointnet2_kcutmix':
            model = PointNet2ClsMsg(num_classes)
            model = model.to(device)
            model = nn.DataParallel(model)

        elif args.model == 'rscnn_kcutmix':
            from models.rscnn import RSCNN
            import models.rscnn_utils.pointnet2_utils as pointnet2_utils

            model = RSCNN(num_classes)
            model = model.to(device)
            model = nn.DataParallel(model)

    checkpoint = torch.load('./checkpoints/modelnet40_%s_train/best.pth' % args.model)
    # args = checkpoint['args']
    model.load_state_dict(checkpoint['model_state_dict'])

    # ##load dataset
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='test',
    #                                     normal_channel=args.normal)
    # test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
    #                                           num_workers=4, drop_last=False)

    # if args.model == 'dgcnn_kcutmix':
    #     criterion = cal_loss
    # else:
    #     criterion = F.cross_entropy  # nn.CrossEntropyLoss()
    
    label_path = "/home/user_tp/workspace/data/ModelNet40-C/label.npy"

    for cor in MAP:
        if cor in ['original']:
            data_path = "/home/user_tp/workspace/data/ModelNet40-C/data_" + cor + ".npy"
            test_dataset = ModelNetDataLoaderC(data_root=data_path, label_root=label_path)
            # print("test_dataset:",test_dataset)
            testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
            with torch.no_grad():
                acc = test(model, testDataLoader)
              # instance_acc, class_acc, weights = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
                log_string('data_%s: Test Accuracy: %f' % (cor, acc))
                # progress_bar(j, len(test_loader), 'data_%s | Test Accuracy: %.3f' % (cor, acc))

        #     continue
        else:
            for sev in [1,2,3,4,5]:
                data_path = "/home/user_tp/workspace/data/ModelNet40-C/data_" + cor + "_" + str(sev) + ".npy"
           
                test_dataset = ModelNetDataLoaderC(data_root=data_path, label_root=label_path)
                # print("test_dataset:",test_dataset)
                testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
                with torch.no_grad():
                    acc = test(model, testDataLoader)
                  # instance_acc, class_acc, weights = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
                    log_string('data_%s_%s: Test Accuracy: %f' % (cor, str(sev), acc))
        ###
    # test_loss, test_acc = test(model, test_loader, criterion)

    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    main(args)
    