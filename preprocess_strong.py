import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path_pre', type=str, default='preprocessed_data', help='path to save results')
    parser.add_argument('--N', type=int, default=5, help='the total number of edge')


    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    N = args.N

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path_pre):
        os.mkdir(args.save_path_pre)

    dir_data = os.path.join(args.save_path_pre, 'random_{}'.format(str(N)))
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)


    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    # if args.dataset == "imagenette":
        # img_net_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
        # for i, elem in enumerate(labels_all):
            # labels_all[i]=img_net_classes.index(elem)
            
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    # data_split = torch.split(images_all, len(images_all)//N)
    # label_split = torch.split(labels_all, len(images_all)//N)
    
    num_data = []
    for i in range(num_classes):
        num_data.extend([len(indices_class[i])//N])
    
    for i in range(N):
        which_index = []
        for j in range(num_classes):
            which_index.extend(indices_class[j][i*num_data[j]:(i+1)*num_data[j]])
        random.shuffle(which_index)
        torch.save(images_all[which_index], os.path.join(dir_data, 'data_{}.pt'.format(str(i))))
        torch.save(labels_all[which_index], os.path.join(dir_data, 'label_{}.pt'.format(str(i))))
    

    # for i,data in enumerate(zip(data_split, label_split)):
        # torch.save(data[0], os.path.join(dir_data, 'data_{}.pt'.format(str(i))))
        # torch.save(data[1], os.path.join(dir_data, 'label_{}.pt'.format(str(i))))
    

if __name__ == '__main__':
    main()


