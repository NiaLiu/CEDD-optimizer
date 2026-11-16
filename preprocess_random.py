import os
import time
import copy
import argparse
import numpy as np
import torch
import random
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

# We use Dirichlet distribution 
# To redesign distribution, please just simply edit ******Distribution******

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path_pre', type=str, default='preprocessed_data', help='path to save results')
    parser.add_argument('--N', type=int, default=10, help='the total number of edge')
    parser.add_argument('--alpha', type=float, default=1.0, help='Dirichlet distribution')


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

    num_data = []

    # ****** distribution ******
    alpha = args.alpha
    min_sample = 5
    for i in range(num_classes):
        random.shuffle(indices_class[i])

        proportions = np.random.dirichlet([alpha] * N)

        counts = np.random.multinomial(len(indices_class[i]) - min_sample*N, proportions)

        num_data.append(counts+min_sample)
    # ***************************
    print(num_data)

    for i in range(N):
        which_index = []

        for j in range(num_classes):
            start = sum(num_data[j][:i])
            end   = sum(num_data[j][:i+1])

            which_index.extend(indices_class[j][start:end])
        torch.save(images_all[which_index], os.path.join(dir_data, 'data_{}.pt'.format(str(i))))
        torch.save(labels_all[which_index], os.path.join(dir_data, 'label_{}.pt'.format(str(i))))

        random.shuffle(which_index)

    

if __name__ == '__main__':
    main()


