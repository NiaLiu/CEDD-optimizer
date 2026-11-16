import os
import datetime
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

# from model_ema import ModelEmaV2
# perform DC on random splited data
# so far no label learning

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='how many dataset to generate at each edge')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path_pre', type=str, default='preprocessed_data', help='path to save results')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    # parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--edge_num', type=int, default=0, help='which edge will start to distill, input the edge id')
    parser.add_argument('--N', type=int, default=5, help='the total number of edge')
    parser.add_argument('--not_eval', action='store_true', help="don't start by evaluate the first couple iterations")
    parser.add_argument('--ol', type=int, default=30, help='which edge will start to distill, input the edge id')


    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.outer_loop, args.inner_loop = args.ipc, args.ol
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    N = args.N

    def memory_report():
        print('Current memory status:Reserved memory {},Allocated memory {}.'.format(torch.cuda.memory_reserved(device=args.device),torch.cuda.memory_allocated(device=args.device)))

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    save_dir = os.path.join(args.save_path, 'edge_'+str(args.edge_num))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    dir_data = os.path.join(args.save_path_pre, 'random_{}'.format(str(N)))
    if not os.path.exists(dir_data):
        print('Please prepare data beforehand')
        exit(0)

    if os.path.exists(os.path.join(save_dir, 'res_%s_%s_%s_%dipc_0_it%d.pt'%(args.method, args.dataset, args.model, args.ipc, args.Iteration))):
        exit(0)
    if args.not_eval:
        eval_it_pool = [] if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration]
    else:
        eval_it_pool = np.arange(0, args.Iteration+1, 1).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    # Load all data
    ''' organize the real dataset '''
    images_all = torch.load(os.path.join(dir_data, 'data_{}.pt'.format(str(args.edge_num))), map_location=args.device)
    labels_all = torch.load(os.path.join(dir_data, 'label_{}.pt'.format(str(args.edge_num))), map_location=args.device)
    indices_class = [[] for c in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            if len(indices_class[c])>= n:
                return images_all[idx_shuffle]
            else:
                holder = images_all[idx_shuffle]
                for repeat in range(1,n//len(indices_class[c])+1):
                    idx_shuffle_0 = np.random.permutation(indices_class[c])[:(n-repeat*len(indices_class[c]))]
                    holder = torch.concatenate((holder,images_all[idx_shuffle_0]))
                return holder

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.rand(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        for i in range(num_classes):
            image_syn[i*args.ipc:(i+1)*args.ipc] = get_images(i, args.ipc)
        
        image_syn.requires_grad_().to(device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.float, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        # label_syn = torch.randn(size=(num_classes*args.ipc, num_classes), dtype=torch.float)
        # label_syn.requires_grad_().to(device=args.device)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        # optimizer_label = torch.optim.SGD([label_syn], lr=0.001, momentum=0.5)
        optimizer_img.zero_grad()
        # ema = ModelEmaV2([image_syn], decay=args.ema_decay)
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())
        print(datetime.datetime.now())
        for it in range(1,args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        # with torch.no_grad():
                            # ema_image_syn = ema.module[0]
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.to('cpu').detach()).to(args.device), copy.deepcopy(label_syn.to('cpu').detach()).to(args.device) # avoid any unaware modification  
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        del net_eval, image_syn_eval, label_syn_eval
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    del acc_test, acc_train

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                        del accs

                ''' visualize and save '''
                save_name = os.path.join(save_dir, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.


            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            total_params = sum(p.numel() for p in net.parameters())
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real).to(args.device)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1])).to(args.device)
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    # lab_syn = label_syn[c*args.ipc:(c+1)*args.ipc]

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)
                    
                    print_detail = False
                    if print_detail == True:
                        with torch.no_grad():
                            img_real_act = img_real.nelement()
                            img_syn_act = img_syn.nelement()
                            output_syn_act = output_syn.nelement()
                            output_real_act = output_real.nelement()
                            gw_real_act = sum([p.nelement() for p in gw_real])
                            gw_syn_act = sum([p.nelement() for p in gw_syn])
                            # gw_update_act = sum([p.nelement() for p in gw_update])
                            loss_act = loss.nelement()
                            print('Now on GPU, \nreal image size:{}\nsynthetic image size:{}\noutput of synthetic dataset size:{}\noutput of real dataset size:{}\ngradient of real dataset size:{}\ngradient of synthetic dataset size:{}\nActivation_size on loss:{}\nloss to syndata activation":{}'.format(img_real_act,img_syn_act,output_syn_act,output_real_act,gw_real_act,gw_syn_act,loss_act))
                            exit(0)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                
                loss_avg += loss.item()
                # ema.update([image_syn])


                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)


            loss_avg /= (num_classes*args.outer_loop)

            del loss, img_real, img_syn, net
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            if it == args.Iteration:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(save_dir, 'res_%s_%s_%s_%dipc_%d_it%d.pt'%(args.method, args.dataset, args.model, args.ipc, exp, it)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


