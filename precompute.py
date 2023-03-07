from __future__ import print_function

import argparse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import models.densenet as dn
from tqdm import tqdm
import pickle
import time

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', '-d', default='CIFAR-100', type=str, help='dataset')
parser.add_argument('--method', default='taylor', type=str, help='odin mahalanobis')
parser.add_argument('--model_arch', default='resnet50', type=str, help='model architecture')


args = parser.parse_args()

def precompute(args):
    if args.dataset == 'CIFAR-100':
        num_classes = 100
        model = dn.DenseNet3(100, num_classes, normalizer=None, p_w=None, p_a=None, LU = True) # LUNCH
        checkpoint = torch.load("./checkpoints/CIFAR-100/densenet/checkpoint_100.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        featdim = 342
    elif args.dataset == 'CIFAR-10':
        num_classes = 10
        model = dn.DenseNet3(100, num_classes, normalizer=None, p_w=None, p_a=None, LU = True) # LUNCH
        checkpoint = torch.load("./checkpoints/CIFAR-10/densenet/checkpoint_100.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        featdim = 342
    elif args.dataset == 'imagenet':
        num_classes = 1000
        from models.resnet import resnet50
        model = resnet50(num_classes=num_classes, pretrained=True,LU=True)
        featdim = 2048

    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    test_batch_size = 64
    net = net.to(device)
    
    if args.dataset in {'CIFAR-10', 'CIFAR-100'}:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        dataset = {
            'CIFAR-10': torchvision.datasets.CIFAR10,
            'CIFAR-100': torchvision.datasets.CIFAR100,
        }
        trainset = dataset[args.dataset](root='./data', train=True, download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=test_batch_size, shuffle=False, num_workers=4)

        id_train_size = 50000

        cache_name = f"cache/{args.dataset}_train_densenet_{args.method}_in.npy"
        if not os.path.exists(cache_name):
            shap_log = np.zeros((id_train_size, featdim))
            score_log = np.zeros((id_train_size, num_classes))
            label_log = np.zeros(id_train_size)

            
            batch_size = 1
            
            net.eval()
            trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=4)
            for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
                
                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx

                if args.method in {'taylor'}:
                    first_order_taylor_scores, outputs = net._compute_taylor_scores(inputs, targets)
                    shap_log[start_ind, :] = first_order_taylor_scores[0].squeeze().cpu().detach().numpy()
                label_log[start_ind] = targets.data.cpu().numpy()
                score_log[start_ind] = outputs.data.cpu().numpy()
        
            np.save(cache_name, (shap_log.T, score_log.T, label_log))
            print("dataset : ", args.dataset)
            print("method : ", args.method)
            print("iteration done")
        else:
            shap_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            shap_log, score_log = shap_log.T, score_log.T
            
            shap_matrix_mean = np.zeros((featdim,num_classes))
            
            for class_num in range(num_classes):
                mask = np.array(label_log==class_num)
                masked_shap = mask[:,np.newaxis] * shap_log
                shap_matrix_mean[:,class_num] = masked_shap.sum(0) / mask.sum()
                 
            np.save(f"cache/{args.dataset}_densenet_{args.method}_mean_class.npy", shap_matrix_mean)
            
            print("dataset : ", args.dataset)
            print("method : ", args.method)
            print("precompute done")
    else:
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    ############################################################################################################
        cache_name_shap = f"cache/{args.dataset}_{args.model_arch}_{args.method}.npy"
        cache_name_score = f"cache/{args.dataset}_{args.model_arch}_{args.method}_score.npy"
        cache_name_label = f"cache/{args.dataset}_{args.model_arch}_{args.method}_label.npy"
        if not os.path.exists(cache_name_shap):
            batch_size = 1
            traindata = torchvision.datasets.ImageFolder('./datasets/ILSVRC-2012/train', transform_test_largescale)
            trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            id_train_size = len(traindata)
            
            shap_log = np.zeros((id_train_size, featdim))
            score_log = np.zeros((id_train_size, num_classes))
            label_log = np.zeros(id_train_size)
            
            net.eval()
            for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
                
                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx
                
                if args.method in {'taylor', 'taylor_abs'}:
                    first_order_taylor_scores, outputs = net._compute_taylor_scores(inputs, targets)
                    shap_log[start_ind, :] = first_order_taylor_scores[0].squeeze().cpu().detach().numpy()
                label_log[start_ind] = targets.data.cpu().numpy()
                score_log[start_ind] = outputs.data.cpu().numpy()
            
            with open(cache_name_shap, 'wb') as f:
                pickle.dump(shap_log.T, f, protocol=pickle.DEFAULT_PROTOCOL)
            with open(cache_name_score, 'wb') as f:
                pickle.dump(score_log.T, f, protocol=pickle.DEFAULT_PROTOCOL)
            with open(cache_name_label, 'wb') as f:
                pickle.dump(label_log.T, f, protocol=pickle.DEFAULT_PROTOCOL)
            print("dataset : ", args.dataset, "method : ", args.method, "iteration done")
        else:
            cache_name_shap = f"cache/{args.dataset}_{args.model_arch}_{args.method}.npy"
            cache_name_score = f"cache/{args.dataset}_{args.model_arch}_{args.method}_score.npy"
            cache_name_label = f"cache/{args.dataset}_{args.model_arch}_{args.method}_label.npy"
            with open(cache_name_shap, 'rb') as f:
                shap_log = pickle.load(f)
            with open(cache_name_score, 'rb') as f:
                score_log = pickle.load(f)
            with open(cache_name_label, 'rb') as f:
                label_log = pickle.load(f)
            shap_log,  label_log = shap_log.T, label_log.T
            
            shap_matrix_mean = np.zeros((featdim,num_classes))

            for class_num in tqdm(range(num_classes)):
                mask = np.where(label_log==class_num)
                masked_shap = shap_log[mask[0][:]]
                num_sample = len(mask[0][:])
                shap_matrix_mean[:,class_num] = masked_shap.sum(0) / num_sample
            np.save(f"cache/{args.dataset}_{args.model_arch}_{args.method}_mean_class.npy", shap_matrix_mean)
    print("done")

if __name__ == '__main__':
    
########## CIFAR precompute ##########
    args.model_arch = 'densenet'
    for dataset in ['CIFAR-10', 'CIFAR-100']:
        args.method = 'taylor'
        args.dataset = dataset
        precompute(args)
        
    # precompute twice for class-wise info        
    for dataset in ['CIFAR-10', 'CIFAR-100']:
        args.method = 'taylor'
        args.dataset = dataset
        precompute(args)

########## ImageNet precompute ##########          
    args.model_arch = 'resnet50'
    args.method = 'taylor'
    args.dataset = 'imagenet'
    precompute(args)
    
    # precompute twice for class-wise info
    args.model_arch = 'resnet50'
    args.method = 'taylor'
    args.dataset = 'imagenet'
    precompute(args)
