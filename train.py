import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import parser_set

import torch.nn as nn
import torch.nn.functional as F
import torch

# nohup python cyclegan.py > log.txt &
os.environ['CUDA_VISIBLE_DEVICES']='5'

test_name = 'test5_gpu5'

    
def PCA_IMG_loader():
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12)),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        PCA_IMG_OBJ_online_Dataset('/home/yezipeng/talkingheadData/caricature_rewrite/npy_save/partAllpca200_icp.npy', '/home/yezipeng/talkingheadData/caricature_rewrite/CelebA-HQ-img256',
        '/home/yezipeng/talkingheadData/caricature_rewrite/celebA_head_wo_pose_norm_npy',
        img_transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
    )
    return dataloader



            
def pca_cyclegan_main():
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    print(opt)

    pca_img_loader = PCA_IMG_loader()
    pca_dim = 200
    
    
    G_IMG2PCA = ResNet50_IMG2PCA()
    D_PCA = Discriminator_PCA_for_BCE()
    
    optimizer_G = torch.optim.Adam(G_IMG2PCA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D_PCA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    criterion_GAN = torch.nn.BCELoss()
    data_root = '/home/yezipeng/talkingheadData/caricature_rewrite/npy_save/'
    #criterion_IDD = IDDloss_fullobj(data_root+'pca200.model', data_root+'warehouse_0.obj')
    #criterion_IDD = IDDloss_fullobj_front(data_root+'pca200_icp.model', data_root+'warehouse_0.obj', data_root+'front_part_v.txt')
    criterion_IDD = IDDloss_fullobj_front_seperate_mean(data_root+'pca200_icp.model', data_root+'warehouse_0.obj', data_root+'front_part_v.txt')
    #return

    if cuda:
        G_IMG2PCA.cuda()
        D_PCA.cuda()
        criterion_GAN.cuda()
        criterion_IDD.cuda()
        
    img_save_folder = 'images/%s/'%(test_name)
    pth_save_folder = 'models/%s/'%(test_name)
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    if not os.path.exists(pth_save_folder):
        os.makedirs(pth_save_folder)
        
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, pca_img in enumerate(pca_img_loader):
            print(epoch, i)
            real_img = Variable(pca_img["img"].type(Tensor))
            real_pca = Variable(pca_img["pca"].type(Tensor))
            img_obj = Variable(pca_img["obj"].type(Tensor))
            valid = Variable(Tensor(np.ones((real_pca.size(0), 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_pca.size(0), 1))), requires_grad=False)

            
            #G loss
            optimizer_G.zero_grad()
            fake_pca = G_IMG2PCA(real_img)
            loss_GAN = criterion_GAN(D_PCA(fake_pca), valid)
            print(loss_GAN)
            
            
            idd_loss_angle, idd_loss_len = criterion_IDD(fake_pca, img_obj)
            print(idd_loss_angle, idd_loss_len)
            
            
            loss_G = loss_GAN + 2*idd_loss_angle + 1*idd_loss_len
            loss_G.backward()
            optimizer_G.step()
            

            #D loss
            optimizer_D.zero_grad()
            loss_real = criterion_GAN(D_PCA(real_pca), valid)
            loss_fake = criterion_GAN(D_PCA(fake_pca.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            print(loss_D_A)
            loss_D_A.backward()
            optimizer_D.step()

            
            
            if (i == 0) and (epoch%10 == 0):
                torch.save(G_IMG2PCA, 'models/%s/epoch%d.pth'%(test_name, epoch))
                save_image(real_img.data, "images/%s/real%d.png" % (test_name, epoch), nrow=8, normalize=True)
                #np.save("images/%s/fake_pca%d.npy" % (test_name, epoch), fake_pca.cpu().detach().numpy()) 
                test_save_obj(fake_pca, "images/%s/epoch%d_"% (test_name, epoch), pca_model = '/home/yezipeng/talkingheadData/caricature_rewrite/npy_save/pca200_icp.model')
                

            
            
            

if __name__ == '__main__':
    opt = parser_set().parse_args()
    pca_cyclegan_main()
    
