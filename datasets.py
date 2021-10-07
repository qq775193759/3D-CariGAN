import glob
import random
import os
import numpy as np
import joblib

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch

def get_obj_file(file):
    #print(file)
    f = open(file)
    obj = f.read()
    obj = obj.split('faces\nv ')[1]
    obj = obj.split('\nf', 1)[0]
    obj = obj.replace('\nv', '')
    obj = obj.split(' ')
    obj = list(map(float, obj))
    obj = np.array(obj)
    return obj
    
def get_tri_mesh(filename):
    with open(filename, 'r') as f:
        points, trilist = [], []
        for line in f:
            if line[:2] == 'v ':
                point = [float(x) for x in line.split()[1:4]]
                points.append(point)
            
        points = np.array(points)
    f.close()
    return points
    
def test_save_obj(pca_data, out_folder, pca_model = '/home/yezipeng/talkingheadData/caricature_rewrite/npy_save/pca200.model'):
    pca_model = joblib.load(pca_model)
    f = open('/home/yezipeng/talkingheadData/cariZhang_fullobj/obj_f.txt')
    obj_f = f.read()
    f.close()
    vertex = pca_model.inverse_transform(pca_data.cpu().detach().numpy())
    vertex = vertex.reshape(-1, 11510, 3)
    for i in range(0, vertex.shape[0]):
        obj_v = vertex[i]
        v_str = ''
        for v3 in obj_v:
            v_str = v_str + 'v ' + ' '.join(map(str, v3)) + '\n'
        obj_str = v_str + obj_f
        f = open(out_folder+str(i)+'.obj', 'w')
        f.write(obj_str)
        f.close()

def test_save_obj_by_rank(pca_data, img_rank, out_folder, pca_model = '/home/yezipeng/talkingheadData/caricature_rewrite/npy_save/pca200.model'):
    pca_model = joblib.load(pca_model)
    f = open('/home/yezipeng/talkingheadData/cariZhang_fullobj/obj_f.txt')
    obj_f = f.read()
    f.close()
    vertex = pca_model.inverse_transform(pca_data.cpu().detach().numpy())
    vertex = vertex.reshape(-1, 11510, 3)
    for i in range(0, vertex.shape[0]):
        obj_v = vertex[i]
        v_str = ''
        for v3 in obj_v:
            v_str = v_str + 'v ' + ' '.join(map(str, v3)) + '\n'
        obj_str = v_str + obj_f
        f = open(out_folder+'%d.obj'%(img_rank[i]), 'w')
        f.write(obj_str)
        f.close()
        
        
def test_save_obj_by_name(pca_data, img_name, out_folder, pca_model = '/home/yezipeng/talkingheadData/caricature_rewrite/npy_save/pca200.model'):
    pca_model = joblib.load(pca_model)
    f = open('/home/yezipeng/talkingheadData/cariZhang_fullobj/obj_f.txt')
    obj_f = f.read()
    f.close()
    vertex = pca_model.inverse_transform(pca_data.cpu().detach().numpy())
    vertex = vertex.reshape(-1, 11510, 3)
    for i in range(0, vertex.shape[0]):
        obj_v = vertex[i]
        v_str = ''
        for v3 in obj_v:
            v_str = v_str + 'v ' + ' '.join(map(str, v3)) + '\n'
        obj_str = v_str + obj_f
        f = open(out_folder+'%s.obj'%(img_name[i]), 'w')
        f.write(obj_str)
        f.close()
        
        
class IMG_filename_Dataset(Dataset):
    def __init__(self, obj_img_list_filename, img_transforms_=None):
        self.obj_img_list = np.load(obj_img_list_filename)
        self.transform = transforms.Compose(img_transforms_)

    def __getitem__(self, index):
        filename = self.obj_img_list[index][0]
        image = Image.open(filename)
        img = self.transform(image)
        return {"filename":filename, "img":img}

    def __len__(self):
        return self.obj_img_list.shape[0]
        
        
class PCA_IMG_OBJ_Dataset(Dataset):
    def __init__(self, pca_root, obj_img_list_filename, img_transforms_=None):
        self.pca_data = np.load(pca_root)
        self.obj_img_list = np.load(obj_img_list_filename)
        self.transform = transforms.Compose(img_transforms_)

    def __getitem__(self, index):
        image = Image.open(self.obj_img_list[index][0])
        img = self.transform(image)
        obj = get_obj_file(self.obj_img_list[index][1])
        pca = self.pca_data[random.randint(0, self.pca_data.shape[0]-1)]
        return {"pca":pca, "img":img, "obj":obj}

    def __len__(self):
        return self.obj_img_list.shape[0]
        
        
class PCA_IMG_OBJ_online_Dataset(Dataset):
    def __init__(self, cari_pca_name, img_root, head_npy_root, img_transforms_=None):
        self.transform = transforms.Compose(img_transforms_)
        self.img_root = img_root
        self.head_npy_root = head_npy_root
        self.headlist = glob.glob(head_npy_root+'/*.npy')
        self.pca_data = np.load(cari_pca_name)
        print(self.pca_data.shape)

    def __getitem__(self, index):
        rand_rank = torch.randint(0, self.pca_data.shape[0], (1,))[0]
        pca = self.pca_data[rand_rank]
        headname = self.headlist[index]
        imagename = headname.replace(self.head_npy_root, self.img_root).replace('.npy','.jpg')
        image = Image.open(imagename)
        img = self.transform(image)
        obj = np.load(headname)
        return {"pca":pca, "img":img, "obj":obj}

    def __len__(self):
        return len(self.headlist)


class IMG_Test_Dataset(Dataset):
    def __init__(self, img_root, img_transforms_=None, rank_st=0, rank_en=2000):
        self.transform = transforms.Compose(img_transforms_)
        self.img_root = img_root
        self.rank_st = rank_st
        self.rank_en = rank_en

    def __getitem__(self, index):
        img_rank = self.rank_st+index
        imagename = self.img_root+'/%d.jpg'%(img_rank)
        image = Image.open(imagename)
        img = self.transform(image)
        return {"img":img, "img_rank":img_rank}

    def __len__(self):
        return self.rank_en - self.rank_st
        
        
        
class IMG_Test_Dataset_name(Dataset):
    def __init__(self, img_root, img_transforms_=None):
        self.transform = transforms.Compose(img_transforms_)
        self.img_root = img_root
        self.img_list = glob.glob(img_root+'/*.jpg')

    def __getitem__(self, index):
        imagename = self.img_list[index]
        image = Image.open(imagename)
        img = self.transform(image)
        short_name = os.path.basename(imagename)[:-4]
        return {"img":img, "short_name":short_name}

    def __len__(self):
        return len(self.img_list)
        
        
