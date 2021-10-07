import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from sklearn.decomposition import PCA
import joblib


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, norm_layer):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            norm_layer(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            norm_layer(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            norm_layer(in_ch),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = x+self.conv(x)
        res = self.relu(res)
        return res


        
class SingleDown(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer):
        super(SingleDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class SingleUp(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

        
class Generator_IMG2PCA(nn.Module):
    def __init__(self, pca_dim = 200):
        super(Generator_IMG2PCA, self).__init__()
        vf_channels = [32, 64, 128 ,256, 512]
        
        norm_layer = nn.BatchNorm2d
        #norm_layer = nn.InstanceNorm2d
        
        self.inc = nn.Sequential(
            nn.Conv2d(3, vf_channels[0], 7, padding=3, bias=False),
            norm_layer(vf_channels[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        for i in range(len(vf_channels)-1):
            layers.append(SingleDown(vf_channels[i], vf_channels[i+1], norm_layer))
        for i in range(9):
            layers.append(ResNetBlock(vf_channels[-1], norm_layer))
        self.model = nn.Sequential(*layers)
        
        self.avgpool = nn.AvgPool2d(8, stride=8)
        self.fc = nn.Linear(2048, pca_dim)
            

    def forward(self, x):
        y = self.inc(x)
        y = self.model(y)
        y = self.avgpool(y)
        y = y.view(y.shape[0], -1)
        z = self.fc(y)
        return z
        
        



class Discriminator_PCA(nn.Module):
    def __init__(self, pca_dim = 200):
        super(Discriminator_PCA, self).__init__()
        norm_layer = nn.BatchNorm2d
        #norm_layer = nn.InstanceNorm2d
        self.model = nn.Sequential(
            nn.Linear(pca_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        y = self.model(x)
        return y
        
class Discriminator_PCA_for_BCE(nn.Module):
    def __init__(self, pca_dim = 200):
        super(Discriminator_PCA_for_BCE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(pca_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x)
        return y
        
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
        
class ResNet50_IMG2PCA(nn.Module):
    def __init__(self,blocks=[3,4,6,3], num_classes=200, expansion = 4):
        super(ResNet50_IMG2PCA,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 
        
        
        
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

def drift2mean(cari_obj):
    cari_obj = cari_obj.reshape(-1, 11510, 3)
    cari_obj_mean = cari_obj.mean(1).reshape(-1,1,3)
    cari_obj = cari_obj - cari_obj_mean
    cari_obj = cari_obj.reshape(-1, 34530)
    return cari_obj
       
class IDDloss_fullobj(nn.Module):
    def __init__(self, pca_model_name, mean_face_name):
        super(IDDloss_fullobj, self).__init__()

        pca_model = joblib.load(pca_model_name)
        pca_model_mean = pca_model.mean_.reshape(34530)
        pca_model_components = pca_model.components_.reshape(-1, 34530)
        
        self.pca_model_components = torch.from_numpy(pca_model_components).float().cuda()
        self.pca_model_mean = torch.from_numpy(pca_model_mean).float().cuda()
        
        mean_face = get_obj_file(mean_face_name)
        self.mean_face = torch.from_numpy(mean_face).float().cuda()
        self.mean_face = drift2mean(self.mean_face)
        
        
    def forward(self, objpca, norm_obj):
        cari_obj = objpca.mm(self.pca_model_components)+self.pca_model_mean
        cari_obj = drift2mean(cari_obj)
        norm_obj = drift2mean(norm_obj)
        
        
        cari_delta = cari_obj - self.mean_face
        norm_delta = norm_obj - self.mean_face
        
        cosab = torch.sum((cari_delta*norm_delta), 1)
        cosaa = torch.sum((cari_delta*cari_delta), 1)
        cosbb = torch.sum((norm_delta*norm_delta), 1)
        proja2b = cosab/cosbb
        proja2b = torch.exp(-proja2b)
        cosab = cosab/cosaa.sqrt()/cosbb.sqrt()
        #print(cosab, cosaa, cosbb)
        return 1-torch.mean(cosab), torch.mean(proja2b)
        
class IDDloss_fullobj_front(nn.Module):
    def __init__(self, pca_model_name, mean_face_name, front_part_v):
        super(IDDloss_fullobj_front, self).__init__()

        pca_model = joblib.load(pca_model_name)
        pca_model_mean = pca_model.mean_.reshape(34530)
        pca_model_components = pca_model.components_.reshape(-1, 34530)
        
        self.pca_model_components = torch.from_numpy(pca_model_components).float().cuda()
        self.pca_model_mean = torch.from_numpy(pca_model_mean).float().cuda()
        
        mean_face = get_obj_file(mean_face_name)
        self.mean_face = torch.from_numpy(mean_face).float().cuda()
        self.mean_face = drift2mean(self.mean_face)
        
        self.front_part_v = np.loadtxt(front_part_v)
        
        
    def forward(self, objpca, norm_obj):
        cari_obj = objpca.mm(self.pca_model_components)+self.pca_model_mean
        cari_obj = drift2mean(cari_obj)
        norm_obj = drift2mean(norm_obj)
        
        
        cari_delta = cari_obj - self.mean_face
        norm_delta = norm_obj - self.mean_face
        
        cari_delta = cari_delta[:, self.front_part_v]
        norm_delta = norm_delta[:, self.front_part_v]
        
        cosab = torch.sum((cari_delta*norm_delta), 1)
        cosaa = torch.sum((cari_delta*cari_delta), 1)
        cosbb = torch.sum((norm_delta*norm_delta), 1)
        proja2b = cosab/cosbb
        proja2b = torch.exp(-proja2b)
        cosab = cosab/cosaa.sqrt()/cosbb.sqrt()
        #print(cosab, cosaa, cosbb)
        return 1-torch.mean(cosab), torch.mean(proja2b)
        
        
        
class IDDloss_fullobj_front_seperate_mean(nn.Module):
    def __init__(self, pca_model_name, mean_face_name, front_part_v):
        super(IDDloss_fullobj_front_seperate_mean, self).__init__()

        pca_model = joblib.load(pca_model_name)
        pca_model_mean = pca_model.mean_.reshape(34530)
        pca_model_components = pca_model.components_.reshape(-1, 34530)
        
        self.pca_model_components = torch.from_numpy(pca_model_components).float().cuda()
        self.pca_model_mean = torch.from_numpy(pca_model_mean).float().cuda()
        
        mean_face = get_obj_file(mean_face_name)
        self.mean_face = torch.from_numpy(mean_face).float().cuda()
        self.mean_face = drift2mean(self.mean_face)
        
        self.front_part_v = np.loadtxt(front_part_v)
        
        
    def forward(self, objpca, norm_obj):
        cari_delta = objpca.mm(self.pca_model_components)
        norm_obj = drift2mean(norm_obj)
        
        
        norm_delta = norm_obj - self.mean_face
        
        cari_delta = cari_delta[:, self.front_part_v]
        norm_delta = norm_delta[:, self.front_part_v]
        
        cosab = torch.sum((cari_delta*norm_delta), 1)
        cosaa = torch.sum((cari_delta*cari_delta), 1)
        cosbb = torch.sum((norm_delta*norm_delta), 1)
        proja2b = cosab/cosbb
        proja2b = torch.exp(-proja2b)
        cosab = cosab/cosaa.sqrt()/cosbb.sqrt()
        #print(cosab, cosaa, cosbb)
        return 1-torch.mean(cosab), torch.mean(proja2b)

