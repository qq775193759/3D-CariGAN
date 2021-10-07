import numpy as np
import torch
from sklearn.decomposition import PCA
import joblib
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
import cv2 
#from get_cari import *
import os
import math
import time
os.environ['CUDA_VISIBLE_DEVICES']='5'

delta_cos = 40

def my_linear_cos(x):
    y = x - 4*math.floor(x/4)
    z = abs(y-2)
    return  z - 1

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
    obj = obj.reshape(-1,3)
    return obj
    
def npy2obj_color(vertex, template_filename, out_name):
    f = open(template_filename)
    obj_f = f.read()
    f.close()
    v_str = ''
    for i in range(vertex.shape[0]):
        v_str = v_str + 'v ' + ' '.join(map(str, vertex[i]))  + '\n'
    obj_str = v_str + obj_f
    f = open(out_name, 'w')
    f.write(obj_str)
    f.close()
    
def npy2obj_vt(vertex, vt, template_filename, out_name, imgfile):
    f = open(template_filename)
    obj_f = f.read()
    f.close()
    imgfile_tail = imgfile.split('/')[-1]
    print(imgfile_tail)
    v_str = 'mtllib '+imgfile_tail+'.mtl\n'
    f = open(imgfile+'.mtl', 'w')
    mtl_str = 'map_Kd ' + imgfile_tail + '\n'
    f.write(mtl_str)
    f.close()
    for i in range(vertex.shape[0]):
        v_str = v_str + 'v ' + ' '.join(map(str, vertex[i]))  + '\n'
        v_str = v_str + 'vt ' + ' '.join(map(str, vt[i]))  + '\n'
    obj_str = v_str + obj_f
    f = open(out_name, 'w')
    f.write(obj_str)
    f.close()
    
    
def get_pos1024(x):
    res = int(x+0.5)
    res = min(res, 1023)
    res = max(res, 0)
    return res
    
def get_front_v(obj_norm, obj, num_v):
    obj2 = obj_norm[:,2]
    obj2min = obj2[5161]#face 2532  9854 5161
    v = obj2 < obj2min
    v_eye = v
    v_eye[11510:] = True
    #hair
    v_hair = v_eye.copy()
    obj1 = obj_norm[:,1]
    obj1min = obj1[7384]#hair 4328
    hair_v = obj1 < obj1min
    for i in range(0, num_v):
        if hair_v[i]:
            v_hair[i] = True
    #make vtx_border
    levels = 128
    dy = 8
    max_x = np.zeros(levels)
    min_x = np.ones(levels)*1024
    for i in range(0, num_v):
        if v_eye[i]:
            rank = int(get_pos1024(obj_norm[i,1])/dy)
            max_x[rank] = max(max_x[rank], obj_norm[i,0])
            min_x[rank] = min(min_x[rank], obj_norm[i,0])

    avg_x = (max_x+min_x)/2
    vtx_border = np.zeros(num_v)
    
    for i in range(0, num_v):
        rank = int(get_pos1024(obj_norm[i,1])/dy)
        if not v_hair[i]:
            if obj_norm[i,0] > avg_x[rank]:
                vtx_border[i] = max_x[rank] - delta_cos
                vtx_border[i] = vtx_border[i]*0.9 + obj_norm[i,0]*0.1
            else:
                vtx_border[i] = min_x[rank] + delta_cos
                vtx_border[i] = vtx_border[i]*0.9 + obj_norm[i,0]*0.1
            #vtx_border[i] = 0
        else:
            if obj_norm[i,0] > avg_x[rank]:
                vtx_border[i] = max_x[rank] - delta_cos
                if obj_norm[i,0] > (max_x[rank] - delta_cos) and obj_norm[i,0] > vtx_border[i]:
                    vtx_border[i] = vtx_border[i]*0.9 + obj_norm[i,0]*0.1
                else:
                    vtx_border[i] = obj_norm[i,0]
            else:
                vtx_border[i] = min_x[rank] + delta_cos
                if obj_norm[i,0] > (min_x[rank] + delta_cos) and obj_norm[i,0] < vtx_border[i]:
                    vtx_border[i] = vtx_border[i]*0.9 + obj_norm[i,0]*0.1
                else:
                    vtx_border[i] = obj_norm[i,0]
    return v_hair, vtx_border/1024

class cariPipeline():
    def __init__(self, model_path, pca_model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.pca = joblib.load(pca_model_path)
        transforms_ = [
            transforms.Resize(256, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(transforms_)
        f = open('obj_f.txt')
        self.obj_f = f.read()
        f.close()
        

    def cari_geometry_once(self, pic_name):
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        st = time.time()
        image = Image.open(pic_name)
        img = self.transform(image)
        img = Variable(img.type(Tensor)).reshape(1,3,256,256)
        #img = img.expand(64, 3, 256, 256)

        st2 = time.time()
        vec = self.model(img)
        en2 = time.time()
        vec = vec.cpu().detach().numpy()
        
        obj = self.pca.inverse_transform(vec).reshape(-1,11510, 3)
        en = time.time()
        print("total: ", en - st)
        print("gpu: ", en2 - st2)
        return obj

        
    def cari_texture_once(self, obj, obj2, imgfile, debugname, outname):
        img = cv2.imread(imgfile)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC) 
        num_v = obj.shape[0]
        color = []
        vt = []
        front_v_eye, vtx_border = get_front_v(obj, obj2, num_v)
        for i in range(0, num_v):
            vtx = vtx_border[i]
            vty = 1 - obj[i,1]/1024
            vt.append((vtx, vty))
        npy2obj_vt(obj2, vt, 'obj_f_eye_vt.txt', outname, imgfile)
        
        for i in range(0, num_v):
            pos =(get_pos1024(obj[i,0]), get_pos1024(obj[i,1]))
            cv2.circle(img, pos, 1, color=(139, 0, 0))
        cv2.imwrite(debugname, img)

    def lm2obj_once(self, rank, outname):
        lms = np.load('/home/yezipeng/talkingheadData/cariZhang_fullobj/celebA_lm_all.npy')
        landmark = lms[rank]
        get_cari(landmark.reshape(2*68), outname)
    
    def save_obj_once(self, obj_v, outname):
        v_str = '# 11510 vertices, 22800 faces\n'
        for v3 in obj_v:
            v_str = v_str + 'v ' + ' '.join(map(str, v3)) + '\n'
        obj_str = v_str + self.obj_f
        f = open(outname, 'w')
        f.write(obj_str)
        f.close()

def eye_complete(name1, name2):
    commend = "eye_complete/eye "+name1+" "+name2+" eye_complete/eye_rank.txt"
    os.system(commend)

def pipeline_once(pipeline, rank):
    obj_v = pipeline.cari_geometry_once('input/'+str(rank)+'.jpg')
    print(obj_v)
    cariname = 'debug/'+str(rank)+'.obj'
    cariname2 = 'debug/eye'+str(rank)+'.obj'
    pipeline.save_obj_once(obj_v[0,:,:], cariname)
    eye_complete(cariname, cariname2)
    
    return
    normalname = 'debug/normal_'+str(rank)+'.obj'
    normalname2 = 'debug/normaleye_'+str(rank)+'.obj'
    if os.path.exists(normalname):
        pass
    else:
        pipeline.lm2obj_once(rank, normalname)
    eye_complete(normalname, normalname2)
    obj = get_obj_file(normalname2)
    obj_v = get_obj_file(cariname2)
    pipeline.cari_texture_once(obj, obj_v, 'input/'+str(rank)+'.jpg', 'debug/'+str(rank)+'.png', 'output/'+str(rank)+'.obj')

def pipeline_main(pipeline):
    pipeline_once(pipeline, 4652)
    
    
    
def pipeline_test100(pipeline):
    for i in range(4650,4664):
        pipeline_once(pipeline, i)
    
    
if __name__ == '__main__':
    pipeline = cariPipeline('/home/yezipeng/talkinghead/caricature3D1to12/latest.pth', '/home/yezipeng/talkingheadData/cariZhang_fullobj/pca.model')
    #pipeline_main(pipeline)
    pipeline_test100(pipeline)
    