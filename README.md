# 3D-CariGAN: An End-to-End Solution to 3D Caricature Generation from Normal Face Photos

This repository contains the source code and dataset for the paper 3D-CariGAN: An End-to-End Solution to 3D Caricature Generation from Normal Face Photos by [Yong-Jin Liu](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm).
 
This repository contains two parts: dataset and source code.

## 2D and 3D Caricature Dataset

### 2D Caricature Dataset

![2d_dataset](./fig/2d.jpg)

we collect 5,343 hand-drawn portrait caricature images from Pinterest.com and WebCaricature dataset with facial landmarks extracted by a landmark detector, followed by human interaction for correction if needed. 

### 3D Caricature Dataset

![3d_dataset](./fig/3d.jpg)

We use [the method](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Alive_Caricature_From_CVPR_2018_paper.pdf) to generate 5,343 3D caricature meshes of the same topology. We align the pose of the generated 3D caricature meshes with the pose of a template 3D head using an ICP method, where we use 5 key landmarks in eyes, nose and mouth as the landmarks for ICP. We normalize the coordinates of the 3D caricature mesh vertices by translating the center of meshes to the origin and scaling them to the same size.

### 3DCariPCA

### Download

You can download the two datasets and PCA in [google drive](https://drive.google.com/drive/folders/13lYYHOIQN_jJG5d-mBglD0BjWY1lqOWy?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1rtFtOeixNS1CACaZagrNLw) (code: 3kz8).

## Source Code

### Running Environment

### Training

### Testing

### Pre-trained Model

You can download pre-trained model ```latest.pth``` in [google drive](https://drive.google.com/drive/folders/13lYYHOIQN_jJG5d-mBglD0BjWY1lqOWy?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1rtFtOeixNS1CACaZagrNLw) (code: 3kz8). You should put it into ```./checkpoints```.