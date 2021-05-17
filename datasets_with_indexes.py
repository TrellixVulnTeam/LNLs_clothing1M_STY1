
from copy import deepcopy
from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from mean_teacher import data
import torch
import random
from data import cifar
from torch.utils.data import Dataset, DataLoader
from mean_teacher import data
from PIL import Image

p = Path(__file__).absolute()
PATH = p.parents[1]
DATA_PATH = PATH / 'data'


class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}    
        self.targets = []

        
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'all':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]   # dictionary
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples: #64000 개의 train_imgs를 확보 대신 class당 4571개씩 balanced하게
                    self.targets.append(label)
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)       
        elif self.mode == "labeled":   
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img1 = self.transform(image)
            img2 = self.transform(image)
            return (img1,img2), target, index        
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target,index
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target,index    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs) 

def clothing1M(final_run, train_size):
    num_classes=14
    transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
        ]) 
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
        ])        
    trainset = clothing_dataset('../data/clothing1M/data',transform=transform_train,mode='all',num_samples=train_size)
    valset = clothing_dataset('../data/clothing1M/data',transform=transform_test, mode='val')
    testset = clothing_dataset('../data/clothing1M/data',transform=transform_test, mode='test')
    


    return trainset, valset, testset, num_classes
 

def cifar10(final_run, val_size=5000):
    num_classes = 10
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023,  0.1994,  0.2010])
    transform_train = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = cifar.CIFAR10(root=DATA_PATH, train=True, download=True,
                                transform=transform_train)
    testset = cifar.CIFAR10(root=DATA_PATH, train=False, download=True,
                               transform=transform_test)
    if final_run:
        return trainset, testset, testset, num_classes
    valset = deepcopy(testset)
    X_train, X_val, y_train, y_val = train_test_split(trainset.data,
                                                      trainset.targets,
                                                      test_size=val_size,
                                                      stratify=trainset.targets)
    trainset.data, trainset.targets = X_train, y_train
    valset.data, valset.targets = X_val, y_val

    return trainset, valset, testset, num_classes


def cifar100(final_run, val_size=5000):
    num_classes = 100
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675,  0.2565,  0.2761])
    transform_train = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = cifar.CIFAR100(root=DATA_PATH, train=True, download=True,
                                 transform=transform_train)
    testset = cifar.CIFAR100(root=DATA_PATH, train=False, download=True,
                                transform=transform_test)
    if final_run:
        return trainset, testset, testset, num_classes
    valset = deepcopy(testset)
    X_train, X_val, y_train, y_val = train_test_split(trainset.data,
                                                      trainset.targets,
                                                      test_size=val_size,
                                                      stratify=trainset.targets)
    trainset.data, trainset.targets = X_train, y_train
    valset.data, valset.targets = X_val, y_val

    return trainset, valset, testset, num_classes